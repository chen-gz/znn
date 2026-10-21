const std = @import("std");
const backend = @import("../backend.zig");
const mlx = backend.mlx;
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const module = @import("module.zig");

// ============================================================================
// Safetensors 零拷贝内存映射归档解析器 (Zero-Copy Memory-Mapped Safetensors Archive)
// ============================================================================

/// Safetensors 格式中的张量元数据 (Tensor Metadata)
pub const TensorMeta = struct {
    file_idx: u8, // 1 或 2
    start: usize,
    end: usize,
    shape: [8]usize,
    shape_len: u8,
    dtype: mlx.DType,
};

/// Safetensors 文件头解析与零拷贝内存映射句柄 (Memory-Mapped Safetensors Archive)
pub const SafetensorsArchive = struct {
    allocator: std.mem.Allocator,
    mmap1: []const u8,
    mmap2: []const u8,
    payload1: []const u8,
    payload2: []const u8,
    tensors: std.StringHashMap(TensorMeta),

    /// 打开包含双分片 (`model-00001-of-00002.safetensors` 与 `model-00002-of-00002.safetensors`) 的模型目录
    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !SafetensorsArchive {
        const file1_path = try std.fs.path.join(allocator, &.{ dir_path, "model-00001-of-00002.safetensors" });
        defer allocator.free(file1_path);
        const file2_path = try std.fs.path.join(allocator, &.{ dir_path, "model-00002-of-00002.safetensors" });
        defer allocator.free(file2_path);

        const cwd = std.Io.Dir.cwd();
        var f1 = try cwd.openFile(io, file1_path, .{});
        defer f1.close(io);
        const stat1 = try f1.stat(io);
        const mmap1 = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat1.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            f1.handle,
            0,
        );
        errdefer std.posix.munmap(@alignCast(mmap1));

        var f2 = try cwd.openFile(io, file2_path, .{});
        defer f2.close(io);
        const stat2 = try f2.stat(io);
        const mmap2 = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat2.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            f2.handle,
            0,
        );
        errdefer std.posix.munmap(@alignCast(mmap2));

        // 建议内核预读 (Kernel Pre-fetching via madvise)
        _ = std.posix.madvise(mmap1.ptr, mmap1.len, std.posix.MADV.WILLNEED) catch {};
        _ = std.posix.madvise(mmap2.ptr, mmap2.len, std.posix.MADV.WILLNEED) catch {};

        var tensors = std.StringHashMap(TensorMeta).init(allocator);
        errdefer tensors.deinit();
        try tensors.ensureTotalCapacity(800);

        const p1 = try parseHeader(allocator, mmap1, 1, &tensors);
        const p2 = try parseHeader(allocator, mmap2, 2, &tensors);

        return SafetensorsArchive{
            .allocator = allocator,
            .mmap1 = mmap1,
            .mmap2 = mmap2,
            .payload1 = p1,
            .payload2 = p2,
            .tensors = tensors,
        };
    }

    /// 打开单个 `.safetensors` 权重文件并构建零拷贝内存映射索引
    pub fn openSingleFile(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !SafetensorsArchive {
        const cwd = std.Io.Dir.cwd();
        var f1 = try cwd.openFile(io, file_path, .{});
        defer f1.close(io);
        const stat1 = try f1.stat(io);
        const mmap1 = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat1.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            f1.handle,
            0,
        );
        errdefer std.posix.munmap(@alignCast(mmap1));

        _ = std.posix.madvise(mmap1.ptr, mmap1.len, std.posix.MADV.WILLNEED) catch {};

        var tensors = std.StringHashMap(TensorMeta).init(allocator);
        errdefer tensors.deinit();
        try tensors.ensureTotalCapacity(512);

        const p1 = try parseHeader(allocator, mmap1, 1, &tensors);

        return SafetensorsArchive{
            .allocator = allocator,
            .mmap1 = mmap1,
            .mmap2 = &.{},
            .payload1 = p1,
            .payload2 = &.{},
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *SafetensorsArchive) void {
        var it = self.tensors.keyIterator();
        while (it.next()) |k| {
            self.allocator.free(k.*);
        }
        self.tensors.deinit();
        if (self.mmap1.len > 0) std.posix.munmap(@alignCast(self.mmap1));
        if (self.mmap2.len > 0) std.posix.munmap(@alignCast(self.mmap2));
    }

    fn parseHeader(
        allocator: std.mem.Allocator,
        mmap_slice: []const u8,
        file_idx: u8,
        tensors: *std.StringHashMap(TensorMeta),
    ) ![]const u8 {
        if (mmap_slice.len < 8) return error.InvalidSafetensorsHeader;
        const header_len = std.mem.readInt(u64, mmap_slice[0..8], .little);
        if (8 + header_len > mmap_slice.len) return error.InvalidSafetensorsHeader;

        const header_bytes = mmap_slice[8 .. 8 + header_len];
        const payload = mmap_slice[8 + header_len ..];

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        if (parsed.value != .object) return error.InvalidSafetensorsHeader;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            if (entry.value_ptr.* != .object) continue;
            const obj = entry.value_ptr.*.object;

            const offsets_v = obj.get("data_offsets") orelse continue;
            if (offsets_v != .array or offsets_v.array.items.len != 2) continue;
            const start: usize = @intCast(offsets_v.array.items[0].integer);
            const end: usize = @intCast(offsets_v.array.items[1].integer);

            const dtype_v = obj.get("dtype") orelse continue;
            var dt: mlx.DType = .float32;
            if (dtype_v == .string) {
                if (std.mem.eql(u8, dtype_v.string, "U32")) {
                    dt = .uint32;
                } else if (std.mem.eql(u8, dtype_v.string, "BF16")) {
                    dt = .bfloat16;
                } else if (std.mem.eql(u8, dtype_v.string, "F16")) {
                    dt = .float16;
                } else if (std.mem.eql(u8, dtype_v.string, "F32")) {
                    dt = .float32;
                }
            }

            var shape: [8]usize = undefined;
            var shape_len: u8 = 0;
            if (obj.get("shape")) |sv| {
                if (sv == .array and sv.array.items.len <= 8) {
                    for (sv.array.items, 0..) |dim_v, d_i| {
                        shape[d_i] = @intCast(dim_v.integer);
                    }
                    shape_len = @intCast(sv.array.items.len);
                }
            }

            const key_dup = try allocator.dupe(u8, entry.key_ptr.*);
            try tensors.put(key_dup, .{
                .file_idx = file_idx,
                .start = start,
                .end = end,
                .shape = shape,
                .shape_len = shape_len,
                .dtype = dt,
            });
        }

        return payload;
    }

    pub fn getTensorBytes(self: *const SafetensorsArchive, name: []const u8) ?[]const u8 {
        const meta = self.tensors.get(name) orelse return null;
        const payload = if (meta.file_idx == 1) self.payload1 else self.payload2;
        if (meta.end > payload.len) return null;
        return payload[meta.start..meta.end];
    }

    pub fn getTensorMeta(self: *const SafetensorsArchive, name: []const u8) ?TensorMeta {
        return self.tensors.get(name);
    }
};

// ============================================================================
// Safetensors 格式模型权重序列化与反序列化
// ============================================================================

pub fn writeTensorEntry(
    json_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    name: []const u8,
    tensor_ptr: *const Tensor,
    offset: *usize,
) !void {
    const size_bytes = tensor_ptr.data.len * 4;
    const start = offset.*;
    const end = start + size_bytes;
    offset.* = end;

    try json_buf.print(allocator, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{name});
    for (0..tensor_ptr.shape.len) |i| {
        if (i > 0) try json_buf.appendSlice(allocator, ",");
        try json_buf.print(allocator, "{}", .{tensor_ptr.shape.dims[i]});
    }
    try json_buf.print(allocator, "],\"data_offsets\":[{},{}]}}", .{ start, end });
}

/// 按字段路径写入每个张量的 Safetensors 头部条目 (键与 `nn.namedParameters` 一致，含缓冲区)
const HeaderWriter = struct {
    json_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    offset: usize = 0,
    first: bool = true,

    pub const wants_path = true;

    pub fn visitTensor(self: *HeaderWriter, path: []const u8, t: *Tensor) !void {
        if (!self.first) try self.json_buf.appendSlice(self.allocator, ",") else self.first = false;
        try writeTensorEntry(self.json_buf, self.allocator, path, t, &self.offset);
    }
};

/// 按与 `HeaderWriter` 相同的遍历顺序写入张量二进制数据
fn DataWriter(comptime Writer: type) type {
    return struct {
        writer: Writer,

        pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
            try self.writer.writeAll(std.mem.sliceAsBytes(t.data));
        }
    };
}

pub fn saveModel(model: anytype, io: std.Io, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFile(io, file_path, .{});
    defer file.close(io);

    // 1. 构建 JSON header
    var json_buf: std.ArrayList(u8) = .empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{");

    var header = HeaderWriter{ .json_buf = &json_buf, .allocator = allocator };
    try module.walk(model, &header);
    try json_buf.appendSlice(allocator, "}");

    // 2. 对齐 JSON 头部长度至 8 字节的倍数（Safetensors 标准）
    const header_len_unpadded = json_buf.items.len;
    const padding = (8 - (header_len_unpadded % 8)) % 8;
    for (0..padding) |_| {
        try json_buf.append(allocator, ' ');
    }
    const final_header_len = json_buf.items.len;

    // 3. 写入 8 字节 of header 长度（小端序 u64）和 header json 字节
    var buf: [65536]u8 = undefined;
    var file_writer = file.writer(io, &buf);
    const writer = &file_writer.interface;

    const header_len_u64 = @as(u64, final_header_len);
    try writer.writeAll(std.mem.asBytes(&header_len_u64));
    try writer.writeAll(json_buf.items);

    // 4. 顺序写入张量二进制权重数据
    var data_writer = DataWriter(@TypeOf(writer)){ .writer = writer };
    try module.walk(model, &data_writer);
    try writer.flush();
}

/// 按字段路径从 Safetensors 数据载荷中还原每个张量
const TensorLoader = struct {
    data_payload: []const u8,
    meta_obj: std.json.ObjectMap,

    pub const wants_path = true;

    pub fn visitTensor(self: *TensorLoader, path: []const u8, t: *Tensor) !void {
        try loadTensorData(self.data_payload, self.meta_obj, path, t);
    }
};

pub fn loadTensorData(
    data_payload: []const u8,
    meta_obj: anytype,
    name: []const u8,
    dest: *Tensor,
) !void {
    const tensor_meta_val = meta_obj.get(name) orelse {
        return error.TensorNotFound;
    };
    if (tensor_meta_val != .object) return error.InvalidSafetensorsHeader;
    const tensor_meta = tensor_meta_val.object;

    // 校验数据类型并确定单个元素字节宽度
    const dtype_val = tensor_meta.get("dtype") orelse return error.InvalidSafetensorsHeader;
    if (dtype_val != .string) return error.InvalidSafetensorsHeader;
    const dtype_str = dtype_val.string;

    const elem_size: usize = if (std.mem.eql(u8, dtype_str, "F32"))
        4
    else if (std.mem.eql(u8, dtype_str, "BF16") or std.mem.eql(u8, dtype_str, "F16"))
        2
    else
        return error.UnsupportedDtype;

    // 校验逻辑形状
    const shape_val = tensor_meta.get("shape") orelse return error.InvalidSafetensorsHeader;
    if (shape_val != .array) return error.InvalidSafetensorsHeader;
    const shape_arr = shape_val.array;
    if (shape_arr.items.len != dest.shape.len) {
        return error.ShapeMismatch;
    }
    for (0..dest.shape.len) |i| {
        const dim_val = shape_arr.items[i];
        if (dim_val != .integer or dim_val.integer < 0 or @as(usize, @intCast(dim_val.integer)) != dest.shape.dims[i]) {
            return error.ShapeMismatch;
        }
    }

    // 校验偏移量并按偏移量随机访问读取（无需强求物理顺序与结构体字段顺序一致）
    const offsets_val = tensor_meta.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
    if (offsets_val != .array or offsets_val.array.items.len != 2) return error.InvalidSafetensorsHeader;
    if (offsets_val.array.items[0] != .integer or offsets_val.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
    if (offsets_val.array.items[0].integer < 0 or offsets_val.array.items[1].integer < offsets_val.array.items[0].integer) {
        return error.InvalidSafetensorsHeader;
    }
    const start_offset = @as(usize, @intCast(offsets_val.array.items[0].integer));
    const end_offset = @as(usize, @intCast(offsets_val.array.items[1].integer));

    const expected_len_bytes = dest.data.len * elem_size;
    if (end_offset - start_offset != expected_len_bytes) {
        return error.SizeMismatch;
    }
    if (end_offset > data_payload.len) {
        return error.UnexpectedEndOfStream;
    }

    const raw_slice = data_payload[start_offset..end_offset];
    if (std.mem.eql(u8, dtype_str, "F32")) {
        @memcpy(std.mem.sliceAsBytes(dest.data), raw_slice);
    } else if (std.mem.eql(u8, dtype_str, "BF16")) {
        const u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw_slice));
        for (dest.data, u16_slice) |*d, b_bits| {
            const val: tensor.bf16 = .{ .bits = b_bits };
            d.* = val.toF32();
        }
    } else if (std.mem.eql(u8, dtype_str, "F16")) {
        const f16_slice: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw_slice));
        for (dest.data, f16_slice) |*d, f_val| {
            d.* = @floatCast(f_val);
        }
    }
}

pub fn loadModel(model: anytype, io: std.Io, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    var buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &buf);
    const reader = &file_reader.interface;

    // 1. 读取 8 字节 header 长度
    var temp_8: [8]u8 = undefined;
    try reader.readSliceAll(&temp_8);
    const header_len = std.mem.readInt(u64, &temp_8, .little);

    // 2. 读取 JSON 头部
    const header_buf = try allocator.alloc(u8, header_len);
    defer allocator.free(header_buf);
    try reader.readSliceAll(header_buf);

    // 3. 解析 JSON
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    const meta_obj = parsed.value.object;

    // 4. 计算数据区最大偏移并读取二进制数据载荷，支持乱序/任意偏移量读取
    var max_end_offset: usize = 0;
    var it = meta_obj.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        if (entry.value_ptr.* != .object) return error.InvalidSafetensorsHeader;
        if (entry.value_ptr.object.get("data_offsets")) |offsets_val| {
            if (offsets_val != .array or offsets_val.array.items.len != 2) return error.InvalidSafetensorsHeader;
            if (offsets_val.array.items[0] != .integer or offsets_val.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
            if (offsets_val.array.items[0].integer < 0 or offsets_val.array.items[1].integer < offsets_val.array.items[0].integer) {
                return error.InvalidSafetensorsHeader;
            }
            const end_off: usize = @intCast(offsets_val.array.items[1].integer);
            if (end_off > max_end_offset) max_end_offset = end_off;
        }
    }

    const data_payload = try allocator.alloc(u8, max_end_offset);
    defer allocator.free(data_payload);
    try reader.readSliceAll(data_payload);

    // 5. 按偏移量还原每一个 Tensor 字段
    var loader = TensorLoader{ .data_payload = data_payload, .meta_obj = meta_obj };
    try module.walk(model, &loader);
}

