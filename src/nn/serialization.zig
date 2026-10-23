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

/// Safetensors 头部长度上限 (与参考实现一致)，防止恶意长度导致整数溢出或超大分配
const max_header_len: u64 = 100 * 1024 * 1024;

/// Safetensors 数据类型字符串到后端数据类型与单元素字节宽度的映射
const DTypeInfo = struct { dtype: mlx.DType, size: usize };
const safetensors_dtypes = std.StaticStringMap(DTypeInfo).initComptime(.{
    .{ "BOOL", DTypeInfo{ .dtype = .bool, .size = 1 } },
    .{ "U8", DTypeInfo{ .dtype = .uint8, .size = 1 } },
    .{ "I8", DTypeInfo{ .dtype = .int8, .size = 1 } },
    .{ "U16", DTypeInfo{ .dtype = .uint16, .size = 2 } },
    .{ "I16", DTypeInfo{ .dtype = .int16, .size = 2 } },
    .{ "F16", DTypeInfo{ .dtype = .float16, .size = 2 } },
    .{ "BF16", DTypeInfo{ .dtype = .bfloat16, .size = 2 } },
    .{ "U32", DTypeInfo{ .dtype = .uint32, .size = 4 } },
    .{ "I32", DTypeInfo{ .dtype = .int32, .size = 4 } },
    .{ "F32", DTypeInfo{ .dtype = .float32, .size = 4 } },
    .{ "U64", DTypeInfo{ .dtype = .uint64, .size = 8 } },
    .{ "I64", DTypeInfo{ .dtype = .int64, .size = 8 } },
    .{ "F64", DTypeInfo{ .dtype = .float64, .size = 8 } },
});

/// 读取 JSON 非负整数并转换为 usize，其他取值视为损坏的头部
fn jsonUsize(v: std.json.Value) !usize {
    if (v != .integer or v.integer < 0) return error.InvalidSafetensorsHeader;
    return std.math.cast(usize, v.integer) orelse error.InvalidSafetensorsHeader;
}

const DataOffsets = struct { start: usize, end: usize };

/// 解析张量条目的 `data_offsets: [start, end]`，要求两者为非负整数且 start <= end
fn parseDataOffsets(obj: std.json.ObjectMap) !DataOffsets {
    const v = obj.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
    if (v != .array or v.array.items.len != 2) return error.InvalidSafetensorsHeader;
    const start = try jsonUsize(v.array.items[0]);
    const end = try jsonUsize(v.array.items[1]);
    if (end < start) return error.InvalidSafetensorsHeader;
    return .{ .start = start, .end = end };
}

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
        errdefer freeTensorIndex(allocator, &tensors);
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
        errdefer freeTensorIndex(allocator, &tensors);
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
        freeTensorIndex(self.allocator, &self.tensors);
        if (self.mmap1.len > 0) std.posix.munmap(@alignCast(self.mmap1));
        if (self.mmap2.len > 0) std.posix.munmap(@alignCast(self.mmap2));
    }

    /// 释放张量索引及其持有的键字符串
    fn freeTensorIndex(allocator: std.mem.Allocator, tensors: *std.StringHashMap(TensorMeta)) void {
        var it = tensors.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        tensors.deinit();
    }

    /// 解析单个分片的 JSON 头部并登记其全部张量，返回数据载荷切片。
    /// 每个条目都经过完整校验：偏移量为非负整数且 start <= end <= 载荷长度，形状维度为非负整数，
    /// 字节数与 `形状元素数 * 元素宽度` 一致；无法识别的数据类型 (如 8 位浮点) 不登记；
    /// 同名张量 (含跨分片) 视为损坏的归档。
    fn parseHeader(
        allocator: std.mem.Allocator,
        mmap_slice: []const u8,
        file_idx: u8,
        tensors: *std.StringHashMap(TensorMeta),
    ) ![]const u8 {
        if (mmap_slice.len < 8) return error.InvalidSafetensorsHeader;
        const header_len = std.mem.readInt(u64, mmap_slice[0..8], .little);
        if (header_len > max_header_len or header_len > mmap_slice.len - 8) return error.InvalidSafetensorsHeader;
        const header_end: usize = @intCast(8 + header_len);

        const header_bytes = mmap_slice[8..header_end];
        const payload = mmap_slice[header_end..];

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        if (parsed.value != .object) return error.InvalidSafetensorsHeader;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            if (entry.value_ptr.* != .object) return error.InvalidSafetensorsHeader;
            const obj = entry.value_ptr.*.object;

            const dtype_v = obj.get("dtype") orelse return error.InvalidSafetensorsHeader;
            if (dtype_v != .string) return error.InvalidSafetensorsHeader;
            const dtype_info = safetensors_dtypes.get(dtype_v.string) orelse continue;

            const offsets = try parseDataOffsets(obj);
            if (offsets.end > payload.len) return error.InvalidSafetensorsHeader;

            var shape: [8]usize = @splat(0);
            var shape_len: u8 = 0;
            var numel: usize = 1;
            if (obj.get("shape")) |sv| {
                if (sv != .array or sv.array.items.len > shape.len) return error.InvalidSafetensorsHeader;
                for (sv.array.items, 0..) |dim_v, d_i| {
                    shape[d_i] = try jsonUsize(dim_v);
                    numel = std.math.mul(usize, numel, shape[d_i]) catch return error.InvalidSafetensorsHeader;
                }
                shape_len = @intCast(sv.array.items.len);
            }
            const expected_bytes = std.math.mul(usize, numel, dtype_info.size) catch return error.InvalidSafetensorsHeader;
            if (offsets.end - offsets.start != expected_bytes) return error.InvalidSafetensorsHeader;

            const gop = try tensors.getOrPut(entry.key_ptr.*);
            if (gop.found_existing) return error.DuplicateTensorName;
            gop.key_ptr.* = allocator.dupe(u8, entry.key_ptr.*) catch |err| {
                tensors.removeByPtr(gop.key_ptr);
                return err;
            };
            gop.value_ptr.* = .{
                .file_idx = file_idx,
                .start = offsets.start,
                .end = offsets.end,
                .shape = shape,
                .shape_len = shape_len,
                .dtype = dtype_info.dtype,
            };
        }

        return payload;
    }

    /// 张量原始字节 (解析时已校验 start <= end <= 载荷长度)
    pub fn getTensorBytes(self: *const SafetensorsArchive, name: []const u8) ?[]const u8 {
        const meta = self.tensors.get(name) orelse return null;
        const payload = if (meta.file_idx == 1) self.payload1 else self.payload2;
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
    const offsets = try parseDataOffsets(tensor_meta);
    const expected_len_bytes = dest.data.len * elem_size;
    if (offsets.end - offsets.start != expected_len_bytes) {
        return error.SizeMismatch;
    }
    if (offsets.end > data_payload.len) {
        return error.UnexpectedEndOfStream;
    }

    // 载荷缓冲区与张量偏移量不保证按元素宽度对齐，半精度数据逐元素按小端序读取
    const raw_slice = data_payload[offsets.start..offsets.end];
    if (elem_size == 4) {
        @memcpy(std.mem.sliceAsBytes(dest.data), raw_slice);
    } else if (std.mem.eql(u8, dtype_str, "BF16")) {
        for (dest.data, 0..) |*d, i| {
            const val: tensor.bf16 = .{ .bits = std.mem.readInt(u16, raw_slice[2 * i ..][0..2], .little) };
            d.* = val.toF32();
        }
    } else {
        for (dest.data, 0..) |*d, i| {
            const bits = std.mem.readInt(u16, raw_slice[2 * i ..][0..2], .little);
            d.* = @floatCast(@as(f16, @bitCast(bits)));
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
    if (header_len > max_header_len) return error.InvalidSafetensorsHeader;

    // 2. 读取 JSON 头部
    const header_buf = try allocator.alloc(u8, @intCast(header_len));
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
        if (!entry.value_ptr.object.contains("data_offsets")) continue;
        const offsets = try parseDataOffsets(entry.value_ptr.object);
        max_end_offset = @max(max_end_offset, offsets.end);
    }

    const data_payload = try allocator.alloc(u8, max_end_offset);
    defer allocator.free(data_payload);
    try reader.readSliceAll(data_payload);

    // 5. 按偏移量还原每一个 Tensor 字段
    var loader = TensorLoader{ .data_payload = data_payload, .meta_obj = meta_obj };
    try module.walk(model, &loader);
}
