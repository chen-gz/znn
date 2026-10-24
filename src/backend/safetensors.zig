//! Safetensors 零拷贝内存映射归档解析器 (Zero-Copy Memory-Mapped Safetensors Archive)
//!
//! 在 Layer 0 (backend) 提供底层轻量级 Safetensors 二进制文件解析，供硬件离线加载与权重直接绑定。

const std = @import("std");
const mlx = @import("mlx.zig");

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
pub const max_header_len: u64 = 100 * 1024 * 1024;

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

pub const DataOffsets = struct { start: usize, end: usize };

/// 解析张量条目的 `data_offsets: [start, end]`，要求两者为非负整数且 start <= end
pub fn parseDataOffsets(obj: std.json.ObjectMap) !DataOffsets {
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

    /// 解析单个分片的 JSON 头部并登记其全部张量，返回数据载荷切片
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
