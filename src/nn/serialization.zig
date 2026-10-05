const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const module = @import("module.zig");

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

