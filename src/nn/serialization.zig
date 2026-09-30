const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;

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

pub fn writeModelTensors(
    model: anytype,
    json_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    offset: *usize,
    first: *bool,
    prefix: []const u8,
) anyerror!void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            var name: std.ArrayList(u8) = .empty;
            defer name.deinit(allocator);
            if (prefix.len > 0) {
                try name.appendSlice(allocator, prefix);
                try name.appendSlice(allocator, ".");
            }
            try name.appendSlice(allocator, field.name);

            if (!first.*) try json_buf.appendSlice(allocator, ",") else first.* = false;
            try writeTensorEntry(json_buf, allocator, name.items, @field(model, field.name), offset);
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |tensor_ptr| {
                var name: std.ArrayList(u8) = .empty;
                defer name.deinit(allocator);
                if (prefix.len > 0) {
                    try name.appendSlice(allocator, prefix);
                    try name.appendSlice(allocator, ".");
                }
                try name.appendSlice(allocator, field.name);

                if (!first.*) try json_buf.appendSlice(allocator, ",") else first.* = false;
                try writeTensorEntry(json_buf, allocator, name.items, tensor_ptr, offset);
            }
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name), 0..) |tensor_ptr, idx| {
                    var name: std.ArrayList(u8) = .empty;
                    defer name.deinit(allocator);
                    if (prefix.len > 0) {
                        try name.appendSlice(allocator, prefix);
                        try name.appendSlice(allocator, ".");
                    }
                    try name.print(allocator, "{s}.{d}", .{ field.name, idx });

                    if (!first.*) try json_buf.appendSlice(allocator, ",") else first.* = false;
                    try writeTensorEntry(json_buf, allocator, name.items, tensor_ptr, offset);
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name), 0..) |*item, idx| {
                    var next_prefix: std.ArrayList(u8) = .empty;
                    defer next_prefix.deinit(allocator);
                    if (prefix.len > 0) {
                        try next_prefix.appendSlice(allocator, prefix);
                        try next_prefix.appendSlice(allocator, ".");
                    }
                    try next_prefix.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try writeModelTensors(item, json_buf, allocator, offset, first, next_prefix.items);
                }
            }
        } else if (field_info == .@"struct") {
            var next_prefix: std.ArrayList(u8) = .empty;
            defer next_prefix.deinit(allocator);
            if (prefix.len > 0) {
                try next_prefix.appendSlice(allocator, prefix);
                try next_prefix.appendSlice(allocator, ".");
            }
            try next_prefix.appendSlice(allocator, field.name);
            try writeModelTensors(&@field(model, field.name), json_buf, allocator, offset, first, next_prefix.items);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name), 0..) |tensor_ptr, idx| {
                    var name: std.ArrayList(u8) = .empty;
                    defer name.deinit(allocator);
                    if (prefix.len > 0) {
                        try name.appendSlice(allocator, prefix);
                        try name.appendSlice(allocator, ".");
                    }
                    try name.print(allocator, "{s}.{d}", .{ field.name, idx });

                    if (!first.*) try json_buf.appendSlice(allocator, ",") else first.* = false;
                    try writeTensorEntry(json_buf, allocator, name.items, tensor_ptr, offset);
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name), 0..) |*item, idx| {
                    var next_prefix: std.ArrayList(u8) = .empty;
                    defer next_prefix.deinit(allocator);
                    if (prefix.len > 0) {
                        try next_prefix.appendSlice(allocator, prefix);
                        try next_prefix.appendSlice(allocator, ".");
                    }
                    try next_prefix.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try writeModelTensors(item, json_buf, allocator, offset, first, next_prefix.items);
                }
            }
        }
    }
}

pub fn writeModelData(
    model: anytype,
    writer: anytype,
) anyerror!void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            try writer.writeAll(std.mem.sliceAsBytes(@field(model, field.name).data));
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |tensor_ptr| {
                try writer.writeAll(std.mem.sliceAsBytes(tensor_ptr.data));
            }
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |tensor_ptr| {
                    try writer.writeAll(std.mem.sliceAsBytes(tensor_ptr.data));
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| {
                    try writeModelData(item, writer);
                }
            }
        } else if (field_info == .@"struct") {
            try writeModelData(&@field(model, field.name), writer);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |tensor_ptr| {
                    try writer.writeAll(std.mem.sliceAsBytes(tensor_ptr.data));
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    try writeModelData(item, writer);
                }
            }
        }
    }
}

pub fn saveModel(model: anytype, io: std.Io, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFile(io, file_path, .{});
    defer file.close(io);

    // 1. 构建 JSON header
    var json_buf: std.ArrayList(u8) = .empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{");
    var first = true;
    var offset: usize = 0;

    try writeModelTensors(model, &json_buf, allocator, &offset, &first, "");
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
    try writeModelData(model, writer);
    try writer.flush();
}

pub fn loadModelTensors(
    model: anytype,
    data_payload: []const u8,
    meta_obj: anytype,
    allocator: std.mem.Allocator,
    prefix: []const u8,
) anyerror!void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            var name: std.ArrayList(u8) = .empty;
            defer name.deinit(allocator);
            if (prefix.len > 0) {
                try name.appendSlice(allocator, prefix);
                try name.appendSlice(allocator, ".");
            }
            try name.appendSlice(allocator, field.name);
            try loadTensorData(data_payload, meta_obj, name.items, @field(model, field.name));
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |tensor_ptr| {
                var name: std.ArrayList(u8) = .empty;
                defer name.deinit(allocator);
                if (prefix.len > 0) {
                    try name.appendSlice(allocator, prefix);
                    try name.appendSlice(allocator, ".");
                }
                try name.appendSlice(allocator, field.name);
                try loadTensorData(data_payload, meta_obj, name.items, tensor_ptr);
            }
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name), 0..) |tensor_ptr, idx| {
                    var name: std.ArrayList(u8) = .empty;
                    defer name.deinit(allocator);
                    if (prefix.len > 0) {
                        try name.appendSlice(allocator, prefix);
                        try name.appendSlice(allocator, ".");
                    }
                    try name.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try loadTensorData(data_payload, meta_obj, name.items, tensor_ptr);
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name), 0..) |*item, idx| {
                    var next_prefix: std.ArrayList(u8) = .empty;
                    defer next_prefix.deinit(allocator);
                    if (prefix.len > 0) {
                        try next_prefix.appendSlice(allocator, prefix);
                        try next_prefix.appendSlice(allocator, ".");
                    }
                    try next_prefix.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try loadModelTensors(item, data_payload, meta_obj, allocator, next_prefix.items);
                }
            }
        } else if (field_info == .@"struct") {
            var next_prefix: std.ArrayList(u8) = .empty;
            defer next_prefix.deinit(allocator);
            if (prefix.len > 0) {
                try next_prefix.appendSlice(allocator, prefix);
                try next_prefix.appendSlice(allocator, ".");
            }
            try next_prefix.appendSlice(allocator, field.name);
            try loadModelTensors(&@field(model, field.name), data_payload, meta_obj, allocator, next_prefix.items);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name), 0..) |tensor_ptr, idx| {
                    var name: std.ArrayList(u8) = .empty;
                    defer name.deinit(allocator);
                    if (prefix.len > 0) {
                        try name.appendSlice(allocator, prefix);
                        try name.appendSlice(allocator, ".");
                    }
                    try name.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try loadTensorData(data_payload, meta_obj, name.items, tensor_ptr);
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name), 0..) |*item, idx| {
                    var next_prefix: std.ArrayList(u8) = .empty;
                    defer next_prefix.deinit(allocator);
                    if (prefix.len > 0) {
                        try next_prefix.appendSlice(allocator, prefix);
                        try next_prefix.appendSlice(allocator, ".");
                    }
                    try next_prefix.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try loadModelTensors(item, data_payload, meta_obj, allocator, next_prefix.items);
                }
            }
        }
    }
}

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

    // 校验数据类型
    const dtype_val = tensor_meta.get("dtype") orelse return error.InvalidSafetensorsHeader;
    if (dtype_val != .string or !std.mem.eql(u8, dtype_val.string, "F32")) {
        return error.UnsupportedDtype;
    }

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

    const expected_len_bytes = dest.data.len * 4;
    if (end_offset - start_offset != expected_len_bytes) {
        return error.SizeMismatch;
    }
    if (end_offset > data_payload.len) {
        return error.UnexpectedEndOfStream;
    }

    @memcpy(std.mem.sliceAsBytes(dest.data), data_payload[start_offset..end_offset]);
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
    try loadModelTensors(model, data_payload, meta_obj, allocator, "");
}

