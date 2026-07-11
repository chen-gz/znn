const std = @import("std");
const autodiff = @import("autodiff.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

// ============================================================================
// 1. PyTorch-like Linear (全连接/线性层) 模块定义
// ============================================================================
pub const Linear = struct {
    weight: *Tensor,   // 权重矩阵
    bias: *Tensor,     // 偏置向量
    v_weight: []f32,            // 权重动量缓存
    v_bias: []f32,              // 偏置动量缓存

    // 初始化一个线性层，自动生成对应的持久化 Tensor 和动量缓冲区，并进行 He 参数初始化
    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, random: std.Random) !Linear {
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        const v_weight = try allocator.alloc(f32, in_features * out_features);
        errdefer allocator.free(v_weight);
        const v_bias = try allocator.alloc(f32, out_features);
        errdefer allocator.free(v_bias);

        @memset(v_weight, 0.0);
        @memset(v_bias, 0.0);

        // 使用 He (Kaiming) 归一化方法初始化权重，偏置设为 0
        initializeWeights(random, weight.data, in_features);
        @memset(bias.data, 0.0);

        return Linear{
            .weight = weight,
            .bias = bias,
            .v_weight = v_weight,
            .v_bias = v_bias,
        };
    }

    // 释放该层持有的所有持久化数据与动量缓存内存
    pub fn deinit(self: Linear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
        allocator.free(self.v_weight);
        allocator.free(self.v_bias);
    }

    // 将本层的梯度设为 0
    pub fn zeroGrad(self: Linear) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

    // 利用带 Momentum 的 SGD 算法更新本层的权重和偏置值
    pub fn updateWeights(self: Linear, lr: f32, beta: f32) void {
        updateLayerWeights(self.weight.data, self.weight.grad, self.v_weight, lr, beta);
        updateLayerWeights(self.bias.data, self.bias.grad, self.v_bias, lr, beta);
    }

    // 实现前向计算链路：Y = X * W + b
    pub fn forward(self: Linear, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const z = try x.matmul(self.weight, allocator, graph);
        if (graph == null) {
            defer tensor.free(allocator, z);
            return try z.addBias(self.bias, allocator, null);
        }
        return try z.addBias(self.bias, allocator, graph);
    }
};

pub const Conv2D = struct {
    weight: *Tensor,
    bias: *Tensor,
    v_weight: []f32,
    v_bias: []f32,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, random: std.Random) !Conv2D {
        const weight_size = out_channels * in_channels * kernel_size * kernel_size;
        const bias_size = out_channels;

        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        // Correct the shape of Conv2D weight to [out_channels, in_channels, kernel_size, kernel_size]
        weight.shape = Shape.init(&.{out_channels, in_channels, kernel_size, kernel_size});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, out_channels, true);
        errdefer freePersistentTensor(allocator, bias);
        bias.shape = Shape.init(&.{out_channels});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        const v_weight = try allocator.alloc(f32, weight_size);
        errdefer allocator.free(v_weight);
        const v_bias = try allocator.alloc(f32, bias_size);
        errdefer allocator.free(v_bias);

        @memset(v_weight, 0.0);
        @memset(v_bias, 0.0);

        const fan_in = in_channels * kernel_size * kernel_size;
        initializeWeights(random, weight.data, fan_in);
        @memset(bias.data, 0.0);

        return Conv2D{
            .weight = weight,
            .bias = bias,
            .v_weight = v_weight,
            .v_bias = v_bias,
        };
    }

    pub fn deinit(self: Conv2D, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
        allocator.free(self.v_weight);
        allocator.free(self.v_bias);
    }

    pub fn zeroGrad(self: Conv2D) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

    pub fn updateWeights(self: Conv2D, lr: f32, beta: f32) void {
        updateLayerWeights(self.weight.data, self.weight.grad, self.v_weight, lr, beta);
        updateLayerWeights(self.bias.data, self.bias.grad, self.v_bias, lr, beta);
    }

    pub fn forward(self: Conv2D, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        if (graph == null) {
            return try x.conv2d(self.weight, self.bias, allocator, null);
        }
        return try graph.?.conv2d(x, self.weight, self.bias);
    }
};

pub fn deinitModel(model: anytype, allocator: std.mem.Allocator) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            freePersistentTensor(allocator, @field(model, field.name));
        } else if (FieldType == []f32) {
            allocator.free(@field(model, field.name));
        } else if (field_info == .@"struct") {
            deinitModel(&@field(model, field.name), allocator);
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    deinitModel(item, allocator);
                }
            }
        }
    }
}

pub fn zeroGradModel(model: anytype) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            @field(model, field.name).zeroGrad();
        } else if (field_info == .@"struct") {
            zeroGradModel(&@field(model, field.name));
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    zeroGradModel(item);
                }
            }
        }
    }
}

pub fn updateWeightsModel(model: anytype, lr: f32, beta: f32) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            const v_name = "v_" ++ field.name;
            if (@hasField(T, v_name)) {
                updateLayerWeights(
                    @field(model, field.name).data,
                    @field(model, field.name).grad,
                    @field(model, v_name),
                    lr,
                    beta,
                );
            }
        } else if (field_info == .@"struct") {
            updateWeightsModel(&@field(model, field.name), lr, beta);
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    updateWeightsModel(item, lr, beta);
                }
            }
        }
    }
}

fn writeTensorEntry(
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

    try json_buf.print(allocator, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{ name });
    for (0..tensor_ptr.shape.len) |i| {
        if (i > 0) try json_buf.appendSlice(allocator, ",");
        try json_buf.print(allocator, "{}", .{tensor_ptr.shape.dims[i]});
    }
    try json_buf.print(allocator, "],\"data_offsets\":[{},{}]}}", .{ start, end });
}

fn writeModelTensors(
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
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
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

fn writeModelData(
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
        } else if (field_info == .@"struct") {
            try writeModelData(&@field(model, field.name), writer);
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
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

fn loadModelTensors(
    model: anytype,
    reader: anytype,
    meta_obj: anytype,
    current_offset: *usize,
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
            try loadTensorData(reader, meta_obj, name.items, @field(model, field.name), current_offset);
        } else if (field_info == .@"struct") {
            var next_prefix: std.ArrayList(u8) = .empty;
            defer next_prefix.deinit(allocator);
            if (prefix.len > 0) {
                try next_prefix.appendSlice(allocator, prefix);
                try next_prefix.appendSlice(allocator, ".");
            }
            try next_prefix.appendSlice(allocator, field.name);
            try loadModelTensors(&@field(model, field.name), reader, meta_obj, current_offset, allocator, next_prefix.items);
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
                for (&@field(model, field.name), 0..) |*item, idx| {
                    var next_prefix: std.ArrayList(u8) = .empty;
                    defer next_prefix.deinit(allocator);
                    if (prefix.len > 0) {
                        try next_prefix.appendSlice(allocator, prefix);
                        try next_prefix.appendSlice(allocator, ".");
                    }
                    try next_prefix.print(allocator, "{s}.{d}", .{ field.name, idx });
                    try loadModelTensors(item, reader, meta_obj, current_offset, allocator, next_prefix.items);
                }
            }
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

    // 4. 顺序还原每一个 Tensor 字段
    var current_offset: usize = 0;
    try loadModelTensors(model, reader, meta_obj, &current_offset, allocator, "");
}

fn loadTensorData(
    reader: anytype,
    meta_obj: anytype,
    name: []const u8,
    dest: *Tensor,
    current_offset: *usize,
) !void {
    const tensor_meta_val = meta_obj.get(name) orelse {
        std.debug.print("Error: Tensor '{s}' not found in Safetensors header\n", .{name});
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
        std.debug.print("Shape dimension mismatch for '{s}': expected {}, got {}\n", .{ name, dest.shape.len, shape_arr.items.len });
        return error.ShapeMismatch;
    }
    for (0..dest.shape.len) |i| {
        const dim_val = shape_arr.items[i];
        if (dim_val != .integer or @as(usize, @intCast(dim_val.integer)) != dest.shape.dims[i]) {
            std.debug.print("Shape dimension {} mismatch for '{s}': expected {}, got {}\n", .{ i, name, dest.shape.dims[i], dim_val });
            return error.ShapeMismatch;
        }
    }

    // 校验偏移量
    const offsets_val = tensor_meta.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
    if (offsets_val != .array or offsets_val.array.items.len != 2) return error.InvalidSafetensorsHeader;
    const start_offset = @as(usize, @intCast(offsets_val.array.items[0].integer));
    const end_offset = @as(usize, @intCast(offsets_val.array.items[1].integer));

    const expected_len_bytes = dest.data.len * 4;
    if (end_offset - start_offset != expected_len_bytes) {
        return error.SizeMismatch;
    }

    if (start_offset < current_offset.*) {
        std.debug.print("Error: Tensor '{s}' start offset {} is less than current offset {}\n", .{ name, start_offset, current_offset.* });
        return error.InvalidSafetensorsOrder;
    }

    // 跳过对齐填充的空字节（如有必要）
    if (start_offset > current_offset.*) {
        try skipBytes(reader, start_offset - current_offset.*);
        current_offset.* = start_offset;
    }

    // 读取物理二进制数据
    try reader.readSliceAll(std.mem.sliceAsBytes(dest.data));
    current_offset.* += expected_len_bytes;
}

fn skipBytes(reader: anytype, count: usize) !void {
    var dummy: [4096]u8 = undefined;
    var remaining = count;
    while (remaining > 0) {
        const to_read = @min(remaining, dummy.len);
        try reader.readSliceAll(dummy[0..to_read]);
        remaining -= to_read;
    }
}

// ============================================================================
// 3. 通用 Module 包装器（类似于 PyTorch 的 nn.Module 基类继承效果）
// ============================================================================
pub fn Module(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        inner: T,

        const Self = @This();

        // 自动托管模型的初始化，接收已初始化的具体模型实例
        pub fn init(allocator: std.mem.Allocator, inner: T) Self {
            return Self{
                .allocator = allocator,
                .inner = inner,
            };
        }

        // 自动托管 deinit：直接利用反射自动释放内部结构体中的全部参数内存
        pub fn deinit(self: *Self) void {
            deinitModel(&self.inner, self.allocator);
        }

        // 自动托管 zeroGrad
        pub fn zeroGrad(self: *Self) void {
            zeroGradModel(&self.inner);
        }

        // 自动托管 updateWeights
        pub fn updateWeights(self: *Self, lr: f32, beta: f32) void {
            updateWeightsModel(&self.inner, lr, beta);
        }

        // 自动托管 save
        pub fn save(self: *const Self, io: std.Io, file_path: []const u8) !void {
            try saveModel(&self.inner, io, file_path, self.allocator);
        }

        // 自动托管 load
        pub fn load(self: *Self, io: std.Io, file_path: []const u8) !void {
            try loadModel(&self.inner, io, file_path, self.allocator);
        }

        // 自动托管前向传播：将接口直接路由到具体实现的 forward 函数
        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            return try self.inner.forward(allocator, graph, x);
        }
    };
}

pub const Embedding = struct {
    weight: *Tensor,
    v_weight: []f32,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize, random: std.Random) !Embedding {
        const weight = try createPersistentTensor(allocator, vocab_size, embedding_dim, true);
        errdefer freePersistentTensor(allocator, weight);

        const v_weight = try allocator.alloc(f32, vocab_size * embedding_dim);
        errdefer allocator.free(v_weight);
        @memset(v_weight, 0.0);

        initializeWeights(random, weight.data, embedding_dim);

        return Embedding{
            .weight = weight,
            .v_weight = v_weight,
        };
    }

    pub fn deinit(self: Embedding, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        allocator.free(self.v_weight);
    }

    pub fn zeroGrad(self: Embedding) void {
        self.weight.zeroGrad();
    }

    pub fn updateWeights(self: Embedding, lr: f32, beta: f32) void {
        updateLayerWeights(self.weight.data, self.weight.grad, self.v_weight, lr, beta);
    }

    pub fn forward(self: Embedding, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try self.weight.embedding(x, allocator, graph);
    }
};

pub const RMSNorm = struct {
    weight: *Tensor,
    v_weight: []f32,
    eps: f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !RMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);

        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const v_weight = try allocator.alloc(f32, dim);
        errdefer allocator.free(v_weight);
        @memset(v_weight, 0.0);

        return RMSNorm{
            .weight = weight,
            .v_weight = v_weight,
            .eps = eps,
        };
    }

    pub fn deinit(self: RMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        allocator.free(self.v_weight);
    }

    pub fn zeroGrad(self: RMSNorm) void {
        self.weight.zeroGrad();
    }

    pub fn updateWeights(self: RMSNorm, lr: f32, beta: f32) void {
        updateLayerWeights(self.weight.data, self.weight.grad, self.v_weight, lr, beta);
    }

    pub fn forward(self: RMSNorm, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.rmsNorm(self.weight, self.eps, allocator, graph);
    }
};

pub const MLP = struct {
    c_fc: Linear,
    c_proj: Linear,

    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize, random: std.Random) !MLP {
        const c_fc = try Linear.init(allocator, dim, hidden_dim, random);
        errdefer c_fc.deinit(allocator);
        const c_proj = try Linear.init(allocator, hidden_dim, dim, random);
        errdefer c_proj.deinit(allocator);

        return MLP{
            .c_fc = c_fc,
            .c_proj = c_proj,
        };
    }

    pub fn deinit(self: MLP, allocator: std.mem.Allocator) void {
        self.c_fc.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    pub fn zeroGrad(self: MLP) void {
        self.c_fc.zeroGrad();
        self.c_proj.zeroGrad();
    }

    pub fn updateWeights(self: MLP, lr: f32, beta: f32) void {
        self.c_fc.updateWeights(lr, beta);
        self.c_proj.updateWeights(lr, beta);
    }

    pub fn forward(self: MLP, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                x_2d = try g.reshape(x, &.{ B * T, D });
            } else {
                x_2d = try x.reshape(&.{ B * T, D }, allocator, null);
            }
        }
        defer {
            if (is_3d and graph == null) {
                tensor.free(allocator, x_2d);
            }
        }

        const h1 = try self.c_fc.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, h1);

        const a1 = if (graph) |g| try g.gelu(h1) else try h1.gelu(allocator, null);
        defer if (graph == null) tensor.free(allocator, a1);

        const h2 = try self.c_proj.forward(allocator, graph, a1);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                return try g.reshape(h2, &.{ B, T, D });
            } else {
                defer tensor.free(allocator, h2);
                return try h2.reshape(&.{ B, T, D }, allocator, null);
            }
        }
        return h2;
    }
};

pub const CausalSelfAttention = struct {
    q_attn: Linear,
    k_attn: Linear,
    v_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,

    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !CausalSelfAttention {
        const q_attn = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer q_attn.deinit(allocator);
        const k_attn = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer k_attn.deinit(allocator);
        const v_attn = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer v_attn.deinit(allocator);
        const c_proj = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer c_proj.deinit(allocator);

        return CausalSelfAttention{
            .q_attn = q_attn,
            .k_attn = k_attn,
            .v_attn = v_attn,
            .c_proj = c_proj,
            .n_head = n_head,
            .n_embd = n_embd,
        };
    }

    pub fn deinit(self: CausalSelfAttention, allocator: std.mem.Allocator) void {
        self.q_attn.deinit(allocator);
        self.k_attn.deinit(allocator);
        self.v_attn.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    pub fn zeroGrad(self: CausalSelfAttention) void {
        self.q_attn.zeroGrad();
        self.k_attn.zeroGrad();
        self.v_attn.zeroGrad();
        self.c_proj.zeroGrad();
    }

    pub fn updateWeights(self: CausalSelfAttention, lr: f32, beta: f32) void {
        self.q_attn.updateWeights(lr, beta);
        self.k_attn.updateWeights(lr, beta);
        self.v_attn.updateWeights(lr, beta);
        self.c_proj.updateWeights(lr, beta);
    }

    pub fn forward(self: CausalSelfAttention, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = x.shape.dims[2];
        const nh = self.n_head;
        const hs = C / nh;

        var x_2d = x;
        if (graph) |g| {
            x_2d = try g.reshape(x, &.{ B * T, C });
        } else {
            x_2d = try x.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, x_2d);

        const q_2d = try self.q_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, q_2d);
        const k_2d = try self.k_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, k_2d);
        const v_2d = try self.v_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, v_2d);

        var q_4d = q_2d;
        var k_4d = k_2d;
        var v_4d = v_2d;
        if (graph) |g| {
            q_4d = try g.reshape(q_2d, &.{ B, T, nh, hs });
            k_4d = try g.reshape(k_2d, &.{ B, T, nh, hs });
            v_4d = try g.reshape(v_2d, &.{ B, T, nh, hs });
        } else {
            q_4d = try q_2d.reshape(&.{ B, T, nh, hs }, allocator, null);
            k_4d = try k_2d.reshape(&.{ B, T, nh, hs }, allocator, null);
            v_4d = try v_2d.reshape(&.{ B, T, nh, hs }, allocator, null);
        }
        defer if (graph == null) {
            tensor.free(allocator, q_4d);
            tensor.free(allocator, k_4d);
            tensor.free(allocator, v_4d);
        };

        var q = q_4d;
        var k = k_4d;
        var v = v_4d;
        if (graph) |g| {
            q = try g.transposeND(q_4d, 1, 2);
            k = try g.transposeND(k_4d, 1, 2);
            v = try g.transposeND(v_4d, 1, 2);
        } else {
            q = try q_4d.transpose(1, 2, allocator, null);
            k = try k_4d.transpose(1, 2, allocator, null);
            v = try v_4d.transpose(1, 2, allocator, null);
        }
        defer if (graph == null) {
            tensor.free(allocator, q);
            tensor.free(allocator, k);
            tensor.free(allocator, v);
        };

        var k_t = k;
        if (graph) |g| {
            k_t = try g.transposeND(k, 2, 3);
        } else {
            k_t = try k.transpose(2, 3, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, k_t);

        var att = q;
        if (graph) |g| {
            att = try g.batchMatMul(q, k_t);
        } else {
            att = try q.batchMatMul(k_t, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        var att_scaled = att;
        if (graph) |g| {
            att_scaled = try g.mulScalar(att, scale);
        } else {
            att_scaled = try att.mulScalar(scale, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_scaled);

        const mask_data = try allocator.alloc(f32, B * nh * T * T);
        defer allocator.free(mask_data);
        @memset(mask_data, 0.0);
        for (0..B) |b| {
            for (0..nh) |h| {
                for (0..T) |i| {
                    for (0..T) |j| {
                        if (j > i) {
                            mask_data[((b * nh + h) * T + i) * T + j] = -1e9;
                        }
                    }
                }
            }
        }
        const mask = try tensor.array(allocator, &.{ B, nh, T, T }, mask_data);
        defer tensor.free(allocator, mask);

        var mask_node = mask;
        if (graph) |g| {
            mask_node = try g.tensorNDWithData(&.{ B, nh, T, T }, mask_data, false);
        }

        var att_masked = att_scaled;
        if (graph) |g| {
            att_masked = try g.add(att_scaled, mask_node);
        } else {
            att_masked = try att_scaled.add(mask, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_masked);

        var att_sm = att_masked;
        if (graph) |g| {
            att_sm = try g.softmax(att_masked);
        } else {
            att_sm = try att_masked.softmax(allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_sm);

        var y_4d = att_sm;
        if (graph) |g| {
            y_4d = try g.batchMatMul(att_sm, v);
        } else {
            y_4d = try att_sm.batchMatMul(v, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_4d);

        var y_trans = y_4d;
        if (graph) |g| {
            y_trans = try g.transposeND(y_4d, 1, 2);
        } else {
            y_trans = try y_4d.transpose(1, 2, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_trans);

        var y_3d = y_trans;
        if (graph) |g| {
            y_3d = try g.reshape(y_trans, &.{ B, T, C });
        } else {
            y_3d = try y_trans.reshape(&.{ B, T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_3d);

        var y_2d = y_3d;
        if (graph) |g| {
            y_2d = try g.reshape(y_3d, &.{ B * T, C });
        } else {
            y_2d = try y_3d.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_2d);

        const out_2d = try self.c_proj.forward(allocator, graph, y_2d);
        defer if (graph == null) tensor.free(allocator, out_2d);

        if (graph) |g| {
            return try g.reshape(out_2d, &.{ B, T, C });
        } else {
            return try out_2d.reshape(&.{ B, T, C }, allocator, null);
        }
    }
};

pub const TransformerBlock = struct {
    ln_1: RMSNorm,
    attn: CausalSelfAttention,
    ln_2: RMSNorm,
    mlp: MLP,

    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !TransformerBlock {
        const ln_1 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer ln_1.deinit(allocator);
        const attn = try CausalSelfAttention.init(allocator, n_embd, n_head, random);
        errdefer attn.deinit(allocator);
        const ln_2 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer ln_2.deinit(allocator);
        const mlp = try MLP.init(allocator, n_embd, 4 * n_embd, random);
        errdefer mlp.deinit(allocator);

        return TransformerBlock{
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
        };
    }

    pub fn deinit(self: TransformerBlock, allocator: std.mem.Allocator) void {
        self.ln_1.deinit(allocator);
        self.attn.deinit(allocator);
        self.ln_2.deinit(allocator);
        self.mlp.deinit(allocator);
    }

    pub fn zeroGrad(self: TransformerBlock) void {
        self.ln_1.zeroGrad();
        self.attn.zeroGrad();
        self.ln_2.zeroGrad();
        self.mlp.zeroGrad();
    }

    pub fn updateWeights(self: TransformerBlock, lr: f32, beta: f32) void {
        self.ln_1.updateWeights(lr, beta);
        self.attn.updateWeights(lr, beta);
        self.ln_2.updateWeights(lr, beta);
        self.mlp.updateWeights(lr, beta);
    }

    pub fn forward(self: TransformerBlock, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const x_norm1 = try self.ln_1.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, x_norm1);

        const x_attn = try self.attn.forward(allocator, graph, x_norm1);
        defer if (graph == null) tensor.free(allocator, x_attn);

        const x1 = if (graph) |g| try g.add(x, x_attn) else try x.add(x_attn, allocator, null);
        defer if (graph == null) tensor.free(allocator, x1);

        const x_norm2 = try self.ln_2.forward(allocator, graph, x1);
        defer if (graph == null) tensor.free(allocator, x_norm2);

        const x_mlp = try self.mlp.forward(allocator, graph, x_norm2);
        defer if (graph == null) tensor.free(allocator, x_mlp);

        if (graph) |g| {
            return try g.add(x1, x_mlp);
        } else {
            return try x1.add(x_mlp, allocator, null);
        }
    }
};

pub const GPTConfig = struct {
    vocab_size: usize,
    block_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
};

pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        wte: Embedding,
        wpe: Embedding,
        h: [config.n_layer]TransformerBlock,
        ln_f: RMSNorm,
        lm_head: Linear,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, random: std.Random) !Self {
            const wte = try Embedding.init(allocator, config.vocab_size, config.n_embd, random);
            errdefer wte.deinit(allocator);
            const wpe = try Embedding.init(allocator, config.block_size, config.n_embd, random);
            errdefer wpe.deinit(allocator);

            var h: [config.n_layer]TransformerBlock = undefined;
            var i: usize = 0;
            errdefer {
                for (0..i) |j| {
                    h[j].deinit(allocator);
                }
            }
            while (i < config.n_layer) : (i += 1) {
                h[i] = try TransformerBlock.init(allocator, config.n_embd, config.n_head, random);
            }

            const ln_f = try RMSNorm.init(allocator, config.n_embd, 1e-5);
            errdefer ln_f.deinit(allocator);

            const lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size, random);
            errdefer lm_head.deinit(allocator);

            return Self{
                .wte = wte,
                .wpe = wpe,
                .h = h,
                .ln_f = ln_f,
                .lm_head = lm_head,
            };
        }

        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            const tok_emb = try self.wte.forward(allocator, graph, x);
            defer if (graph == null) tensor.free(allocator, tok_emb);

            const pos_data = try allocator.alloc(f32, B * T);
            defer allocator.free(pos_data);
            for (0..B) |b| {
                for (0..T) |t| {
                    pos_data[b * T + t] = @as(f32, @floatFromInt(t));
                }
            }
            const pos_tensor = try tensor.array(allocator, &.{ B, T }, pos_data);
            defer tensor.free(allocator, pos_tensor);

            var pos_node = pos_tensor;
            if (graph) |g| {
                pos_node = try g.tensorNDWithData(&.{ B, T }, pos_data, false);
            }

            const pos_emb = try self.wpe.forward(allocator, graph, pos_node);
            defer if (graph == null) tensor.free(allocator, pos_emb);

            var h_x = tok_emb;
            if (graph) |g| {
                h_x = try g.add(tok_emb, pos_emb);
            } else {
                h_x = try tok_emb.add(pos_emb, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, h_x);

            var current_h = h_x;
            if (graph == null) {
                current_h = try h_x.clone(allocator);
            }
            inline for (0..config.n_layer) |i| {
                const next_h = try self.h[i].forward(allocator, graph, current_h);
                if (graph == null) {
                    tensor.free(allocator, current_h);
                    current_h = next_h;
                } else {
                    current_h = next_h;
                }
            }

            const ln_x = try self.ln_f.forward(allocator, graph, current_h);
            defer if (graph == null) tensor.free(allocator, ln_x);
            if (graph == null) tensor.free(allocator, current_h);

            var ln_x_2d = ln_x;
            if (graph) |g| {
                ln_x_2d = try g.reshape(ln_x, &.{ B * T, config.n_embd });
            } else {
                ln_x_2d = try ln_x.reshape(&.{ B * T, config.n_embd }, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, ln_x_2d);

            const logits_2d = try self.lm_head.forward(allocator, graph, ln_x_2d);
            defer if (graph == null) tensor.free(allocator, logits_2d);

            if (graph) |g| {
                return try g.reshape(logits_2d, &.{ B, T, config.vocab_size });
            } else {
                return try logits_2d.reshape(&.{ B, T, config.vocab_size }, allocator, null);
            }
        }
    };
}








// ============================================================================
// 5. 底层数学与内存辅助函数
// ============================================================================

fn updateLayerWeights(w: []f32, dw: []const f32, v: []f32, lr: f32, beta: f32) void {
    for (w, dw, v) |*weight, grad, *velocity| {
        velocity.* = beta * velocity.* + lr * grad;
        weight.* -= velocity.*;
    }
}

fn initializeWeights(random: std.Random, w: []f32, fan_in: usize) void {
    const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));
    for (w) |*val| {
        val.* = normalRandom(random) * std_dev;
    }
}

fn normalRandom(random: std.Random) f32 {
    var u_1: f32 = random.float(f32);
    while (u_1 == 0.0) {
        u_1 = random.float(f32);
    }
    const u_2 = random.float(f32);
    return @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
}

fn createPersistentTensor(allocator: std.mem.Allocator, rows: usize, cols: usize, requires_grad: bool) !*Tensor {
    const t = try allocator.create(Tensor);
    const shape = Shape.init(&.{rows, cols});
    const strides = tensor.computeContiguousStrides(shape);
    t.* = Tensor{
        .data = try allocator.alloc(f32, rows * cols),
        .grad = if (requires_grad) try allocator.alloc(f32, rows * cols) else &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = requires_grad,
        .creator = null,
    };
    @memset(t.data, 0.0);
    if (requires_grad) {
        @memset(t.grad, 0.0);
    }
    return t;
}

fn freePersistentTensor(allocator: std.mem.Allocator, t: *Tensor) void {
    allocator.free(t.data);
    if (t.requires_grad) {
        allocator.free(t.grad);
    }
    allocator.destroy(t);
}

test "Embedding Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4, random);
    defer emb.deinit(arena);

    var x = try createPersistentTensor(arena, 2, 3, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y_eager = try emb.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);
}

test "Embedding Module Graph Mode" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4, random);
    defer emb.deinit(arena);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x = try graph.tensorND(&.{2, 3}, false);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y = try emb.forward(arena, &graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    for (0..6) |i| {
        for (0..4) |j| {
            try std.testing.expectEqual(@as(f32, 1.0), emb.weight.grad[i * 4 + j]);
        }
    }
    for (6..10) |i| {
        for (0..4) |j| {
            try std.testing.expectEqual(@as(f32, 0.0), emb.weight.grad[i * 4 + j]);
        }
    }
}

test "RMSNorm Module" {
    const arena = std.testing.allocator;
    var norm = try RMSNorm.init(arena, 4, 1e-5);
    defer norm.deinit(arena);

    var x = try createPersistentTensor(arena, 2, 4, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0; x.data[3] = 4.0;
    x.data[4] = 5.0; x.data[5] = 6.0; x.data[6] = 7.0; x.data[7] = 8.0;

    const y_eager = try norm.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 4}, true);
    @memcpy(x_node.data, x.data);

    const y = try norm.forward(arena, &graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var norm_g_grad_sum: f32 = 0.0;
    for (norm.weight.grad) |g| norm_g_grad_sum += @abs(g);
    try std.testing.expect(norm_g_grad_sum > 0.0);

    var x_grad_sum: f32 = 0.0;
    for (x_node.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 0.0);
}

test "MLP Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var mlp = try MLP.init(arena, 4, 8, random);
    defer mlp.deinit(arena);

    const x_3d = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3, 4});
    x_3d.* = Tensor{
        .data = try arena.alloc(f32, 24),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x_3d.data);
        arena.destroy(x_3d);
    }
    for (x_3d.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const y_eager = try mlp.forward(arena, null, x_3d);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 4}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try mlp.forward(arena, &graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var w1_grad_sum: f32 = 0.0;
    for (mlp.c_fc.weight.grad) |g| w1_grad_sum += @abs(g);
    try std.testing.expect(w1_grad_sum > 0.0);
}

test "CausalSelfAttention Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var attn = try CausalSelfAttention.init(arena, 8, 2, random);
    defer attn.deinit(arena);

    const x_3d = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3, 8});
    x_3d.* = Tensor{
        .data = try arena.alloc(f32, 48),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x_3d.data);
        arena.destroy(x_3d);
    }
    for (x_3d.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const y_eager = try attn.forward(arena, null, x_3d);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 8}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try attn.forward(arena, &graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var q_grad_sum: f32 = 0.0;
    for (attn.q_attn.weight.grad) |g| q_grad_sum += @abs(g);
    try std.testing.expect(q_grad_sum > 0.0);
}

test "GPT Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const config = GPTConfig{
        .vocab_size = 10,
        .block_size = 5,
        .n_embd = 8,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try GPT(config).init(arena, random);
    defer deinitModel(&gpt, arena);

    const x = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3});
    x.* = Tensor{
        .data = try arena.alloc(f32, 6),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x.data);
        arena.destroy(x);
    }
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y_eager = try gpt.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3}, false);
    @memcpy(x_node.data, x.data);

    const y = try gpt.forward(arena, &graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var wte_grad_sum: f32 = 0.0;
    for (gpt.wte.weight.grad) |g| wte_grad_sum += @abs(g);
    try std.testing.expect(wte_grad_sum > 0.0);
}

test "GPT Module Save and Load" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const config = GPTConfig{
        .vocab_size = 10,
        .block_size = 5,
        .n_embd = 8,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try GPT(config).init(arena, random);
    defer deinitModel(&gpt, arena);

    try saveModel(&gpt, std.testing.io, "test_gpt_model.safetensors", arena);
    defer {
        std.Io.Dir.cwd().deleteFile(std.testing.io, "test_gpt_model.safetensors") catch {};
    }

    var gpt2 = try GPT(config).init(arena, random);
    defer deinitModel(&gpt2, arena);

    try loadModel(&gpt2, std.testing.io, "test_gpt_model.safetensors", arena);

    for (gpt.wte.weight.data, gpt2.wte.weight.data) |w1, w2| {
        try std.testing.expectEqual(w1, w2);
    }
}


