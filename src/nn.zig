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

    // 初始化一个线性层，自动生成对应的持久化 Tensor，并进行 He 参数初始化
    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, random: std.Random) !Linear {
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        // 使用 He (Kaiming) 归一化方法初始化权重，偏置设为 0
        initializeWeights(random, weight.data, in_features);
        @memset(bias.data, 0.0);

        return Linear{
            .weight = weight,
            .bias = bias,
        };
    }

    // 释放该层持有的所有持久化数据内存
    pub fn deinit(self: Linear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    // 将本层的梯度设为 0
    pub fn zeroGrad(self: Linear) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
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

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, random: std.Random) !Conv2D {
        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        // Correct the shape of Conv2D weight to [out_channels, in_channels, kernel_size, kernel_size]
        weight.shape = Shape.init(&.{out_channels, in_channels, kernel_size, kernel_size});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, out_channels, true);
        errdefer freePersistentTensor(allocator, bias);
        bias.shape = Shape.init(&.{out_channels});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        const fan_in = in_channels * kernel_size * kernel_size;
        initializeWeights(random, weight.data, fan_in);
        @memset(bias.data, 0.0);

        return Conv2D{
            .weight = weight,
            .bias = bias,
        };
    }

    pub fn deinit(self: Conv2D, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    pub fn zeroGrad(self: Conv2D) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
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

pub fn collectParameters(model: anytype, allocator: std.mem.Allocator) ![]*Tensor {
    var list: std.ArrayList(*Tensor) = .empty;
    errdefer list.deinit(allocator);
    try collectParametersInternal(model, &list, allocator);
    return list.toOwnedSlice(allocator);
}

fn collectParametersInternal(model: anytype, list: *std.ArrayList(*Tensor), allocator: std.mem.Allocator) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            const tensor_ptr = @field(model, field.name);
            if (tensor_ptr.requires_grad) {
                try list.append(allocator, tensor_ptr);
            }
        } else if (field_info == .@"struct") {
            try collectParametersInternal(&@field(model, field.name), list, allocator);
        } else if (field_info == .@"array") {
            const elem_info = @typeInfo(field_info.@"array".child);
            if (elem_info == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    try collectParametersInternal(item, list, allocator);
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

/// 嵌入层 (Embedding Layer)
/// 用于将离散的 Token ID（例如整数索引）映射为连续的低维稠密向量。
/// 在数学上，这等价于使用 One-hot 编码与权重矩阵相乘，而在实现上通过高效的查找表 (Lookup Table) 实现。
/// 
/// 权重形状：[vocab_size, embedding_dim]
pub const Embedding = struct {
    weight: *Tensor,        // 嵌入层权重矩阵表 (Shape: [vocab_size, embedding_dim])

    /// 初始化嵌入层
    /// vocab_size: 词表大小（可索引的最大整数范围）
    /// embedding_dim: 映射出的隐藏嵌入维度大小
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize, random: std.Random) !Embedding {
        const weight = try createPersistentTensor(allocator, vocab_size, embedding_dim, true);
        errdefer freePersistentTensor(allocator, weight);

        // 使用正态分布 He/Kaiming 随机数初始化权重表
        initializeWeights(random, weight.data, embedding_dim);

        return Embedding{
            .weight = weight,
        };
    }

    /// 释放层内所有关联的 Tensor 内存资源
    pub fn deinit(self: Embedding, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    /// 清空权重对应的梯度
    pub fn zeroGrad(self: Embedding) void {
        self.weight.zeroGrad();
    }

    /// 查找映射前向传播
    /// 输入 x 为包含 Token ID 的任意维度 Tensor，输出形状为 x.shape + [embedding_dim]
    pub fn forward(self: Embedding, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try self.weight.embedding(x, allocator, graph);
    }
};

/// Root Mean Square Layer Normalization (RMSNorm) 层
/// 相比于传统的 LayerNorm，RMSNorm 移除了平移均值（mean centering）的步骤，
/// 仅根据隐藏特征的均方根进行缩放，不仅能够保证与 LayerNorm 相当甚至更好的效果，
/// 还能减少约 7%-50% 的计算开销。
/// 
/// 数学公式：
/// \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
/// \text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \gamma_i
/// 其中 \gamma 为可学习的缩放因子（对应结构体中的 weight 属性）。
pub const RMSNorm = struct {
    weight: *Tensor,        // 可学习的缩放因子 gamma (Shape: [dim])
    eps: f32,               // 均方根分母防止除以 0 的极小常数 (epsilon)

    /// 初始化 RMSNorm 层
    /// dim: 隐藏特征的维度大小
    /// eps: 防止分母为 0 的微小偏差值（通常为 1e-5）
    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !RMSNorm {
        // 创建持久化的 Tensor 并在内存中申请空间，包含梯度
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        // 初始化缩放因子为 1.0
        @memset(weight.data, 1.0);

        // 设置形状和步长为 1D [dim]
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        return RMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    /// 释放 RMSNorm 占用的所有系统和显存资源
    pub fn deinit(self: RMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    /// 将权重的梯度清零，用于下一次反向传播
    pub fn zeroGrad(self: RMSNorm) void {
        self.weight.zeroGrad();
    }

    /// 执行 RMSNorm 的前向传播过程
    /// 支持 Eager (无图) 模式与 Graph (计算图自动微分) 模式
    pub fn forward(self: RMSNorm, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.rmsNorm(self.weight, self.eps, allocator, graph);
    }
};

/// 多层感知机 (MLP) / 前馈网络 (Feed-forward Network) 模块
/// Transformer 架构中的重要组件，紧跟在 Self-Attention 之后，
/// 用于在每个 Token 位置上独立地进行非线性特征投影与融合。
/// 
/// 数学公式：
/// \text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
/// 结构：
/// Linear(dim -> hidden_dim) -> GELU 激活函数 -> Linear(hidden_dim -> dim)
/// 其中 hidden_dim 通常设置为 4 * dim。
pub const MLP = struct {
    c_fc: Linear,           // 升维投影层 (dim -> hidden_dim)
    c_proj: Linear,         // 降维投影层 (hidden_dim -> dim)

    /// 初始化 MLP 模块
    /// dim: 输入与输出隐藏维度
    /// hidden_dim: 中间隐藏维度 (一般为 4 * dim)
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

    /// 释放子层的所有内存资源
    pub fn deinit(self: MLP, allocator: std.mem.Allocator) void {
        self.c_fc.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    /// 子层梯度全部清零
    pub fn zeroGrad(self: MLP) void {
        self.c_fc.zeroGrad();
        self.c_proj.zeroGrad();
    }



    /// 前向传播逻辑
    /// 支持输入 2D Tensor [B*T, D] 或 3D Tensor [B, T, D]
    pub fn forward(self: MLP, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        
        // 1. 如果输入是 3D [B, T, D]，则将其打平为 2D [B*T, D] 以满足 Linear 矩阵乘法的输入规范
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
            // Eager 模式下需要释放临时 reshape 生成的 Tensor 内存
            if (is_3d and graph == null) {
                tensor.free(allocator, x_2d);
            }
        }

        // 2. 升维映射: [B*T, D] -> [B*T, hidden_dim]
        const h1 = try self.c_fc.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, h1);

        // 3. GELU 激活函数引入非线性
        const a1 = if (graph) |g| try g.gelu(h1) else try h1.gelu(allocator, null);
        defer if (graph == null) tensor.free(allocator, a1);

        // 4. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const h2 = try self.c_proj.forward(allocator, graph, a1);

        // 5. 如果输入原本是 3D，需要将输出再重新恢复成 3D 形状: [B, T, D]
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

/// 因果自注意力机制 (Causal Self-Attention / Masked Multi-Head Attention)
/// Transformer 的核心机制，负责建模序列中不同位置的依赖关系。
/// 
/// 数学公式：
/// Q = X W_q, \quad K = X W_k, \quad V = X W_v
/// \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
/// \text{Output} = \text{Attention}(Q, K, V) W_p
/// 其中 M 是因果掩码矩阵，上三角（未来位置）元素为 -\infty，其余为 0。
/// 
/// 包含以下关键设计：
/// 1. 多头注意力 (Multi-Head)：将特征通道划分为 nh 个头，让模型在多个不同的投影子空间内并行关注信息。
/// 2. 因果掩码 (Causal Mask)：通过加上上三角矩阵（值为 -inf），阻止当前位置关注未来的位置，确保自回归生成时的因果律。
pub const CausalSelfAttention = struct {
    q_attn: Linear,         // Query 线性投影层
    k_attn: Linear,         // Key 线性投影层
    v_attn: Linear,         // Value 线性投影层
    c_proj: Linear,         // 最终的多头输出融合与投影层 (c_proj)
    n_head: usize,          // 注意力头数 (n_head)
    n_embd: usize,          // 嵌入维度 (n_embd)

    /// 初始化因果自注意力层
    /// n_embd: 隐藏嵌入维度，必须能被 n_head 整除
    /// n_head: 注意力头数
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

    /// 释放所有线性投射子层的内存资源
    pub fn deinit(self: CausalSelfAttention, allocator: std.mem.Allocator) void {
        self.q_attn.deinit(allocator);
        self.k_attn.deinit(allocator);
        self.v_attn.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    /// 所有线性投射子层的梯度清零
    pub fn zeroGrad(self: CausalSelfAttention) void {
        self.q_attn.zeroGrad();
        self.k_attn.zeroGrad();
        self.v_attn.zeroGrad();
        self.c_proj.zeroGrad();
    }



    /// 前向注意力计算流程
    /// 输入 x 的形状必须为 3D: [B, T, C]
    /// 其中 B 为批次大小 (Batch Size)，T 为时间步长度 (Sequence Length)，C 为通道特征维数 (n_embd)
    pub fn forward(self: CausalSelfAttention, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = x.shape.dims[2];
        const nh = self.n_head;
        const hs = C / nh; // 每个注意力头的维度大小 (head size)

        // 1. 将 3D 输入 [B, T, C] 展平为 2D [B*T, C] 便于做常规的线性矩阵映射
        var x_2d = x;
        if (graph) |g| {
            x_2d = try g.reshape(x, &.{ B * T, C });
        } else {
            x_2d = try x.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, x_2d);

        // 2. 投影计算 Query, Key, Value
        // 输出形状均为 [B*T, C]
        const q_2d = try self.q_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, q_2d);
        const k_2d = try self.k_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, k_2d);
        const v_2d = try self.v_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, v_2d);

        // 3. 将投影后的数据重新塑形为 4D 多头结构: [B*T, C] -> [B, T, nh, hs]
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

        // 4. 转置特征轴，使得“注意力头数 nh”维度排在前部以进行 Batch 矩阵乘法
        // 转置变化: [B, T, nh, hs] -> [B, nh, T, hs]
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

        // 5. 转置 Key 用于计算点积注意力: [B, nh, T, hs] -> [B, nh, hs, T]
        var k_t = k;
        if (graph) |g| {
            k_t = try g.transposeND(k, 2, 3);
        } else {
            k_t = try k.transpose(2, 3, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, k_t);

        // 6. 计算注意力原始得分: Q * K^T
        // 输出矩阵形状: [B, nh, T, hs] * [B, nh, hs, T] -> [B, nh, T, T]
        var att = q;
        if (graph) |g| {
            att = try g.batchMatMul(q, k_t);
        } else {
            att = try q.batchMatMul(k_t, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att);

        // 7. 缩放得分，除以 sqrt(head_size) 避免梯度消失/爆炸: score = (Q * K^T) / sqrt(hs)
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        var att_scaled = att;
        if (graph) |g| {
            att_scaled = try g.mulScalar(att, scale);
        } else {
            att_scaled = try att.mulScalar(scale, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_scaled);

        // 8. 构造因果掩码 (Causal Mask) 矩阵
        // 该矩阵只包含 0 和 -1e9。上三角（未来位置 j > 当前位置 i）部分全部填充 -1e9。
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

        // 9. 将掩码加上注意力得分: score + mask
        // 未来时刻对应的得分将变为极小值 (-1e9)，进而在 Softmax 后权重归零。
        var att_masked = att_scaled;
        if (graph) |g| {
            att_masked = try g.add(att_scaled, mask_node);
        } else {
            att_masked = try att_scaled.add(mask, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_masked);

        // 10. Softmax 归一化，得到归一化的注意力概率分布图: [B, nh, T, T]
        var att_sm = att_masked;
        if (graph) |g| {
            att_sm = try g.softmax(att_masked);
        } else {
            att_sm = try att_masked.softmax(allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_sm);

        // 11. 用注意力权重与 Value 相乘: weight * V
        // 形状变化: [B, nh, T, T] * [B, nh, T, hs] -> [B, nh, T, hs]
        var y_4d = att_sm;
        if (graph) |g| {
            y_4d = try g.batchMatMul(att_sm, v);
        } else {
            y_4d = try att_sm.batchMatMul(v, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_4d);

        // 12. 将多头的输出转置回去，重新展平拼接成单头向量表示
        // 转置: [B, nh, T, hs] -> [B, T, nh, hs]
        var y_trans = y_4d;
        if (graph) |g| {
            y_trans = try g.transposeND(y_4d, 1, 2);
        } else {
            y_trans = try y_4d.transpose(1, 2, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_trans);

        // 整合形状为 3D: [B, T, nh * hs] = [B, T, C]
        var y_3d = y_trans;
        if (graph) |g| {
            y_3d = try g.reshape(y_trans, &.{ B, T, C });
        } else {
            y_3d = try y_trans.reshape(&.{ B, T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_3d);

        // 13. 将输出展平为 2D，以便穿过最后的输出投影线性层 (c_proj)
        // 重塑: [B, T, C] -> [B*T, C]
        var y_2d = y_3d;
        if (graph) |g| {
            y_2d = try g.reshape(y_3d, &.{ B * T, C });
        } else {
            y_2d = try y_3d.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_2d);

        // 投影输出映射: [B*T, C] -> [B*T, C]
        const out_2d = try self.c_proj.forward(allocator, graph, y_2d);
        defer if (graph == null) tensor.free(allocator, out_2d);

        // 14. 恢复并输出最终的 3D 表示: [B, T, C]
        if (graph) |g| {
            return try g.reshape(out_2d, &.{ B, T, C });
        } else {
            return try out_2d.reshape(&.{ B, T, C }, allocator, null);
        }
    }
};

/// Transformer 编码器/解码器 Block 模块 (Transformer Block)
/// 采用 Pre-LN (Layer Normalization Pre-activation) 架构进行组装：
/// 1. x_norm1 = RMSNorm(x)
/// 2. x_attn = SelfAttention(x_norm1)
/// 3. x1 = x + x_attn  (第一层残差连接)
/// 4. x_norm2 = RMSNorm(x1)
/// 5. x_mlp = MLP(x_norm2)
/// 6. out = x1 + x_mlp (第二层残差连接)
/// 
/// 相比于 Post-LN，Pre-LN 可以在初始化阶段使梯度更直接地传导到低层网络，从而支持训练极深的网络。
pub const TransformerBlock = struct {
    ln_1: RMSNorm,          // 第一层归一化层，在 Attention 计算前执行
    attn: CausalSelfAttention, // 因果自注意力机制层
    ln_2: RMSNorm,          // 第二层归一化层，在 MLP 计算前执行
    mlp: MLP,               // 前馈多层感知机层

    /// 初始化 Transformer 块
    /// n_embd: 隐藏特征特征维度
    /// n_head: 注意力头数
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

    /// 释放所有内部子层的资源
    pub fn deinit(self: TransformerBlock, allocator: std.mem.Allocator) void {
        self.ln_1.deinit(allocator);
        self.attn.deinit(allocator);
        self.ln_2.deinit(allocator);
        self.mlp.deinit(allocator);
    }

    /// 块内所有子层的梯度清零
    pub fn zeroGrad(self: TransformerBlock) void {
        self.ln_1.zeroGrad();
        self.attn.zeroGrad();
        self.ln_2.zeroGrad();
        self.mlp.zeroGrad();
    }



    /// 前向传播流程：x -> Block(x) -> out
    pub fn forward(self: TransformerBlock, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        // 1. 第一条支路: RMSNorm -> Attention
        const x_norm1 = try self.ln_1.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, x_norm1);

        const x_attn = try self.attn.forward(allocator, graph, x_norm1);
        defer if (graph == null) tensor.free(allocator, x_attn);

        // 2. 第一条残差混合: x1 = x + Attention(RMSNorm(x))
        const x1 = if (graph) |g| try g.add(x, x_attn) else try x.add(x_attn, allocator, null);
        defer if (graph == null) tensor.free(allocator, x1);

        // 3. 第二条支路: RMSNorm -> MLP
        const x_norm2 = try self.ln_2.forward(allocator, graph, x1);
        defer if (graph == null) tensor.free(allocator, x_norm2);

        const x_mlp = try self.mlp.forward(allocator, graph, x_norm2);
        defer if (graph == null) tensor.free(allocator, x_mlp);

        // 4. 第二条残差混合: out = x1 + MLP(RMSNorm(x1))
        if (graph) |g| {
            return try g.add(x1, x_mlp);
        } else {
            return try x1.add(x_mlp, allocator, null);
        }
    }
};

/// 堆叠多层 Transformer 块的解码器主干网络 (Transformer Decoder)
/// 类似于 PyTorch 的 nn.TransformerDecoder。
/// 它接收输入特征，依次串联通过 n_layer 个 TransformerBlock，
/// 并在最末端使用一个 RMSNorm 进行最终的标准化，用作整个 Decoder 骨架的输出。
pub fn TransformerDecoder(comptime n_layer: usize) type {
    return struct {
        h: [n_layer]TransformerBlock, // 堆叠的 Blocks 数组
        ln_f: RMSNorm,                // 骨架最末端用于规范化的归一化层

        const Self = @This();

        /// 初始化整个解码器组件
        pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !Self {
            var h: [n_layer]TransformerBlock = undefined;
            var i: usize = 0;
            errdefer {
                for (0..i) |j| {
                    h[j].deinit(allocator);
                }
            }
            // 循环初始化每一层 TransformerBlock
            while (i < n_layer) : (i += 1) {
                h[i] = try TransformerBlock.init(allocator, n_embd, n_head, random);
            }

            // 初始化最后的层归一化层
            const ln_f = try RMSNorm.init(allocator, n_embd, 1e-5);
            errdefer {
                for (0..n_layer) |j| {
                    h[j].deinit(allocator);
                }
                ln_f.deinit(allocator);
            }

            return Self{
                .h = h,
                .ln_f = ln_f,
            };
        }

        /// 释放整个骨架层及各 Block 的内存
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            for (self.h) |layer| {
                layer.deinit(allocator);
            }
            self.ln_f.deinit(allocator);
        }

        /// 将所有 Block 和最末端 Norm 层的梯度全部清零
        pub fn zeroGrad(self: Self) void {
            for (self.h) |layer| {
                layer.zeroGrad();
            }
            self.ln_f.zeroGrad();
        }



        /// 解码器主干网络的前向传播流程
        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            var current_x = x;
            // 依次贯穿每一层 Block
            for (self.h) |layer| {
                const next_x = try layer.forward(allocator, graph, current_x);
                // 释放 Eager 模式下的中间隐特征 Tensor 内存，避免泄漏
                if (graph == null and current_x != x) {
                    tensor.free(allocator, current_x);
                }
                current_x = next_x;
            }

            // 执行最后一层 RMSNorm 映射输出
            const out = try self.ln_f.forward(allocator, graph, current_x);
            if (graph == null and current_x != x) {
                tensor.free(allocator, current_x);
            }
            return out;
        }
    };
}

/// GPT 模型配置结构体
pub const GPTConfig = struct {
    vocab_size: usize,      // 词表大小 (Vocab Size)，决定输入和输出层的映射维度
    block_size: usize,      // 最大上下文长度/时间步长度 (Context Length / Block Size)
    n_embd: usize,          // 隐藏特征嵌入维度 (Embedding Dimension)
    n_head: usize,          // 多头注意力头数 (Attention Heads)
    n_layer: usize,         // Transformer 块堆叠的层数 (Number of Decoder Layers)
};

/// 泛型 GPT 模型定义函数
/// 接收一个 comptime 的 `GPTConfig` 配置，返回一个对应的 GPT 模型结构体类型。
/// 
/// 结构设计：
/// 1. 输入层：
///    * `token_embedding`: 将 Token ID 转换为嵌入向量。
///    * `position_embedding`: 学习一个位置表向量，将其与 Token 嵌入相加。
/// 2. 解码器主架 (`decoder`)：
///    * 包含 `n_layer` 层 `TransformerBlock` 残差堆栈与最终层归一化层 `ln_f`。
/// 3. 输出层 (`lm_head`)：
///    * 线性层，用于将特征层映射为词表中每个 Token 的概率未归一化对数 (Logits)。
pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        token_embedding: Embedding,                 // Token 嵌入层
        position_embedding: Embedding,              // 位置嵌入层
        decoder: TransformerDecoder(config.n_layer),// 堆叠的解码器层与最终归一化层
        lm_head: Linear,                            // 最终输出概率的线性分类投影头

        const Self = @This();

        /// 初始化 GPT 模型中的所有网络层权重
        pub fn init(allocator: std.mem.Allocator, random: std.Random) !Self {
            // 初始化 Token 嵌入矩阵 [vocab_size, n_embd]
            const token_embedding = try Embedding.init(allocator, config.vocab_size, config.n_embd, random);
            errdefer token_embedding.deinit(allocator);
            
            // 初始化位置嵌入矩阵 [block_size, n_embd]
            const position_embedding = try Embedding.init(allocator, config.block_size, config.n_embd, random);
            errdefer position_embedding.deinit(allocator);

            // 初始化 Decoder 主干网络
            const decoder = try TransformerDecoder(config.n_layer).init(allocator, config.n_embd, config.n_head, random);
            errdefer {
                token_embedding.deinit(allocator);
                position_embedding.deinit(allocator);
                decoder.deinit(allocator);
            }

            // 初始化输出映射分类头 [n_embd, vocab_size]
            const lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size, random);
            errdefer {
                token_embedding.deinit(allocator);
                position_embedding.deinit(allocator);
                decoder.deinit(allocator);
                lm_head.deinit(allocator);
            }

            return Self{
                .token_embedding = token_embedding,
                .position_embedding = position_embedding,
                .decoder = decoder,
                .lm_head = lm_head,
            };
        }

        /// 前向推理传播流程
        /// 输入 x 为包含 Token ID 的 2D 整数 Tensor，形状为 [B, T]
        /// 输出为未归一化的预测对数 (Logits)，形状为 3D: [B, T, vocab_size]
        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            // 1. 获取 Token 嵌入向量: [B, T] -> [B, T, n_embd]
            const tok_emb = try self.token_embedding.forward(allocator, graph, x);
            defer if (graph == null) tensor.free(allocator, tok_emb);

            // 2. 生成对应的时间/位置索引 [0, 1, 2, ... T-1]，并将其转换为 2D 位置 Tensor [B, T]
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

            // 3. 获取对应的 Learned 位置嵌入向量: [B, T] -> [B, T, n_embd]
            const pos_emb = try self.position_embedding.forward(allocator, graph, pos_node);
            defer if (graph == null) tensor.free(allocator, pos_emb);

            // 4. 将 Token 嵌入和位置嵌入进行求和融合，作为初始隐藏输入: h = tok_emb + pos_emb
            var h_x = tok_emb;
            if (graph) |g| {
                h_x = try g.add(tok_emb, pos_emb);
            } else {
                h_x = try tok_emb.add(pos_emb, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, h_x);

            // 5. 将混合后的输入送进层叠的 Decoder 主干网络中依次计算
            // 输出形状保持为: [B, T, n_embd]
            const decoder_out = try self.decoder.forward(allocator, graph, h_x);
            defer if (graph == null) tensor.free(allocator, decoder_out);

            // 6. 将输出展平为 2D，以便进行最终分类头的全连接投影计算: [B, T, n_embd] -> [B*T, n_embd]
            var ln_x_2d = decoder_out;
            if (graph) |g| {
                ln_x_2d = try g.reshape(decoder_out, &.{ B * T, config.n_embd });
            } else {
                ln_x_2d = try decoder_out.reshape(&.{ B * T, config.n_embd }, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, ln_x_2d);

            // 7. 进行投影以获得词表空间未归一化的分类 Logits: [B*T, n_embd] -> [B*T, vocab_size]
            const logits_2d = try self.lm_head.forward(allocator, graph, ln_x_2d);
            defer if (graph == null) tensor.free(allocator, logits_2d);

            // 8. 将形状重塑还原成 3D 形式返回: [B, T, vocab_size]
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

    var token_embedding_grad_sum: f32 = 0.0;
    for (gpt.token_embedding.weight.grad) |g| token_embedding_grad_sum += @abs(g);
    try std.testing.expect(token_embedding_grad_sum > 0.0);
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

    for (gpt.token_embedding.weight.data, gpt2.token_embedding.weight.data) |w1, w2| {
        try std.testing.expectEqual(w1, w2);
    }
}


