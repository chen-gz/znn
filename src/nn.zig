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
    pub fn forward(self: Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const z = try graph.matmul(x, self.weight);
        return try graph.addBias(z, self.bias);
    }
};

// ============================================================================
// 2. Comptime 编译期反射函数（用于底层递归扫描，由 Module 包装器调用）
// ============================================================================

// 自动扫描结构体，调用子层（如 Linear）的 deinit 或释放对应的 Tensor 和 Slice 字段
pub fn deinitModel(model: anytype, allocator: std.mem.Allocator) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            @field(model, field.name).deinit(allocator);
        } else if (field.type == *Tensor) {
            freePersistentTensor(allocator, @field(model, field.name));
        } else if (field.type == []f32) {
            allocator.free(@field(model, field.name));
        }
    }
}

// 自动扫描结构体中的子层或 Tensor，并将它们的梯度统一清零
pub fn zeroGradModel(model: anytype) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            @field(model, field.name).zeroGrad();
        } else if (field.type == *Tensor) {
            @field(model, field.name).zeroGrad();
        }
    }
}

// 自动扫描子层并进行参数梯度更新
pub fn updateWeightsModel(model: anytype, lr: f32, beta: f32) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            @field(model, field.name).updateWeights(lr, beta);
        } else if (field.type == *Tensor) {
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
        }
    }
}

// 自动扫描子层并使用 Safetensors 格式将所有持久化参数二进制数据写入磁盘
pub fn saveModel(model: anytype, io: std.Io, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFile(io, file_path, .{});
    defer file.close(io);

    // 1. 构建 JSON header
    var json_buf: std.ArrayList(u8) = .empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{");
    var first = true;
    var offset: usize = 0;

    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            const layer = @field(model, field.name);
            // weight
            if (!first) try json_buf.appendSlice(allocator, ",") else first = false;
            try writeTensorEntry(&json_buf, allocator, field.name ++ ".weight", layer.weight, &offset);
            // bias
            try json_buf.appendSlice(allocator, ",");
            try writeTensorEntry(&json_buf, allocator, field.name ++ ".bias", layer.bias, &offset);
        } else if (field.type == *Tensor) {
            const t = @field(model, field.name);
            if (!first) try json_buf.appendSlice(allocator, ",") else first = false;
            try writeTensorEntry(&json_buf, allocator, field.name, t, &offset);
        }
    }
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
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            const layer = @field(model, field.name);
            try writer.writeAll(std.mem.sliceAsBytes(layer.weight.data));
            try writer.writeAll(std.mem.sliceAsBytes(layer.bias.data));
        } else if (field.type == *Tensor) {
            try writer.writeAll(std.mem.sliceAsBytes(@field(model, field.name).data));
        }
    }
    try file_writer.flush();
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

// 从磁盘中二进制还原所有子层参数数据（使用 Safetensors 格式）
pub fn loadModel(model: anytype, io: std.Io, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
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
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            const layer = @field(model, field.name);
            try loadTensorData(reader, meta_obj, field.name ++ ".weight", layer.weight, &current_offset);
            try loadTensorData(reader, meta_obj, field.name ++ ".bias", layer.bias, &current_offset);
        } else if (field.type == *Tensor) {
            const t = @field(model, field.name);
            try loadTensorData(reader, meta_obj, field.name, t, &current_offset);
        }
    }
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

        // 自动托管模型的初始化，并将参数转发给内部具体模型结构体的 init 方法
        pub fn init(allocator: std.mem.Allocator, ni: usize, nh1: usize, nh2: usize, no: usize, seed: u64) !Self {
            return Self{
                .allocator = allocator,
                .inner = try T.init(allocator, ni, nh1, nh2, no, seed),
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
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return try self.inner.forward(graph, x);
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
