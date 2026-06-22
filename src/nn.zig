const std = @import("std");
const autodiff = @import("autodiff.zig");

// ============================================================================
// 1. PyTorch-like Linear (全连接/线性层) 模块定义
// ============================================================================
pub const Linear = struct {
    weight: *autodiff.Tensor,   // 权重矩阵
    bias: *autodiff.Tensor,     // 偏置向量
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
    pub fn forward(self: Linear, graph: *autodiff.Graph, x: *autodiff.Tensor) !*autodiff.Tensor {
        const z = try graph.matmul(x, self.weight);
        return try graph.addBias(z, self.bias);
    }
};

// ============================================================================
// 2. Comptime 编译期反射函数（实现类似于 PyTorch 基类继承的自动化处理）
// ============================================================================

// 自动扫描结构体，调用子层（如 Linear）的 deinit 或释放对应的 Tensor 和 Slice 字段
pub fn deinitModel(model: anytype) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            @field(model, field.name).deinit(model.allocator);
        } else if (field.type == *autodiff.Tensor) {
            freePersistentTensor(model.allocator, @field(model, field.name));
        } else if (field.type == []f32) {
            model.allocator.free(@field(model, field.name));
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
        } else if (field.type == *autodiff.Tensor) {
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
        } else if (field.type == *autodiff.Tensor) {
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

// 自动扫描子层并将所有持久化参数二进制数据写入磁盘
pub fn saveModel(model: anytype, io: std.Io, file_path: []const u8) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFile(io, file_path, .{});
    defer file.close(io);

    var buf: [65536]u8 = undefined;
    var file_writer = file.writer(io, &buf);
    const writer = &file_writer.interface;

    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            const layer = @field(model, field.name);
            try writer.writeAll(std.mem.sliceAsBytes(layer.weight.data));
            try writer.writeAll(std.mem.sliceAsBytes(layer.bias.data));
        } else if (field.type == *autodiff.Tensor) {
            try writer.writeAll(std.mem.sliceAsBytes(@field(model, field.name).data));
        }
    }
    try file_writer.flush();
}

// 从磁盘中二进制还原所有子层参数数据
pub fn loadModel(model: anytype, io: std.Io, file_path: []const u8) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    var buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &buf);
    const reader = &file_reader.interface;

    inline for (info.@"struct".fields) |field| {
        if (field.type == Linear) {
            const layer = @field(model, field.name);
            try reader.readSliceAll(std.mem.sliceAsBytes(layer.weight.data));
            try reader.readSliceAll(std.mem.sliceAsBytes(layer.bias.data));
        } else if (field.type == *autodiff.Tensor) {
            try reader.readSliceAll(std.mem.sliceAsBytes(@field(model, field.name).data));
        }
    }
}

// ============================================================================
// 3. 多层感知机 NeuralNetwork（只需定义层结构和 forward）
// ============================================================================
pub const NeuralNetwork = struct {
    allocator: std.mem.Allocator,

    // 定义子层（类似于 PyTorch 的 fc1, fc2, fc3）
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,

    // 初始化模型
    pub fn init(allocator: std.mem.Allocator, ni: usize, nh1: usize, nh2: usize, no: usize, seed: u64) !NeuralNetwork {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        const fc1 = try Linear.init(allocator, ni, nh1, random);
        errdefer fc1.deinit(allocator);

        const fc2 = try Linear.init(allocator, nh1, nh2, random);
        errdefer fc2.deinit(allocator);

        const fc3 = try Linear.init(allocator, nh2, no, random);
        errdefer fc3.deinit(allocator);

        return NeuralNetwork{
            .allocator = allocator,
            .fc1 = fc1,
            .fc2 = fc2,
            .fc3 = fc3,
        };
    }

    // 如下核心基类方法已被 Comptime 自动托管实现！完全零模板代码（Boilerplate）

    pub fn deinit(self: *NeuralNetwork) void {
        deinitModel(self);
    }

    pub fn zeroGrad(self: *NeuralNetwork) void {
        zeroGradModel(self);
    }

    pub fn updateWeights(self: *NeuralNetwork, lr: f32, beta: f32) void {
        updateWeightsModel(self, lr, beta);
    }

    pub fn save(self: *const NeuralNetwork, io: std.Io, file_path: []const u8) !void {
        try saveModel(self, io, file_path);
    }

    pub fn load(self: *NeuralNetwork, io: std.Io, file_path: []const u8) !void {
        try loadModel(self, io, file_path);
    }

    // 用户只需专注定义前向传播逻辑
    pub fn forward(self: *const NeuralNetwork, graph: *autodiff.Graph, x: *autodiff.Tensor) !*autodiff.Tensor {
        const x1 = try self.fc1.forward(graph, x);
        const a1 = try graph.relu(x1);

        const x2 = try self.fc2.forward(graph, a1);
        const a2 = try graph.relu(x2);

        return try self.fc3.forward(graph, a2);
    }
};

// ============================================================================
// 4. 底层数学与内存辅助函数
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

fn createPersistentTensor(allocator: std.mem.Allocator, rows: usize, cols: usize, requires_grad: bool) !*autodiff.Tensor {
    const t = try allocator.create(autodiff.Tensor);
    t.* = autodiff.Tensor{
        .data = try allocator.alloc(f32, rows * cols),
        .grad = if (requires_grad) try allocator.alloc(f32, rows * cols) else &.{},
        .rows = rows,
        .cols = cols,
        .requires_grad = requires_grad,
        .creator = null,
    };
    @memset(t.data, 0.0);
    if (requires_grad) {
        @memset(t.grad, 0.0);
    }
    return t;
}

fn freePersistentTensor(allocator: std.mem.Allocator, t: *autodiff.Tensor) void {
    allocator.free(t.data);
    if (t.requires_grad) {
        allocator.free(t.grad);
    }
    allocator.destroy(t);
}
