const std = @import("std");
const autodiff = @import("autodiff.zig");

pub const NeuralNetwork = struct {
    allocator: std.mem.Allocator,

    // Layer sizes
    ni: usize,
    nh1: usize,
    nh2: usize,
    no: usize,

    // Persistent Weights and Biases (autodiff.Tensor)
    w1: *autodiff.Tensor,
    b1: *autodiff.Tensor,
    w2: *autodiff.Tensor,
    b2: *autodiff.Tensor,
    w3: *autodiff.Tensor,
    b3: *autodiff.Tensor,

    // Momentum velocity buffers (stored as f32 slices)
    v_w1: []f32,
    v_b1: []f32,
    v_w2: []f32,
    v_b2: []f32,
    v_w3: []f32,
    v_b3: []f32,

    pub fn init(allocator: std.mem.Allocator, ni: usize, nh1: usize, nh2: usize, no: usize, seed: u64) !NeuralNetwork {
        const w1 = try createPersistentTensor(allocator, ni, nh1, true);
        errdefer freePersistentTensor(allocator, w1);
        const b1 = try createPersistentTensor(allocator, 1, nh1, true);
        errdefer freePersistentTensor(allocator, b1);

        const w2 = try createPersistentTensor(allocator, nh1, nh2, true);
        errdefer freePersistentTensor(allocator, w2);
        const b2 = try createPersistentTensor(allocator, 1, nh2, true);
        errdefer freePersistentTensor(allocator, b2);

        const w3 = try createPersistentTensor(allocator, nh2, no, true);
        errdefer freePersistentTensor(allocator, w3);
        const b3 = try createPersistentTensor(allocator, 1, no, true);
        errdefer freePersistentTensor(allocator, b3);

        const self = NeuralNetwork{
            .allocator = allocator,
            .ni = ni,
            .nh1 = nh1,
            .nh2 = nh2,
            .no = no,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .w3 = w3,
            .b3 = b3,
            .v_w1 = try allocator.alloc(f32, ni * nh1),
            .v_b1 = try allocator.alloc(f32, nh1),
            .v_w2 = try allocator.alloc(f32, nh1 * nh2),
            .v_b2 = try allocator.alloc(f32, nh2),
            .v_w3 = try allocator.alloc(f32, nh2 * no),
            .v_b3 = try allocator.alloc(f32, no),
        };

        // Initialize momentum velocities to 0
        @memset(self.v_w1, 0.0);
        @memset(self.v_b1, 0.0);
        @memset(self.v_w2, 0.0);
        @memset(self.v_b2, 0.0);
        @memset(self.v_w3, 0.0);
        @memset(self.v_b3, 0.0);

        // Seed RNG and initialize weights
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        // He (Kaiming) initialization
        initializeWeights(random, self.w1.data, ni);
        initializeWeights(random, self.w2.data, self.nh1);
        initializeWeights(random, self.w3.data, self.nh2);

        @memset(self.b1.data, 0.0);
        @memset(self.b2.data, 0.0);
        @memset(self.b3.data, 0.0);

        return self;
    }

    // 类似于 PyTorch 中 nn.Module.forward
    pub fn forward(self: *const NeuralNetwork, graph: *autodiff.Graph, x: *autodiff.Tensor) !*autodiff.Tensor {
        // 第一层：Linear -> ReLU
        const z1 = try graph.matmul(x, self.w1);
        const z1_bias = try graph.addBias(z1, self.b1);
        const a1 = try graph.relu(z1_bias);

        // 第二层：Linear -> ReLU
        const z2 = try graph.matmul(a1, self.w2);
        const z2_bias = try graph.addBias(z2, self.b2);
        const a2 = try graph.relu(z2_bias);

        // 第三层：Linear (logits)
        const z3 = try graph.matmul(a2, self.w3);
        const logits = try graph.addBias(z3, self.b3);

        return logits;
    }

    pub fn deinit(self: *NeuralNetwork) void {
        freePersistentTensor(self.allocator, self.w1);
        freePersistentTensor(self.allocator, self.b1);
        freePersistentTensor(self.allocator, self.w2);
        freePersistentTensor(self.allocator, self.b2);
        freePersistentTensor(self.allocator, self.w3);
        freePersistentTensor(self.allocator, self.b3);
        self.allocator.free(self.v_w1);
        self.allocator.free(self.v_b1);
        self.allocator.free(self.v_w2);
        self.allocator.free(self.v_b2);
        self.allocator.free(self.v_w3);
        self.allocator.free(self.v_b3);
    }

    pub fn zeroGrad(self: *NeuralNetwork) void {
        self.w1.zeroGrad();
        self.b1.zeroGrad();
        self.w2.zeroGrad();
        self.b2.zeroGrad();
        self.w3.zeroGrad();
        self.b3.zeroGrad();
    }

    pub fn updateWeights(self: *NeuralNetwork, lr: f32, beta: f32) void {
        updateLayerWeights(self.w1.data, self.w1.grad, self.v_w1, lr, beta);
        updateLayerWeights(self.b1.data, self.b1.grad, self.v_b1, lr, beta);
        updateLayerWeights(self.w2.data, self.w2.grad, self.v_w2, lr, beta);
        updateLayerWeights(self.b2.data, self.b2.grad, self.v_b2, lr, beta);
        updateLayerWeights(self.w3.data, self.w3.grad, self.v_w3, lr, beta);
        updateLayerWeights(self.b3.data, self.b3.grad, self.v_b3, lr, beta);
    }

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

    pub fn save(self: *const NeuralNetwork, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        try writer.writeAll(std.mem.sliceAsBytes(self.w1.data));
        try writer.writeAll(std.mem.sliceAsBytes(self.b1.data));
        try writer.writeAll(std.mem.sliceAsBytes(self.w2.data));
        try writer.writeAll(std.mem.sliceAsBytes(self.b2.data));
        try writer.writeAll(std.mem.sliceAsBytes(self.w3.data));
        try writer.writeAll(std.mem.sliceAsBytes(self.b3.data));

        try file_writer.flush();
    }

    pub fn load(self: *NeuralNetwork, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_reader = file.reader(io, &buf);
        const reader = &file_reader.interface;

        try reader.readSliceAll(std.mem.sliceAsBytes(self.w1.data));
        try reader.readSliceAll(std.mem.sliceAsBytes(self.b1.data));
        try reader.readSliceAll(std.mem.sliceAsBytes(self.w2.data));
        try reader.readSliceAll(std.mem.sliceAsBytes(self.b2.data));
        try reader.readSliceAll(std.mem.sliceAsBytes(self.w3.data));
        try reader.readSliceAll(std.mem.sliceAsBytes(self.b3.data));
    }
};

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
