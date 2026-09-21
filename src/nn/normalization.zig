const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;

/// 均方根层归一化 (Root Mean Square Normalization / RMSNorm)
pub const RMSNorm = struct {
    weight: *Tensor,        // 可学习的缩放因子 gamma (Shape: [dim])
    eps: f32,               // 均方根分母防止除以 0 的极小常数 (epsilon)

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !RMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);

        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        return RMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    pub fn deinit(self: RMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    pub fn zeroGrad(self: RMSNorm) void {
        self.weight.zeroGrad();
    }

    pub fn forward(self: RMSNorm, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.rmsNorm(self.weight, self.eps, allocator, graph);
    }
};

/// 标准层归一化 (Layer Normalization)
pub const LayerNorm = struct {
    weight: *Tensor,        // 可学习的缩放因子 gamma [dim]
    bias: *Tensor,          // 可学习的平移偏置 beta [dim]
    eps: f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !LayerNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, bias);
        @memset(bias.data, 0.0);
        bias.shape = Shape.init(&.{dim});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        return LayerNorm{
            .weight = weight,
            .bias = bias,
            .eps = eps,
        };
    }

    pub fn deinit(self: LayerNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    pub fn zeroGrad(self: LayerNorm) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

    pub fn forward(self: LayerNorm, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const dim = self.weight.shape.dims[0];
        const num_elements = x.data.len;
        const batch_items = num_elements / dim;

        const out = if (graph) |g| try g.tensorND(x.shape.dims[0..x.shape.len], x.requires_grad) else try tensor.zeros(allocator, x.shape.dims[0..x.shape.len]);

        for (0..batch_items) |b| {
            const row = x.data[b * dim .. (b + 1) * dim];
            const out_row = out.data[b * dim .. (b + 1) * dim];

            var sum: f32 = 0.0;
            for (row) |val| sum += val;
            const mean = sum / @as(f32, @floatFromInt(dim));

            var var_sum: f32 = 0.0;
            for (row) |val| {
                const diff = val - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(dim));
            const std_inv = 1.0 / @sqrt(variance + self.eps);

            for (0..dim) |i| {
                out_row[i] = (row[i] - mean) * std_inv * self.weight.data[i] + self.bias.data[i];
            }
        }
        return out;
    }
};

/// 二维批量归一化 (Batch Normalization 2D)
pub const BatchNorm2d = struct {
    num_features: usize,
    eps: f32,
    momentum: f32,
    training: bool,
    gamma: *Tensor,
    beta: *Tensor,
    running_mean: *Tensor,
    running_var: *Tensor,

    pub fn init(allocator: std.mem.Allocator, num_features: usize, eps: f32, momentum: f32) !BatchNorm2d {
        const gamma = try createPersistentTensor(allocator, 1, num_features, true);
        errdefer freePersistentTensor(allocator, gamma);
        @memset(gamma.data, 1.0);
        gamma.shape = Shape.init(&.{num_features});
        gamma.strides = tensor.computeContiguousStrides(gamma.shape);

        const beta = try createPersistentTensor(allocator, 1, num_features, true);
        errdefer freePersistentTensor(allocator, beta);
        @memset(beta.data, 0.0);
        beta.shape = Shape.init(&.{num_features});
        beta.strides = tensor.computeContiguousStrides(beta.shape);

        const running_mean = try createPersistentTensor(allocator, 1, num_features, false);
        errdefer freePersistentTensor(allocator, running_mean);
        @memset(running_mean.data, 0.0);
        running_mean.shape = Shape.init(&.{num_features});
        running_mean.strides = tensor.computeContiguousStrides(running_mean.shape);

        const running_var = try createPersistentTensor(allocator, 1, num_features, false);
        errdefer freePersistentTensor(allocator, running_var);
        @memset(running_var.data, 1.0);
        running_var.shape = Shape.init(&.{num_features});
        running_var.strides = tensor.computeContiguousStrides(running_var.shape);

        return BatchNorm2d{
            .num_features = num_features,
            .eps = eps,
            .momentum = momentum,
            .training = true,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
        };
    }

    pub fn deinit(self: BatchNorm2d, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.gamma);
        freePersistentTensor(allocator, self.beta);
        freePersistentTensor(allocator, self.running_mean);
        freePersistentTensor(allocator, self.running_var);
    }

    pub fn zeroGrad(self: BatchNorm2d) void {
        self.gamma.zeroGrad();
        self.beta.zeroGrad();
    }

    pub fn forward(self: *BatchNorm2d, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        std.debug.assert(x.shape.len == 4); // [N, C, H, W]
        const N = x.shape.dims[0];
        const C = x.shape.dims[1];
        const H = x.shape.dims[2];
        const W = x.shape.dims[3];
        std.debug.assert(C == self.num_features);

        const spatial_size = H * W;
        const total_samples = N * spatial_size;
        const out = if (graph) |g| try g.tensorND(x.shape.dims[0..4], x.requires_grad) else try tensor.zeros(allocator, x.shape.dims[0..4]);

        for (0..C) |c| {
            var mean: f32 = 0.0;
            var variance: f32 = 0.0;

            if (self.training) {
                var sum: f32 = 0.0;
                for (0..N) |n| {
                    const c_slice = x.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                    for (c_slice) |val| sum += val;
                }
                mean = sum / @as(f32, @floatFromInt(total_samples));

                var var_sum: f32 = 0.0;
                for (0..N) |n| {
                    const c_slice = x.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                    for (c_slice) |val| {
                        const diff = val - mean;
                        var_sum += diff * diff;
                    }
                }
                variance = var_sum / @as(f32, @floatFromInt(total_samples));

                self.running_mean.data[c] = (1.0 - self.momentum) * self.running_mean.data[c] + self.momentum * mean;
                self.running_var.data[c] = (1.0 - self.momentum) * self.running_var.data[c] + self.momentum * variance;
            } else {
                mean = self.running_mean.data[c];
                variance = self.running_var.data[c];
            }

            const inv_std = 1.0 / @sqrt(variance + self.eps);
            const g = self.gamma.data[c];
            const b = self.beta.data[c];

            for (0..N) |n| {
                const in_slice = x.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                const out_slice = out.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                for (in_slice, out_slice) |val, *o| {
                    o.* = (val - mean) * inv_std * g + b;
                }
            }
        }
        return out;
    }
};

/// Dropout 随机丢弃正则化层
pub const Dropout = struct {
    p: f32,                 // 丢弃概率 (0.0 <= p < 1.0)
    training: bool = true,  // 是否处于训练模式

    pub fn init(p: f32) Dropout {
        return .{ .p = p, .training = true };
    }

    pub fn forward(self: Dropout, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor, random: ?std.Random) !*Tensor {
        if (!self.training or self.p == 0.0 or random == null) {
            return x;
        }

        const out = if (graph) |g| try g.tensorND(x.shape.dims[0..x.shape.len], x.requires_grad) else try tensor.zeros(allocator, x.shape.dims[0..x.shape.len]);
        const scale = 1.0 / (1.0 - self.p);
        const rand = random.?;

        for (x.data, out.data) |val, *o| {
            if (rand.float(f32) < self.p) {
                o.* = 0.0;
            } else {
                o.* = val * scale;
            }
        }
        return out;
    }
};

/// 二维平均池化 (Average Pooling 2D)
pub const AvgPool2D = struct {
    kernel_size: usize,
    stride: usize,

    pub fn init(kernel_size: usize, stride: usize) AvgPool2D {
        return .{ .kernel_size = kernel_size, .stride = stride };
    }

    pub fn forward(self: AvgPool2D, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        std.debug.assert(x.shape.len == 4);
        const N = x.shape.dims[0];
        const C = x.shape.dims[1];
        const H = x.shape.dims[2];
        const W = x.shape.dims[3];

        const out_h = (H - self.kernel_size) / self.stride + 1;
        const out_w = (W - self.kernel_size) / self.stride + 1;
        const out = if (graph) |g| try g.tensorND(&.{ N, C, out_h, out_w }, x.requires_grad) else try tensor.zeros(allocator, &.{ N, C, out_h, out_w });
        const pool_area = @as(f32, @floatFromInt(self.kernel_size * self.kernel_size));

        for (0..N) |n| {
            for (0..C) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const ih_start = oh * self.stride;
                        const iw_start = ow * self.stride;
                        var sum: f32 = 0.0;

                        for (0..self.kernel_size) |kh| {
                            for (0..self.kernel_size) |kw| {
                                const ih = ih_start + kh;
                                const iw = iw_start + kw;
                                sum += x.data[((n * C + c) * H + ih) * W + iw];
                            }
                        }
                        out.data[((n * C + c) * out_h + oh) * out_w + ow] = sum / pool_area;
                    }
                }
            }
        }
        return out;
    }
};
