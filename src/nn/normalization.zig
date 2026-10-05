const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const enterModuleScope = core.enterModuleScope;
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;

/// 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
pub const RMSNorm = struct {
    weight: *Tensor, // 可学习的缩放因子 gamma (Shape: [dim])
    eps: f32, // 均方根分母防止除以 0 的极小常数 (epsilon, eps)
    name: ?[]const u8 = null,
    module_type: []const u8 = "RMSNorm",

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

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot \\gamma";

    pub fn forward(self: *const RMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// 标准层归一化 (Layer Normalization, LayerNorm)
pub const LayerNorm = struct {
    weight: *Tensor, // 可学习的缩放因子 gamma [dim]
    bias: *Tensor, // 可学习的平移偏置 beta [dim]
    eps: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "LayerNorm",

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

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta";

    pub fn forward(self: *const LayerNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.layerNorm(x, self.weight, self.bias, self.eps);
    }
};

/// 二维批量归一化 (2-Dimensional Batch Normalization, BatchNorm2d)
pub const BatchNorm2d = struct {
    num_features: usize,
    eps: f32,
    momentum: f32,
    training: bool,
    gamma: *Tensor,
    beta: *Tensor,
    running_mean: *Tensor,
    running_var: *Tensor,
    name: ?[]const u8 = null,
    module_type: []const u8 = "BatchNorm2d",

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

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} \\odot \\gamma + \\beta";

    pub fn forward(self: *BatchNorm2d, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.batchNorm2d(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training,
        );
    }
};

/// 随机失活正则化层 (Dropout Regularization, Dropout)
pub const Dropout = struct {
    p: f32 = 0.5, // 丢弃概率 (0.0 <= p < 1.0)
    training: bool = true, // 是否处于训练模式
    name: ?[]const u8 = null,
    module_type: []const u8 = "Dropout",

    pub const formula = "y = \\frac{m \\odot x}{1 - p}";

    pub const default: Dropout = .{};

    pub fn defaultOptions() Dropout {
        return .{};
    }

    pub fn initDefault() Dropout {
        return .{};
    }

    pub fn init(p: f32) Dropout {
        return .{ .p = p, .training = true };
    }

    pub fn forward(self: *const Dropout, graph: *autodiff.Graph, x: *Tensor, random: ?std.Random) !*Tensor {
        if (!self.training or self.p == 0.0 or random == null) {
            return x;
        }
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.dropout(x, self.p, random.?);
    }
};

/// 一维最大池化 (1-Dimensional Max Pooling, MaxPool1D)
pub const MaxPool1D = struct {
    pool_size: usize,
    stride: usize,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MaxPool1D",

    pub const Options = tensor.PoolOptions;
    pub const formula = "y = \\max_{k}(x)";

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(pool_size: usize, options: Options) MaxPool1D {
        return .{
            .pool_size = pool_size,
            .stride = options.resolveStride(pool_size),
            .padding = options.padding,
        };
    }

    pub fn forward(self: *const MaxPool1D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.maxpool1d(x, self.pool_size, .{ .stride = self.stride, .padding = self.padding });
    }
};

/// 二维最大池化 (2-Dimensional Max Pooling, MaxPool2D)
pub const MaxPool2D = struct {
    pool_size: usize,
    stride: usize,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MaxPool2D",

    pub const Options = tensor.PoolOptions;
    pub const formula = "y = \\max_{k \\times k}(x)";

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(pool_size: usize, options: Options) MaxPool2D {
        return .{
            .pool_size = pool_size,
            .stride = options.resolveStride(pool_size),
            .padding = options.padding,
        };
    }

    pub fn forward(self: *const MaxPool2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.maxpool2d(x, self.pool_size, .{ .stride = self.stride, .padding = self.padding });
    }
};

/// 一维平均池化 (1-Dimensional Average Pooling, AvgPool1D)
pub const AvgPool1D = struct {
    kernel_size: usize,
    stride: usize,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "AvgPool1D",

    pub const Options = tensor.PoolOptions;
    pub const formula = "y = \\frac{1}{k} \\sum_{k} x";

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(kernel_size: usize, options: Options) AvgPool1D {
        return .{
            .kernel_size = kernel_size,
            .stride = options.resolveStride(kernel_size),
            .padding = options.padding,
        };
    }

    pub fn forward(self: *const AvgPool1D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.avgpool1d(x, self.kernel_size, .{ .stride = self.stride, .padding = self.padding });
    }
};

/// 二维平均池化 (2-Dimensional Average Pooling, AvgPool2D)
pub const AvgPool2D = struct {
    kernel_size: usize,
    stride: usize,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "AvgPool2D",

    pub const Options = tensor.PoolOptions;
    pub const formula = "y = \\frac{1}{k^2} \\sum_{k \\times k} x";

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(kernel_size: usize, options: Options) AvgPool2D {
        return .{
            .kernel_size = kernel_size,
            .stride = options.resolveStride(kernel_size),
            .padding = options.padding,
        };
    }

    pub fn forward(self: *const AvgPool2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.avgpool2d(x, self.kernel_size, .{ .stride = self.stride, .padding = self.padding });
    }
};
