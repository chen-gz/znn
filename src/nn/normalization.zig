const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;

/// 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
pub const RMSNorm = struct {
    weight: *Tensor, // 可学习的缩放因子 gamma (Shape: [dim])
    eps: f32, // 均方根分母防止除以 0 的极小常数 (epsilon, eps)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
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

    pub fn setName(self: *RMSNorm, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
    }

    pub fn setNameFormatted(self: *RMSNorm, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("rmsnorm");
        }
    }

    pub fn getName(self: *const RMSNorm) ?[]const u8 {
        return self.name;
    }

    pub fn deinit(self: RMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    pub fn zeroGrad(self: RMSNorm) void {
        self.weight.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot \\gamma";

    pub fn registerFormula(self: *const RMSNorm, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: RMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// 标准层归一化 (Layer Normalization, LayerNorm)
pub const LayerNorm = struct {
    weight: *Tensor, // 可学习的缩放因子 gamma [dim]
    bias: *Tensor, // 可学习的平移偏置 beta [dim]
    eps: f32,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "LayerNorm",

    pub fn setName(self: *LayerNorm, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
        self.bias.setNameFormatted("{s}.bias", .{self.name.?});
    }

    pub fn setNameFormatted(self: *LayerNorm, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("layernorm");
        }
    }

    pub fn getName(self: *const LayerNorm) ?[]const u8 {
        return self.name;
    }

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

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta";

    pub fn registerFormula(self: *const LayerNorm, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: LayerNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
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
    name_buf: [64]u8 = undefined,
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

    pub fn setName(self: *BatchNorm2d, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.gamma.setNameFormatted("{s}.gamma", .{self.name.?});
        self.beta.setNameFormatted("{s}.beta", .{self.name.?});
        self.running_mean.setNameFormatted("{s}.running_mean", .{self.name.?});
        self.running_var.setNameFormatted("{s}.running_var", .{self.name.?});
    }

    pub fn setNameFormatted(self: *BatchNorm2d, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("batchnorm");
        }
    }

    pub fn getName(self: *const BatchNorm2d) ?[]const u8 {
        return self.name;
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

    /// 模块标准数学变换公式
    pub const formula = "y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} \\odot \\gamma + \\beta";

    pub fn registerFormula(self: *const BatchNorm2d, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: *BatchNorm2d, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
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
    name_buf: [64]u8 = undefined,
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

    pub fn setName(self: *Dropout, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
    }

    pub fn setNameFormatted(self: *Dropout, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("dropout");
        }
    }

    pub fn getName(self: *const Dropout) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const Dropout, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: Dropout, graph: *autodiff.Graph, x: *Tensor, random: ?std.Random) !*Tensor {
        if (!self.training or self.p == 0.0 or random == null) {
            return x;
        }
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        return try graph.dropout(x, self.p, random.?);
    }
};

/// 二维平均池化 (2-Dimensional Average Pooling, AvgPool2D)
pub const AvgPool2D = struct {
    kernel_size: usize,
    stride: usize,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "AvgPool2D",

    pub const formula = "y = \\frac{1}{k^2} \\sum_{k \\times k} x";

    pub fn init(kernel_size: usize, stride: usize) AvgPool2D {
        return .{ .kernel_size = kernel_size, .stride = stride };
    }

    pub fn setName(self: *AvgPool2D, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
    }

    pub fn setNameFormatted(self: *AvgPool2D, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("avgpool2d");
        }
    }

    pub fn getName(self: *const AvgPool2D) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const AvgPool2D, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: AvgPool2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        return try graph.avgpool2d(x, self.kernel_size, self.stride);
    }
};
