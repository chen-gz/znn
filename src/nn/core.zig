const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const module = @import("module.zig");
pub const walk = module.walk;
pub const deinitModel = module.deinitModel;
pub const zeroGradModel = module.zeroGradModel;
pub const parameters = module.parameters;
pub const NamedParameter = module.NamedParameter;
pub const NamedParameterList = module.NamedParameterList;
pub const namedParameters = module.namedParameters;
pub const numParameters = module.numParameters;
pub const setRequiresGrad = module.setRequiresGrad;
pub const setTrainingModel = module.setTrainingModel;
pub const trainModel = module.trainModel;
pub const evalModel = module.evalModel;
pub const nameModules = module.nameModules;
pub const enterModuleScope = module.enterModuleScope;
const applyCustomInit = module.applyCustomInit;

// ============================================================================
// 底层权重初始化与持久化张量内存辅助
// ============================================================================

pub const init_mod = @import("init.zig");
pub const Nonlinearity = init_mod.Nonlinearity;
pub const calculateGain = init_mod.calculateGain;
pub const InitMethod = init_mod.InitMethod;
pub const InitOptions = init_mod.InitOptions;
pub const normalRandom = init_mod.normalRandom;
pub const initWeights = init_mod.initWeights;

pub fn createPersistentTensor(allocator: std.mem.Allocator, rows: usize, cols: usize, requires_grad: bool) !*Tensor {
    const t = try allocator.create(Tensor);
    const shape = Shape.init(&.{ rows, cols });
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

pub fn freePersistentTensor(allocator: std.mem.Allocator, t: *Tensor) void {
    allocator.free(t.data);
    if (t.grad.len > 0) allocator.free(t.grad);
    allocator.destroy(t);
}

// ============================================================================
// 线性层 (Linear) 与二维卷积模块 (2D Convolution, Conv2D / 2D Transposed Convolution, ConvTranspose2D)
// ============================================================================

pub const Linear = struct {
    weight: *Tensor,
    bias: *Tensor,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Linear",

    /// 构造线性层：只分配参数内存 (权重与偏置全零)，不做任何数值初始化。
    /// 参数数值由 nn.initModel (外部 customInit 或内置 resetParameters)、Graph.initWeights 或直接调用 resetParameters 设置。
    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Linear {
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        return Linear{
            .weight = weight,
            .bias = bias,
        };
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充权重与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *Linear, random: std.Random, options: InitOptions) void {
        const in_features = self.weight.shape.dims[0];
        const out_features = self.weight.shape.dims[1];
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, in_features, out_features, w_init);
        initWeights(random, self.bias.data, in_features, out_features, options.bias_init);
    }

    /// 模块标准数学变换公式
    pub const formula = "y = x W^T + b";

    pub fn forward(self: *const Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        const z = try graph.matmul(x, self.weight);
        return try graph.addBias(z, self.bias);
    }
};

pub const Conv1D = struct {
    weight: *Tensor,
    bias: *Tensor,
    stride: usize = 1,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Conv1D",

    pub const Options = tensor.ConvOptions;

    /// 构造一维卷积层：只分配参数内存 (卷积核与偏置全零)，不做任何数值初始化
    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: Options,
    ) !Conv1D {
        if (options.stride == 0) return error.InvalidStride;
        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        weight.shape = Shape.init(&.{ out_channels, in_channels, kernel_size });
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, out_channels, true);
        errdefer freePersistentTensor(allocator, bias);
        bias.shape = Shape.init(&.{out_channels});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        return Conv1D{
            .weight = weight,
            .bias = bias,
            .stride = options.stride,
            .padding = options.padding,
        };
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充卷积核与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *Conv1D, random: std.Random, options: InitOptions) void {
        const out_channels = self.weight.shape.dims[0];
        const in_channels = self.weight.shape.dims[1];
        const kernel_size = self.weight.shape.dims[2];
        const fan_in = in_channels * kernel_size;
        const fan_out = out_channels * kernel_size;
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, fan_in, fan_out, w_init);
        initWeights(random, self.bias.data, fan_in, fan_out, options.bias_init);
    }

    /// 模块标准数学变换公式
    pub const formula = "y = x \\ast W + b";

    pub fn forward(self: *const Conv1D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.conv1d(x, self.weight, self.bias, .{ .stride = self.stride, .padding = self.padding });
    }
};

pub const Conv2D = struct {
    weight: *Tensor,
    bias: *Tensor,
    stride: usize = 1,
    padding: usize = 0,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Conv2D",

    pub const Options = tensor.ConvOptions;

    /// 构造卷积层：只分配参数内存 (卷积核与偏置全零)，不做任何数值初始化
    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: Options,
    ) !Conv2D {
        if (options.stride == 0) return error.InvalidStride;
        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        weight.shape = Shape.init(&.{ out_channels, in_channels, kernel_size, kernel_size });
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, out_channels, true);
        errdefer freePersistentTensor(allocator, bias);
        bias.shape = Shape.init(&.{out_channels});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        return Conv2D{
            .weight = weight,
            .bias = bias,
            .stride = options.stride,
            .padding = options.padding,
        };
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充卷积核与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *Conv2D, random: std.Random, options: InitOptions) void {
        const out_channels = self.weight.shape.dims[0];
        const in_channels = self.weight.shape.dims[1];
        const kernel_size = self.weight.shape.dims[2];
        const fan_in = in_channels * kernel_size * kernel_size;
        const fan_out = out_channels * kernel_size * kernel_size;
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, fan_in, fan_out, w_init);
        initWeights(random, self.bias.data, fan_in, fan_out, options.bias_init);
    }

    /// 模块标准数学变换公式
    pub const formula = "y = x \\ast W + b";

    pub fn forward(self: *const Conv2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.conv2d(x, self.weight, self.bias, .{ .stride = self.stride, .padding = self.padding });
    }
};

pub const ConvTranspose2D = struct {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight: *Tensor,
    bias: ?*Tensor,
    name: ?[]const u8 = null,
    module_type: []const u8 = "ConvTranspose2D",

    /// 构造反卷积层：只分配参数内存 (反卷积核与可选偏置全零)，不做任何数值初始化
    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) !ConvTranspose2D {
        const weight = try createPersistentTensor(allocator, 1, in_channels * out_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        weight.shape = Shape.init(&.{ in_channels, out_channels, kernel_size, kernel_size });
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        var bias: ?*Tensor = null;
        if (use_bias) {
            const b = try createPersistentTensor(allocator, 1, out_channels, true);
            errdefer freePersistentTensor(allocator, b);
            b.shape = Shape.init(&.{out_channels});
            b.strides = tensor.computeContiguousStrides(b.shape);
            bias = b;
        }

        return ConvTranspose2D{
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .weight = weight,
            .bias = bias,
        };
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充反卷积核与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *ConvTranspose2D, random: std.Random, options: InitOptions) void {
        const fan_in = self.in_channels * self.kernel_size * self.kernel_size;
        const fan_out = self.out_channels * self.kernel_size * self.kernel_size;
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, fan_in, fan_out, w_init);
        if (self.bias) |b| {
            initWeights(random, b.data, fan_in, fan_out, options.bias_init);
        }
    }

    pub fn forward(self: *const ConvTranspose2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.convTranspose2D(x, self.weight, self.bias, self.stride, self.padding);
    }
};

// ============================================================================
// 泛型模型元编程与参数搜集
// ============================================================================

/// 参数初始化状态统计：只统计可训练的权重矩阵。
/// 向量形参数 (除最后一维外各维均为 1，如 [n] / [1, n] 的偏置与归一化 γ / β) 可能按设计为常量或全零，不作为判据；
/// 单个权重矩阵全零也可能是设计使然 (如 LoRA 旁路 B)，因此只在所有权重矩阵都全零时才判定为未初始化。
pub const ParameterInitReport = struct {
    /// 可训练权重矩阵 (非向量形参数) 的数量
    weight_tensors: usize = 0,
    /// 其中数据全为 0 的权重矩阵数量
    zero_weight_tensors: usize = 0,

    /// 所有可训练权重矩阵都全为 0：通常意味着只调用了 init 而遗漏了 nn.initModel / Graph.initWeights
    pub fn looksUninitialized(self: ParameterInitReport) bool {
        return self.weight_tensors > 0 and self.zero_weight_tensors == self.weight_tensors;
    }
};

/// 统计参数列表中可训练权重矩阵的初始化状态
pub fn inspectParameterInit(params: []const *Tensor) ParameterInitReport {
    var report = ParameterInitReport{};
    for (params) |p| {
        if (!p.requires_grad or isVectorShaped(p)) continue;
        report.weight_tensors += 1;
        const all_zero = for (p.data) |v| {
            if (v != 0.0) break false;
        } else true;
        if (all_zero) report.zero_weight_tensors += 1;
    }
    return report;
}

fn isVectorShaped(t: *const Tensor) bool {
    if (t.shape.len == 0) return true;
    for (t.shape.dims[0 .. t.shape.len - 1]) |d| {
        if (d != 1) return false;
    }
    return true;
}

/// 调试构建 (Debug) 下检查参数是否已初始化：所有可训练权重矩阵都全为 0 时输出警告。
/// 优化器在构造时调用 (训练开始前)；非 Debug 构建下为空操作，不产生任何开销。
pub fn warnIfParametersUninitialized(params: []const *Tensor) void {
    if (@import("builtin").mode != .Debug) return;
    const report = inspectParameterInit(params);
    if (report.looksUninitialized()) {
        std.log.warn(
            "all {d} trainable weight matrices are zero before training; layer init only allocates memory, " ++
                "build a forward graph and call nn.initModel(&model, &graph, random) or nn.initModelWithSample first",
            .{report.weight_tensors},
        );
    }
}

/// 模型参数初始化入口：必须在前向计算图建立之后调用。库内所有层的 `init` 只分配内存，参数数值统一在此设置：
/// 1. 外部指定：递归遍历模型，模块类型定义了 `customInit` (由库外用户模块定义) 时调用之，并将该模块内所有
///    可训练参数标记为 `is_custom_initialized = true`。支持两种签名：
///    - `pub fn customInit(self: *Self) void`：确定性初始化 (常量、预设矩阵等)，不消耗随机数；
///    - `pub fn customInit(self: *Self, random: std.Random) void`：需要随机数的自定义初始化；
/// 2. 外部未指定：其余参数由 `graph.initWeights` 依据计算图中各参数的下游激活函数推导初始化策略
///    (ReLU / GELU / SiLU -> He、Tanh / Sigmoid -> Xavier、归一化 γ -> 1、偏置 -> 0、Embedding -> Normal(0, 0.02) 等)，
///    已被 customInit 初始化的参数不会被覆盖。
/// `graph` 必须是开启梯度记录 (`Graph.init`) 并已对该模型执行过一次前向计算的计算图；
/// Debug 构建下若有未出现在计算图中的参数 (前向未经过的分支)，输出警告，这些参数保持 `init` 分配时的默认值。
pub fn initModel(model: anytype, graph: *autodiff.Graph, random: std.Random) !void {
    if (!graph.enable_grad) return error.GraphNotRecordingOps;
    applyCustomInit(model, random);
    graph.initWeights(random);
    if (@import("builtin").mode == .Debug) try warnParametersOutsideGraph(model, graph);
}

/// 便捷入口：用样本输入建立一次前向计算图，再调用 `initModel` 完成依据计算图的参数初始化。
/// `sample_args` 为传给 `model.forward(graph, ...)` 的除计算图外的参数：单个值 (如 `x`) 或元组 (如 `.{ x, h_0 }`)。
pub fn initModelWithSample(model: anytype, allocator: std.mem.Allocator, random: std.Random, sample_args: anytype) !void {
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    const T = @TypeOf(model.*);
    const result = @call(.auto, T.forward, .{ forwardSelf(model), &graph } ++ forwardArgs(sample_args));
    if (@typeInfo(@TypeOf(result)) == .error_union) _ = try result;
    try initModel(model, &graph, random);
}

/// 以参数 `Args` (单个值或元组，见 `callForward`) 调用 `T.forward` 时的返回类型
pub fn ForwardResult(comptime T: type, comptime Args: type) type {
    return @TypeOf(@call(.auto, T.forward, .{ @as(ForwardSelf(T), undefined), @as(*autodiff.Graph, undefined) } ++ @as(ForwardArgs(Args), undefined)));
}

/// 以 `args` 调用 `model.forward(graph, ...)`，等价于 PyTorch 的 `module(*args)`：
/// `args` 为元组时展开为多个参数 (如 `.{ x, h_0 }`)，否则作为单个参数 (如 `x`)；
/// 自动适配 `forward` 按指针或按值接收 `self`
pub fn callForward(model: anytype, graph: *autodiff.Graph, args: anytype) ForwardResult(@TypeOf(model.*), @TypeOf(args)) {
    return @call(.auto, @TypeOf(model.*).forward, .{ forwardSelf(model), graph } ++ forwardArgs(args));
}

fn ForwardSelf(comptime T: type) type {
    return @typeInfo(@TypeOf(T.forward)).@"fn".params[0].type.?;
}

fn forwardSelf(model: anytype) ForwardSelf(@TypeOf(model.*)) {
    return if (comptime @typeInfo(ForwardSelf(@TypeOf(model.*))) == .pointer) model else model.*;
}

fn isTuple(comptime A: type) bool {
    const info = @typeInfo(A);
    return info == .@"struct" and info.@"struct".is_tuple;
}

fn ForwardArgs(comptime A: type) type {
    return if (isTuple(A)) A else std.meta.Tuple(&.{A});
}

fn forwardArgs(args: anytype) ForwardArgs(@TypeOf(args)) {
    return if (comptime isTuple(@TypeOf(args))) args else .{args};
}

/// Debug 构建下统计未出现在计算图中、且未经 customInit 初始化的可训练参数并输出警告
fn warnParametersOutsideGraph(model: anytype, graph: *autodiff.Graph) !void {
    const allocator = graph.arenaAllocator();
    const params = try parameters(model, allocator);
    var reached = std.AutoHashMap(*const Tensor, void).init(allocator);
    for (graph.ops.items) |op| {
        for (op.inputs) |inp| try reached.put(inp, {});
    }
    for (graph.tensors.items) |gt| try reached.put(gt, {});
    var missing: usize = 0;
    for (params) |p| {
        if (p.is_custom_initialized or reached.contains(p)) continue;
        missing += 1;
    }
    if (missing > 0) {
        std.log.warn(
            "{d} of {d} trainable parameters were not reached by the forward graph and keep their allocation defaults; " ++
                "run a forward pass that exercises every module before nn.initModel, or initialize them in customInit",
            .{ missing, params.len },
        );
    }
}

/// 模型包装器 (类似 PyTorch `nn.Module` 实例)：持有分配器与内部模型，并自带存放模块 / 参数名称的 arena
pub fn Module(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        inner: T,
        names: std.heap.ArenaAllocator,

        const Self = @This();

        /// 包装模型，并按字段路径为全部子模块与参数命名 (根模块不命名，如 `decoder.h.0.attn`；需要前缀时调用 `setName`)
        pub fn init(allocator: std.mem.Allocator, inner: T) !Self {
            var self = Self{
                .allocator = allocator,
                .inner = inner,
                .names = std.heap.ArenaAllocator.init(allocator),
            };
            errdefer self.names.deinit();
            try nameModules(&self.inner, self.names.allocator(), "");
            return self;
        }

        /// 以 `root` 为根模块名重新命名全部子模块与参数 (如 `setName("gpt")` 得到 `gpt.decoder.h.0.attn`)
        pub fn setName(self: *Self, root: []const u8) !void {
            _ = self.names.reset(.retain_capacity);
            try nameModules(&self.inner, self.names.allocator(), root);
        }

        /// 依据前向计算图初始化内部模型参数 (见 `initModel`)：graph 需已对本模块执行过一次前向计算
        pub fn initParameters(self: *Self, graph: *autodiff.Graph, random: std.Random) !void {
            try initModel(&self.inner, graph, random);
        }

        /// 用样本输入建立前向计算图后初始化内部模型参数 (见 `initModelWithSample`)
        pub fn initParametersWithSample(self: *Self, random: std.Random, sample_args: anytype) !void {
            try initModelWithSample(&self.inner, self.allocator, random, sample_args);
        }

        pub fn deinit(self: *Self) void {
            deinitModel(&self.inner, self.allocator);
            self.names.deinit();
        }

        pub fn zeroGrad(self: *Self) void {
            zeroGradModel(&self.inner);
        }

        pub fn setTraining(self: *Self, is_training: bool) void {
            setTrainingModel(&self.inner, is_training);
        }

        pub fn train(self: *Self) void {
            trainModel(&self.inner);
        }

        pub fn eval(self: *Self) void {
            evalModel(&self.inner);
        }

        pub fn save(self: *const Self, io: std.Io, file_path: []const u8) !void {
            const serialization = @import("serialization.zig");
            try serialization.saveModel(&self.inner, io, file_path, self.allocator);
        }

        pub fn load(self: *Self, io: std.Io, file_path: []const u8) !void {
            const serialization = @import("serialization.zig");
            try serialization.loadModel(&self.inner, io, file_path, self.allocator);
        }

        /// 前向计算，等价于 PyTorch 的 `module(*args)`：单个输入直接传入 (`m.forward(&g, x)`)，
        /// 多个输入以元组传入 (如 `gru.forward(&g, .{ inputs, h_0 })`)，返回类型与内部模型的 `forward` 相同
        pub fn forward(self: *Self, graph: *autodiff.Graph, args: anytype) ForwardResult(T, @TypeOf(args)) {
            return callForward(&self.inner, graph, args);
        }

        /// 可训练参数列表 (PyTorch `parameters()`)，返回的切片由调用方释放
        pub fn parameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
            return module.parameters(&self.inner, allocator);
        }

        /// 带字段路径名称的可训练参数列表 (PyTorch `named_parameters()`)，调用 `deinit` 释放
        pub fn namedParameters(self: *Self, allocator: std.mem.Allocator) !NamedParameterList {
            return module.namedParameters(&self.inner, allocator);
        }

        /// 可训练参数的元素总数
        pub fn numParameters(self: *const Self) usize {
            return module.numParameters(&self.inner);
        }

        /// 冻结 / 解冻全部参数 (PyTorch `requires_grad_()`)；冻结子模块时对 `&module.inner.<field>` 调用 `nn.setRequiresGrad`
        pub fn setRequiresGrad(self: *Self, requires_grad: bool) !void {
            try module.setRequiresGrad(&self.inner, self.allocator, requires_grad);
        }
    };
}

pub fn Sequential(comptime LayersTuple: type) type {
    return struct {
        layers: LayersTuple,

        const Self = @This();

        pub fn init(layers: LayersTuple) Self {
            return .{ .layers = layers };
        }

        pub fn setTraining(self: *Self, is_training: bool) void {
            setTrainingModel(&self.layers, is_training);
        }

        pub fn train(self: *Self) void {
            trainModel(&self.layers);
        }

        pub fn eval(self: *Self) void {
            evalModel(&self.layers);
        }

        pub fn forward(self: *const Self, graph: *autodiff.Graph, input: *Tensor) !*Tensor {
            var current = input;
            inline for (@typeInfo(LayersTuple).@"struct".fields) |field| {
                const layer = @field(self.layers, field.name);
                current = try layer.forward(graph, current);
            }
            return current;
        }
    };
}

pub fn sequential(layers: anytype) Sequential(@TypeOf(layers)) {
    return Sequential(@TypeOf(layers)).init(layers);
}
