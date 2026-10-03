const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

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
    if (t.requires_grad) {
        allocator.free(t.grad);
    }
    allocator.destroy(t);
}

// ============================================================================
// 线性层 (Linear) 与二维卷积模块 (2D Convolution, Conv2D / 2D Transposed Convolution, ConvTranspose2D)
// ============================================================================

pub const Linear = struct {
    weight: *Tensor,
    bias: *Tensor,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "Linear",

    /// 构造线性层并分配参数内存。
    /// random 非 null 时按库默认策略调用 resetParameters 填充参数；
    /// 为 null 时参数保持全零，交由 nn.initModel / Graph.initWeights 在模型组装后统一初始化。
    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, random: ?std.Random) !Linear {
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        var l = Linear{
            .weight = weight,
            .bias = bias,
        };
        if (random) |rnd| l.resetParameters(rnd, InitOptions.default);
        return l;
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充权重与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *Linear, random: std.Random, options: InitOptions) void {
        const in_features = self.weight.shape.dims[0];
        const out_features = self.weight.shape.dims[1];
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, in_features, out_features, w_init);
        initWeights(random, self.bias.data, in_features, out_features, options.bias_init);
    }

    /// 为层内权重与偏置张量统一设置人类可读的名称 (如传入 "fc1"，自动设置 "fc1.weight" 与 "fc1.bias")
    pub fn setName(self: *Linear, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
        self.bias.setNameFormatted("{s}.bias", .{self.name.?});
    }

    /// 使用格式化模板为层设置人类可读的名称 (如 "{s}.fc1", parent_name)
    pub fn setNameFormatted(self: *Linear, comptime fmt: []const u8, args: anytype) void {
        if (std.fmt.bufPrint(&self.name_buf, fmt, args)) |s| {
            self.name = s;
        } else |_| {
            self.name = "linear";
        }
        if (self.name) |n| {
            self.weight.setNameFormatted("{s}.weight", .{n});
            self.bias.setNameFormatted("{s}.bias", .{n});
        }
    }

    /// 获取层的人类可读名称
    pub fn getName(self: *const Linear) ?[]const u8 {
        return self.name;
    }

    pub fn deinit(self: Linear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    pub fn zeroGrad(self: Linear) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = x W^T + b";

    /// 向计算图注册该模块的数学公式
    pub fn registerFormula(self: *const Linear, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        const z = try graph.matmul(x, self.weight);
        return try graph.addBias(z, self.bias);
    }
};

pub const Conv2D = struct {
    weight: *Tensor,
    bias: *Tensor,
    stride: usize = 1,
    padding: usize = 0,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "Conv2D",

    /// 构造 stride = 1、padding = 0 的卷积层；random 语义同 Linear.init
    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, random: ?std.Random) !Conv2D {
        return initWithConfig(allocator, in_channels, out_channels, kernel_size, 1, 0, random);
    }

    /// 构造支持自定义 stride 与 padding 的卷积层并分配参数内存。
    /// random 非 null 时按库默认策略调用 resetParameters 填充参数；为 null 时参数保持全零，留待模型组装后统一初始化。
    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        random: ?std.Random,
    ) !Conv2D {
        if (stride == 0) return error.InvalidStride;
        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        weight.shape = Shape.init(&.{ out_channels, in_channels, kernel_size, kernel_size });
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        const bias = try createPersistentTensor(allocator, 1, out_channels, true);
        errdefer freePersistentTensor(allocator, bias);
        bias.shape = Shape.init(&.{out_channels});
        bias.strides = tensor.computeContiguousStrides(bias.shape);

        var c = Conv2D{
            .weight = weight,
            .bias = bias,
            .stride = stride,
            .padding = padding,
        };
        if (random) |rnd| c.resetParameters(rnd, InitOptions.default);
        return c;
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

    /// 为层内权重与偏置张量统一设置人类可读的名称 (如传入 "conv1"，自动设置 "conv1.weight" 与 "conv1.bias")
    pub fn setName(self: *Conv2D, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
        self.bias.setNameFormatted("{s}.bias", .{self.name.?});
    }

    /// 使用格式化模板为层设置人类可读的名称 (如 "{s}.conv1", parent_name)
    pub fn setNameFormatted(self: *Conv2D, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("conv2d");
        }
    }

    /// 获取层的人类可读名称
    pub fn getName(self: *const Conv2D) ?[]const u8 {
        return self.name;
    }

    pub fn deinit(self: Conv2D, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    pub fn zeroGrad(self: Conv2D) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = x \\ast W + b";

    pub fn registerFormula(self: *const Conv2D, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    pub fn forward(self: Conv2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        return try graph.conv2dWithConfig(x, self.weight, self.bias, self.stride, self.padding);
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
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "ConvTranspose2D",

    /// 构造反卷积层并分配参数内存；random 语义同 Linear.init
    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        random: ?std.Random,
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

        var c = ConvTranspose2D{
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .weight = weight,
            .bias = bias,
        };
        if (random) |rnd| c.resetParameters(rnd, InitOptions.default);
        return c;
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

    /// 为层内权重与偏置张量统一设置人类可读的名称 (如传入 "deconv1"，自动设置 "deconv1.weight" 与 "deconv1.bias")
    pub fn setName(self: *ConvTranspose2D, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
        if (self.bias) |b| b.setNameFormatted("{s}.bias", .{self.name.?});
    }

    /// 使用格式化模板为层设置人类可读的名称 (如 "{s}.deconv1", parent_name)
    pub fn setNameFormatted(self: *ConvTranspose2D, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("deconv2d");
        }
    }

    /// 获取层的人类可读名称
    pub fn getName(self: *const ConvTranspose2D) ?[]const u8 {
        return self.name;
    }

    pub fn deinit(self: ConvTranspose2D, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        if (self.bias) |b| freePersistentTensor(allocator, b);
    }

    pub fn zeroGrad(self: ConvTranspose2D) void {
        self.weight.zeroGrad();
        if (self.bias) |b| b.zeroGrad();
    }

    pub fn forward(self: ConvTranspose2D, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        return try graph.convTranspose2D(x, self.weight, self.bias, self.stride, self.padding);
    }
};

// ============================================================================
// 泛型模型元编程与参数搜集
// ============================================================================

pub fn deinitModel(model: anytype, allocator: std.mem.Allocator) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            freePersistentTensor(allocator, @field(model, field.name));
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |t| {
                freePersistentTensor(allocator, t);
            }
        } else if (FieldType == []f32) {
            allocator.free(@field(model, field.name));
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| {
                    freePersistentTensor(allocator, t);
                }
                allocator.free(@field(model, field.name));
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| {
                    deinitModel(item, allocator);
                }
                allocator.free(@field(model, field.name));
            }
        } else if (field_info == .@"struct") {
            deinitModel(&@field(model, field.name), allocator);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| {
                    freePersistentTensor(allocator, t);
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
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
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |t| {
                t.zeroGrad();
            }
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| {
                    t.zeroGrad();
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| {
                    zeroGradModel(item);
                }
            }
        } else if (field_info == .@"struct") {
            zeroGradModel(&@field(model, field.name));
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| {
                    t.zeroGrad();
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
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
    if (@hasField(T, "lora_a") and @hasField(T, "lora_b") and @hasField(T, "weight")) {
        model.weight.requires_grad = false;
    }
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        if (@sizeOf(FieldType) == 0) continue;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            const tensor_ptr = @field(model, field.name);
            if (tensor_ptr.requires_grad) {
                try list.append(allocator, tensor_ptr);
            }
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |tensor_ptr| {
                if (tensor_ptr.requires_grad) {
                    try list.append(allocator, tensor_ptr);
                }
            }
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |tensor_ptr| {
                    if (tensor_ptr.requires_grad) {
                        try list.append(allocator, tensor_ptr);
                    }
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| {
                    try collectParametersInternal(item, list, allocator);
                }
            }
        } else if (field_info == .@"struct") {
            try collectParametersInternal(&@field(model, field.name), list, allocator);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |tensor_ptr| {
                    if (tensor_ptr.requires_grad) {
                        try list.append(allocator, tensor_ptr);
                    }
                }
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    try collectParametersInternal(item, list, allocator);
                }
            }
        }
    }
}

pub fn setTrainingModel(model: anytype, is_training: bool) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    if (info != .@"struct") return;
    if (@hasField(T, "training") and @TypeOf(@field(model, "training")) == bool) {
        @field(model, "training") = is_training;
    }
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        if (@sizeOf(FieldType) == 0) continue;
        const field_info = @typeInfo(FieldType);
        if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| {
                    setTrainingModel(item, is_training);
                }
            }
        } else if (field_info == .@"struct") {
            setTrainingModel(&@field(model, field.name), is_training);
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name)) |*item| {
                    setTrainingModel(item, is_training);
                }
            }
        }
    }
}

pub fn trainModel(model: anytype) void {
    setTrainingModel(model, true);
}

pub fn evalModel(model: anytype) void {
    setTrainingModel(model, false);
}

/// 模型参数初始化入口 (编译期反射分派)：
/// 1. 若模块类型定义了 `pub fn customInit(self: *Self, random: std.Random) void` (由库外用户模块定义)，
///    则调用该函数，并将该模块内所有可训练参数标记为 `is_custom_initialized = true`
///    (`Graph.initWeights` 不再覆盖，模型图导出为 `CUSTOM_INIT`)；
/// 2. 否则使用库内置初始化：模块定义了 `resetParameters(self, random, options)` 时以默认选项调用，
///    `Sequential` 调用 `autoInit`；其余结构体递归处理各子模块字段。
pub fn initModel(model: anytype, random: std.Random) void {
    const T = @TypeOf(model.*);
    if (@typeInfo(T) != .@"struct") return;

    if (@hasDecl(T, "customInit")) {
        model.customInit(random);
        markCustomInitializedModel(model);
        return;
    }
    if (@hasDecl(T, "resetParameters")) {
        model.resetParameters(random, .{});
        return;
    }
    if (@hasDecl(T, "autoInit")) {
        model.autoInit(random);
        return;
    }

    inline for (@typeInfo(T).@"struct".fields) |field| {
        const FieldType = field.type;
        if (field.is_comptime or @sizeOf(FieldType) == 0) continue;
        const field_info = @typeInfo(FieldType);
        if (field_info == .@"struct") {
            initModel(&@field(model, field.name), random);
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            if (@typeInfo(field_info.pointer.child) == .@"struct") {
                for (@field(model, field.name)) |*item| initModel(item, random);
            }
        } else if (field_info == .@"array") {
            if (@typeInfo(field_info.@"array".child) == .@"struct") {
                for (&@field(model, field.name)) |*item| initModel(item, random);
            }
        }
    }
}

/// 将模型内所有可训练参数标记为自定义初始化 (由 `initModel` 在调用外部模块的 `customInit` 后执行)
fn markCustomInitializedModel(model: anytype) void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    if (info != .@"struct") return;
    inline for (info.@"struct".fields) |field| {
        const FieldType = field.type;
        if (field.is_comptime or @sizeOf(FieldType) == 0) continue;
        const field_info = @typeInfo(FieldType);
        if (FieldType == *Tensor) {
            markTensorCustomInitialized(@field(model, field.name));
        } else if (field_info == .optional and field_info.optional.child == *Tensor) {
            if (@field(model, field.name)) |t| markTensorCustomInitialized(t);
        } else if (field_info == .pointer and field_info.pointer.size == .slice and !field_info.pointer.is_const) {
            const ElemT = field_info.pointer.child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| markTensorCustomInitialized(t);
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (@field(model, field.name)) |*item| markCustomInitializedModel(item);
            }
        } else if (field_info == .@"struct") {
            markCustomInitializedModel(&@field(model, field.name));
        } else if (field_info == .@"array") {
            const ElemT = field_info.@"array".child;
            if (ElemT == *Tensor) {
                for (@field(model, field.name)) |t| markTensorCustomInitialized(t);
            } else if (@typeInfo(ElemT) == .@"struct") {
                for (&@field(model, field.name)) |*item| markCustomInitializedModel(item);
            }
        }
    }
}

fn markTensorCustomInitialized(t: *Tensor) void {
    if (t.requires_grad) t.is_custom_initialized = true;
}

pub fn Module(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        inner: T,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, inner: T) Self {
            return Self{
                .allocator = allocator,
                .inner = inner,
            };
        }

        /// 初始化内部模型参数：内部模型定义了 customInit 时调用之，否则使用库内置初始化 (见 `initModel`)
        pub fn initParameters(self: *Self, random: std.Random) void {
            initModel(&self.inner, random);
        }

        pub fn deinit(self: *Self) void {
            deinitModel(&self.inner, self.allocator);
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

        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return try self.inner.forward(graph, x);
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

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            inline for (@typeInfo(LayersTuple).@"struct".fields) |field| {
                const layer = @field(self.layers, field.name);
                const LayerT = @TypeOf(layer);
                if (@hasDecl(LayerT, "deinit")) {
                    layer.deinit(allocator);
                }
            }
        }

        pub fn zeroGrad(self: Self) void {
            inline for (@typeInfo(LayersTuple).@"struct".fields) |field| {
                const layer = @field(self.layers, field.name);
                const LayerT = @TypeOf(layer);
                if (@hasDecl(LayerT, "zeroGrad")) {
                    layer.zeroGrad();
                }
            }
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

        /// 按层初始化：外部层定义了 customInit 时调用之 (经 `initModel`)；
        /// Linear / Conv2D / ConvTranspose2D 依据其后的激活函数选择内置初始化；其余层使用库内置初始化
        pub fn autoInit(self: *Self, random: std.Random) void {
            const fields = @typeInfo(LayersTuple).@"struct".fields;
            inline for (fields, 0..) |field, i| {
                const LayerT = field.type;
                if (field.is_comptime or @sizeOf(LayerT) == 0) continue;
                if (@hasDecl(LayerT, "customInit")) {
                    initModel(&@field(self.layers, field.name), random);
                } else if (LayerT == Linear or LayerT == Conv2D or LayerT == ConvTranspose2D) {
                    const act = comptime detectNextActivation(LayersTuple, i);
                    @field(self.layers, field.name).resetParameters(random, .{ .nonlinearity = act });
                } else {
                    initModel(&@field(self.layers, field.name), random);
                }
            }
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

/// 编译期静态检测 Sequential 元组中第 i 个层之后的首个有效激活函数
pub fn detectNextActivation(comptime LayersTuple: type, comptime current_idx: usize) Nonlinearity {
    const fields = @typeInfo(LayersTuple).@"struct".fields;
    const activations = @import("activations.zig");

    inline for (current_idx + 1..fields.len) |next_idx| {
        const NextT = fields[next_idx].type;

        if (NextT == activations.ReLU) return .relu;
        if (NextT == activations.Tanh) return .tanh;
        if (NextT == activations.Sigmoid) return .sigmoid;
        if (NextT == activations.GELU) return .gelu;
        if (NextT == activations.SiLU) return .silu;
        if (NextT == activations.LeakyReLU) return .{ .leaky_relu = 0.2 };

        // 如果又遇到了另一个参数层 (例如 Linear 或 Conv2D)，说明当前层后续没有激活函数 (如多层特征变换或网络出口 Logits)
        if (NextT == Linear or NextT == Conv2D or NextT == ConvTranspose2D) {
            return .linear;
        }
    }
    // 直到末尾都未遇到激活函数，说明是网络输出层 (Logits)
    return .linear;
}

pub fn sequential(layers: anytype) Sequential(@TypeOf(layers)) {
    return Sequential(@TypeOf(layers)).init(layers);
}

/// 构造并自动根据各层后续激活函数自适应初始化权重的 Sequential 模型
pub fn autoSequential(layers: anytype, random: std.Random) Sequential(@TypeOf(layers)) {
    var seq = Sequential(@TypeOf(layers)).init(layers);
    seq.autoInit(random);
    return seq;
}
