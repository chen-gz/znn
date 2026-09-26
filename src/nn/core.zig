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
pub const initializeWeights = init_mod.initializeWeights;

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
// 线性层 (Linear) 与卷积模块 (Conv2D, ConvTranspose2D)
// ============================================================================

pub const Linear = struct {
    weight: *Tensor,
    bias: *Tensor,

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, random: std.Random) !Linear {
        return initWithOptions(allocator, in_features, out_features, random, InitOptions.default);
    }

    pub fn initUninitialized(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Linear {
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        return Linear{
            .weight = weight,
            .bias = bias,
        };
    }

    pub fn reinit(self: *Linear, random: std.Random, options: InitOptions) void {
        const in_features = self.weight.shape.dims[0];
        const out_features = self.weight.shape.dims[1];
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, in_features, out_features, w_init);
        initWeights(random, self.bias.data, in_features, out_features, options.bias_init);
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        random: std.Random,
        options: InitOptions,
    ) !Linear {
        var linear = try initUninitialized(allocator, in_features, out_features);
        linear.reinit(random, options);
        return linear;
    }

    pub fn deinit(self: Linear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        freePersistentTensor(allocator, self.bias);
    }

    pub fn zeroGrad(self: Linear) void {
        self.weight.zeroGrad();
        self.bias.zeroGrad();
    }

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
        return initWithOptions(allocator, in_channels, out_channels, kernel_size, random, InitOptions.default);
    }

    pub fn initUninitialized(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize) !Conv2D {
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
        };
    }

    pub fn reinit(self: *Conv2D, random: std.Random, options: InitOptions) void {
        const out_channels = self.weight.shape.dims[0];
        const in_channels = self.weight.shape.dims[1];
        const kernel_size = self.weight.shape.dims[2];
        const fan_in = in_channels * kernel_size * kernel_size;
        const fan_out = out_channels * kernel_size * kernel_size;
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, fan_in, fan_out, w_init);
        initWeights(random, self.bias.data, fan_in, fan_out, options.bias_init);
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        random: std.Random,
        options: InitOptions,
    ) !Conv2D {
        var conv = try initUninitialized(allocator, in_channels, out_channels, kernel_size);
        conv.reinit(random, options);
        return conv;
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

pub const ConvTranspose2D = struct {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight: *Tensor,
    bias: ?*Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        random: std.Random,
    ) !ConvTranspose2D {
        return initWithOptions(allocator, in_channels, out_channels, kernel_size, stride, padding, use_bias, random, InitOptions.default);
    }

    pub fn initUninitialized(
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

    pub fn reinit(self: *ConvTranspose2D, random: std.Random, options: InitOptions) void {
        const fan_in = self.in_channels * self.kernel_size * self.kernel_size;
        const fan_out = self.out_channels * self.kernel_size * self.kernel_size;
        const w_init = options.resolveWeightInit();
        initWeights(random, self.weight.data, fan_in, fan_out, w_init);
        if (self.bias) |b| {
            initWeights(random, b.data, fan_in, fan_out, options.bias_init);
        }
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        random: std.Random,
        options: InitOptions,
    ) !ConvTranspose2D {
        var conv = try initUninitialized(allocator, in_channels, out_channels, kernel_size, stride, padding, use_bias);
        conv.reinit(random, options);
        return conv;
    }

    pub fn deinit(self: ConvTranspose2D, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        if (self.bias) |b| freePersistentTensor(allocator, b);
    }

    pub fn zeroGrad(self: ConvTranspose2D) void {
        self.weight.zeroGrad();
        if (self.bias) |b| b.zeroGrad();
    }

    pub fn forward(self: ConvTranspose2D, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        if (graph) |g| {
            return try g.convTranspose2D(x, self.weight, self.bias, self.stride, self.padding);
        }
        return try x.convTranspose2d(self.weight, self.bias, self.stride, self.padding, allocator, null);
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

        pub fn deinit(self: *Self) void {
            deinitModel(&self.inner, self.allocator);
        }

        pub fn zeroGrad(self: *Self) void {
            zeroGradModel(&self.inner);
        }

        pub fn save(self: *const Self, io: std.Io, file_path: []const u8) !void {
            const serialization = @import("serialization.zig");
            try serialization.saveModel(&self.inner, io, file_path, self.allocator);
        }

        pub fn load(self: *Self, io: std.Io, file_path: []const u8) !void {
            const serialization = @import("serialization.zig");
            try serialization.loadModel(&self.inner, io, file_path, self.allocator);
        }

        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            return try self.inner.forward(allocator, graph, x);
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

        pub fn autoInit(self: *Self, random: std.Random) void {
            const fields = @typeInfo(LayersTuple).@"struct".fields;
            inline for (fields, 0..) |field, i| {
                const LayerT = field.type;
                if (LayerT == Linear or LayerT == Conv2D or LayerT == ConvTranspose2D) {
                    const act = comptime detectNextActivation(LayersTuple, i);
                    @field(self.layers, field.name).reinit(random, .{ .nonlinearity = act });
                }
            }
        }

        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, input: *Tensor) !*Tensor {
            var current = input;

            inline for (@typeInfo(LayersTuple).@"struct".fields) |field| {
                const layer = @field(self.layers, field.name);
                const next = try layer.forward(allocator, graph, current);
                if (graph == null and current != input) {
                    tensor.free(allocator, current);
                }
                current = next;
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
