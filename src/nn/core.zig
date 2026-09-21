const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

// ============================================================================
// 底层权重初始化与持久化张量内存辅助
// ============================================================================

pub fn normalRandom(random: std.Random) f32 {
    var u_1: f32 = random.float(f32);
    while (u_1 == 0.0) {
        u_1 = random.float(f32);
    }
    const u_2: f32 = random.float(f32);
    return @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
}

pub fn initializeWeights(random: std.Random, w: []f32, fan_in: usize) void {
    const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));
    for (w) |*val| {
        val.* = normalRandom(random) * std_dev;
    }
}

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
        const weight = try createPersistentTensor(allocator, in_features, out_features, true);
        errdefer freePersistentTensor(allocator, weight);
        const bias = try createPersistentTensor(allocator, 1, out_features, true);
        errdefer freePersistentTensor(allocator, bias);

        initializeWeights(random, weight.data, in_features);
        @memset(bias.data, 0.0);

        return Linear{
            .weight = weight,
            .bias = bias,
        };
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
        const weight = try createPersistentTensor(allocator, out_channels, in_channels * kernel_size * kernel_size, true);
        errdefer freePersistentTensor(allocator, weight);
        weight.shape = Shape.init(&.{ out_channels, in_channels, kernel_size, kernel_size });
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
            @memset(b.data, 0.0);
            bias = b;
        }

        const fan_in = in_channels * kernel_size * kernel_size;
        initializeWeights(random, weight.data, fan_in);

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

pub fn sequential(layers: anytype) Sequential(@TypeOf(layers)) {
    return Sequential(@TypeOf(layers)).init(layers);
}
