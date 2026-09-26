const std = @import("std");

/// 激活函数类型 (Nonlinearity / Activation Type)
pub const Nonlinearity = union(enum) {
    /// 线性 / 恒等变换 (无激活函数，如最后的 Logits 输出层)
    linear,
    /// 线性整流单元 ReLU
    relu,
    /// 带泄露的 ReLU (携带斜率负半轴负斜率 alpha，默认为 0.2)
    leaky_relu: f32,
    /// 双曲正切激活 Tanh
    tanh,
    /// S 型激活 Sigmoid
    sigmoid,
    /// 高斯误差线性单元 GELU
    gelu,
    /// Sigmoid 线性单元 SiLU / Swish
    silu,
    /// 缩放指数线性单元 SELU
    selu,

    /// 默认激活函数为 ReLU
    pub const default: Nonlinearity = .relu;
    pub fn defaultNonlinearity() Nonlinearity {
        return .relu;
    }
};

/// 根据激活函数计算理论最优方差增益因子 (Gain)
/// 方差关系满足：Var(W) = gain^2 / fan_in (或 fan_in + fan_out)
pub fn calculateGain(nonlinearity: Nonlinearity) f32 {
    return switch (nonlinearity) {
        .linear, .sigmoid => 1.0,
        .relu => @sqrt(2.0), // ~1.41421356
        .leaky_relu => |alpha| @sqrt(2.0 / (1.0 + alpha * alpha)),
        .tanh => 5.0 / 3.0, // ~1.66666667
        .gelu, .silu => 1.0,
        .selu => 3.0 / 4.0, // 0.75
    };
}

/// 权重初始化策略枚举 (Weight Initialization Methods)
pub const InitMethod = union(enum) {
    /// 全零初始化 (通常仅适用于偏置，不适用于权重)
    zeros,
    /// 全一初始化 (适用于特定缩放因子如 RMSNorm/LayerNorm gamma)
    ones,
    /// 常数初始化
    constant: f32,
    /// 普通正态分布：mean, std
    normal: struct { mean: f32 = 0.0, std: f32 = 0.01 },
    /// 普通均匀分布：min, max
    uniform: struct { min: f32 = -0.01, max: f32 = 0.01 },
    /// Xavier / Glorot 正态分布：std = gain * sqrt(2 / (fan_in + fan_out))
    xavier_normal: struct { gain: f32 = 1.0 },
    /// Xavier / Glorot 均匀分布：limit = gain * sqrt(6 / (fan_in + fan_out))
    xavier_uniform: struct { gain: f32 = 1.0 },
    /// He / Kaiming 正态分布：std = gain * sqrt(1 / fan_in) (当 gain=sqrt(2) 时退化为 std=sqrt(2/fan_in))
    he_normal: struct { gain: f32 = 1.41421356 },
    /// He / Kaiming 均匀分布：limit = gain * sqrt(3 / fan_in) (当 gain=sqrt(2) 时退化为 limit=sqrt(6/fan_in))
    he_uniform: struct { gain: f32 = 1.41421356 },
    /// LeCun 正态分布：std = sqrt(1 / fan_in) (SELU / 线性首选)
    lecun_normal,
    /// LeCun 均匀分布：limit = sqrt(3 / fan_in)
    lecun_uniform,
};

/// 权重与偏置初始化配置选项 (Initialization Options)
pub const InitOptions = struct {
    /// 若非 null，则根据该激活函数自动推导 weight_init 的增益与方法
    nonlinearity: ?Nonlinearity = null,
    /// 显式指定的权重初始化策略。若为 null，则根据 nonlinearity 自动决策（若 nonlinearity 亦为 null 则默认采用 He Normal (gain=sqrt(2))）
    weight_init: ?InitMethod = null,
    /// 偏置初始化策略，默认为全零
    bias_init: InitMethod = .zeros,

    /// 针对线性/卷积层推荐的默认配置 (He Normal + Zeros Bias)
    pub const default: InitOptions = .{};

    /// 可调用的默认配置获取函数
    pub fn defaultOptions() InitOptions {
        return .{};
    }

    /// 解析出最终生效的 InitMethod
    pub fn resolveWeightInit(self: InitOptions) InitMethod {
        if (self.weight_init) |m| {
            return m;
        }
        if (self.nonlinearity) |nl| {
            const gain = calculateGain(nl);
            return switch (nl) {
                .tanh, .sigmoid => .{ .xavier_normal = .{ .gain = gain } },
                .selu => .lecun_normal,
                else => .{ .he_normal = .{ .gain = gain } },
            };
        }
        return .{ .he_normal = .{ .gain = @sqrt(2.0) } };
    }
};

/// Box-Muller 变换生成标准正态分布随机数 N(0, 1)
pub fn normalRandom(random: std.Random) f32 {
    var u_1: f32 = random.float(f32);
    while (u_1 == 0.0) {
        u_1 = random.float(f32);
    }
    const u_2: f32 = random.float(f32);
    return @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
}

/// 核心权重初始化函数
pub fn initWeights(
    random: std.Random,
    w: []f32,
    fan_in: usize,
    fan_out: usize,
    method: InitMethod,
) void {
    const f_in: f32 = @floatFromInt(fan_in);
    const f_out: f32 = @floatFromInt(fan_out);

    switch (method) {
        .zeros => {
            @memset(w, 0.0);
        },
        .ones => {
            @memset(w, 1.0);
        },
        .constant => |c| {
            @memset(w, c);
        },
        .normal => |p| {
            for (w) |*val| {
                val.* = p.mean + normalRandom(random) * p.std;
            }
        },
        .uniform => |p| {
            const range = p.max - p.min;
            for (w) |*val| {
                val.* = p.min + random.float(f32) * range;
            }
        },
        .xavier_normal => |p| {
            const denom = f_in + f_out;
            const std_dev = if (denom > 0.0) p.gain * @sqrt(2.0 / denom) else 0.0;
            for (w) |*val| {
                val.* = normalRandom(random) * std_dev;
            }
        },
        .xavier_uniform => |p| {
            const denom = f_in + f_out;
            const limit = if (denom > 0.0) p.gain * @sqrt(6.0 / denom) else 0.0;
            for (w) |*val| {
                val.* = (random.float(f32) * 2.0 - 1.0) * limit;
            }
        },
        .he_normal => |p| {
            const std_dev = if (f_in > 0.0) p.gain * @sqrt(1.0 / f_in) else 0.0;
            for (w) |*val| {
                val.* = normalRandom(random) * std_dev;
            }
        },
        .he_uniform => |p| {
            const limit = if (f_in > 0.0) p.gain * @sqrt(3.0 / f_in) else 0.0;
            for (w) |*val| {
                val.* = (random.float(f32) * 2.0 - 1.0) * limit;
            }
        },
        .lecun_normal => {
            const std_dev = if (f_in > 0.0) @sqrt(1.0 / f_in) else 0.0;
            for (w) |*val| {
                val.* = normalRandom(random) * std_dev;
            }
        },
        .lecun_uniform => {
            const limit = if (f_in > 0.0) @sqrt(3.0 / f_in) else 0.0;
            for (w) |*val| {
                val.* = (random.float(f32) * 2.0 - 1.0) * limit;
            }
        },
    }
}

/// 兼容旧版本的初始化函数 (默认采用 He 正态分布，gain=sqrt(2))
pub fn initializeWeights(random: std.Random, w: []f32, fan_in: usize) void {
    initWeights(random, w, fan_in, fan_in, .{ .he_normal = .{} });
}
