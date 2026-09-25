const std = @import("std");

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
    /// Xavier / Glorot 正态分布：std = sqrt(2 / (fan_in + fan_out))
    xavier_normal,
    /// Xavier / Glorot 均匀分布：limit = sqrt(6 / (fan_in + fan_out))
    xavier_uniform,
    /// He / Kaiming 正态分布：std = sqrt(2 / fan_in) (ReLU 首选)
    he_normal,
    /// He / Kaiming 均匀分布：limit = sqrt(6 / fan_in)
    he_uniform,
    /// LeCun 正态分布：std = sqrt(1 / fan_in) (SELU / 线性首选)
    lecun_normal,
    /// LeCun 均匀分布：limit = sqrt(3 / fan_in)
    lecun_uniform,
};

/// 权重与偏置初始化配置选项 (Initialization Options)
pub const InitOptions = struct {
    weight_init: InitMethod = .he_normal,
    bias_init: InitMethod = .zeros,

    /// 针对线性/卷积层推荐的默认配置 (He Normal + Zeros Bias)
    pub const default: InitOptions = .{};

    /// 可调用的默认配置获取函数
    pub fn defaultOptions() InitOptions {
        return .{};
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
        .xavier_normal => {
            const denom = f_in + f_out;
            const std_dev = if (denom > 0.0) @sqrt(2.0 / denom) else 0.0;
            for (w) |*val| {
                val.* = normalRandom(random) * std_dev;
            }
        },
        .xavier_uniform => {
            const denom = f_in + f_out;
            const limit = if (denom > 0.0) @sqrt(6.0 / denom) else 0.0;
            for (w) |*val| {
                val.* = (random.float(f32) * 2.0 - 1.0) * limit;
            }
        },
        .he_normal => {
            const std_dev = if (f_in > 0.0) @sqrt(2.0 / f_in) else 0.0;
            for (w) |*val| {
                val.* = normalRandom(random) * std_dev;
            }
        },
        .he_uniform => {
            const limit = if (f_in > 0.0) @sqrt(6.0 / f_in) else 0.0;
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

/// 兼容旧版本的初始化函数 (默认采用 He 正态分布)
pub fn initializeWeights(random: std.Random, w: []f32, fan_in: usize) void {
    initWeights(random, w, fan_in, fan_in, .he_normal);
}
