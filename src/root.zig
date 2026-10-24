pub const VERSION = "0.3.0";
pub const version = @import("std").SemanticVersion{ .major = 0, .minor = 3, .patch = 0 };

pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const dataset = @import("dataset.zig");
pub const autodiff = @import("autodiff.zig");
pub const optim = @import("optim.zig");
pub const regression = @import("regression.zig");
pub const manifold = @import("manifold.zig");
pub const cv = @import("cross_validation.zig");
pub const engine = @import("engine.zig");
pub const bench = @import("bench.zig");
pub const models = @import("models.zig");
pub const backend = @import("backend.zig");
pub const Executor = backend.Executor;
pub const mlx = backend.mlx;
pub const offload = backend.offload;
pub const Device = backend.Device;

pub const gemma4 = models.gemma4;
pub const gemma4_q4 = models.gemma4_q4;
pub const gemma4_e4b = models.gemma4_e4b;
pub const Gemma4E4BModel = models.Gemma4E4BModel;
pub const SafetensorsArchive = nn.SafetensorsArchive;
pub const TensorMeta = nn.TensorMeta;

pub const TSNE = manifold.TSNE;
pub const TSNEOptions = manifold.TSNEOptions;
pub const tsne = manifold.tsne;
pub const tsneDefault = manifold.tsneDefault;

pub const Nonlinearity = nn.Nonlinearity;
pub const calculateGain = nn.calculateGain;
pub const InitMethod = nn.InitMethod;
pub const InitOptions = nn.InitOptions;
pub const initWeights = nn.initWeights;
pub const initModel = nn.initModel;
pub const initModelWithSample = nn.initModelWithSample;
pub const setTrainingModel = nn.setTrainingModel;
pub const trainModel = nn.trainModel;
pub const evalModel = nn.evalModel;
pub const ScaledDotProductAttention = nn.ScaledDotProductAttention;
pub const DefaultGPT = models.DefaultGPT;
pub const Recurrent = nn.Recurrent;
pub const RecurrentResult = nn.RecurrentResult;
pub const RNNResult = nn.RNNResult;
pub const LSTMResult = nn.LSTMResult;
pub const StackedLSTMResult = nn.StackedLSTMResult;
pub const GRUResult = nn.GRUResult;

pub const StaticTensor = tensor.StaticTensor;
pub const GenericTensor = tensor.GenericTensor;
pub const FloatTensor = tensor.FloatTensor;
pub const DoubleTensor = tensor.DoubleTensor;
pub const IntTensor = tensor.IntTensor;
pub const LongTensor = tensor.LongTensor;
pub const BoolTensor = tensor.BoolTensor;
pub const BFloat16Tensor = tensor.BFloat16Tensor;
pub const bf16 = tensor.bf16;
pub const DType = tensor.DType;
pub const SliceRange = tensor.SliceRange;
pub const ConvOptions = tensor.ConvOptions;
pub const PoolOptions = tensor.PoolOptions;
pub const RopeOptions = tensor.RopeOptions;
pub const QRResult = tensor.QRResult;
pub const SymEigResult = tensor.SymEigResult;
pub const SVDResult = tensor.SVDResult;
pub const qr = tensor.qr;
pub const symeig = tensor.symeig;
pub const svd = tensor.svd;

pub const CrossValidationOptions = cv.CrossValidationOptions;

pub const SGDConfig = optim.SGDConfig;
pub const AdamConfig = optim.AdamConfig;
pub const AdamWConfig = optim.AdamWConfig;
pub const CosineScheduler = optim.CosineScheduler;
pub const StepLRScheduler = optim.StepLRScheduler;
pub const LinearWarmupScheduler = optim.LinearWarmupScheduler;
pub const ExponentialLRScheduler = optim.ExponentialLRScheduler;
pub const LRScheduler = optim.LRScheduler;
pub const GradClipConfig = optim.GradClipConfig;

/// 测量一次函数调用的耗时 (单调时钟，纳秒)
pub fn measureTime(comptime func: anytype, args: anytype) !struct {
    result: @TypeOf(@call(.auto, func, args)),
    elapsed_ns: u64,
} {
    const start_ns = bench.getTimeNs();
    const result = @call(.auto, func, args);
    return .{
        .result = result,
        .elapsed_ns = bench.getTimeNs() - start_ns,
    };
}

/// 代码块耗时打印：`start` 记录起点，`end` 打印 `[PROFILE] <label> took <ms>`
pub const ProfileBlock = struct {
    label: []const u8,
    start_ns: u64,

    pub fn start(label: []const u8) ProfileBlock {
        return .{
            .label = label,
            .start_ns = bench.getTimeNs(),
        };
    }

    pub fn end(self: ProfileBlock) void {
        const std = @import("std");
        const builtin = @import("builtin");
        const elapsed_ms = @as(f64, @floatFromInt(bench.getTimeNs() - self.start_ns)) / 1_000_000.0;
        if (!builtin.is_test) {
            std.debug.print("[PROFILE] {s} took {d:.3}ms\n", .{ self.label, elapsed_ms });
        }
    }
};

/// 作用域计时器：`end` 把耗时 (纳秒) 写入 `elapsed_ns_ptr`
pub const ScopeTimer = struct {
    start_ns: u64,
    elapsed_ns_ptr: *u64,

    pub fn start(elapsed_ns_ptr: *u64) ScopeTimer {
        return .{
            .start_ns = bench.getTimeNs(),
            .elapsed_ns_ptr = elapsed_ns_ptr,
        };
    }

    pub fn end(self: ScopeTimer) void {
        self.elapsed_ns_ptr.* = bench.getTimeNs() - self.start_ns;
    }
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}
