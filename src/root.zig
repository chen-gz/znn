pub const VERSION = "0.2.7";
pub const version = @import("std").SemanticVersion{ .major = 0, .minor = 2, .patch = 7 };

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
pub const DefaultGPT = nn.DefaultGPT;
pub const DefaultGemma4 = nn.DefaultGemma4;
pub const TinyGemma4 = nn.TinyGemma4;
pub const TinyQ4Gemma4 = nn.TinyQ4Gemma4;
pub const Q4Linear = nn.Q4Linear;
pub const Q4Block = nn.Q4Block;
pub const Gemma4Config = nn.Gemma4Config;
pub const RNNResult = nn.RNNResult;
pub const LSTMResult = nn.LSTMResult;
pub const StackedLSTMResult = nn.StackedLSTMResult;
pub const GRUResult = nn.GRUResult;

pub const StaticTensor = tensor.StaticTensor;
pub const GenericTensor = tensor.GenericTensor;
pub const TensorOf = tensor.TensorOf;
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


pub fn measureTime(comptime func: anytype, args: anytype) !struct {
    result: @TypeOf(@call(.auto, func, args)),
    elapsed_ns: u64,
} {
    const std = @import("std");
    var start_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);

    const result = @call(.auto, func, args);

    var end_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);

    const start_ns = @as(u64, @intCast(start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(start_ts.nsec));
    const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
    return .{
        .result = result,
        .elapsed_ns = end_ns - start_ns,
    };
}


pub const ProfileBlock = struct {
    label: []const u8,
    start_ts: @import("std").posix.system.timespec,

    pub fn start(label: []const u8) ProfileBlock {
        const std = @import("std");
        var start_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);
        return .{
            .label = label,
            .start_ts = start_ts,
        };
    }

    pub fn end(self: ProfileBlock) void {
        const std = @import("std");
        const builtin = @import("builtin");
        var end_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);
        const start_ns = @as(u64, @intCast(self.start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(self.start_ts.nsec));
        const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
        const elapsed_ms = @as(f64, @floatFromInt(end_ns - start_ns)) / 1_000_000.0;
        if (!builtin.is_test) {
            std.debug.print("[PROFILE] {s} took {d:.3}ms\n", .{ self.label, elapsed_ms });
        }
    }
};

pub const ScopeTimer = struct {
    start_ts: @import("std").posix.system.timespec,
    elapsed_ns_ptr: *u64,

    pub fn start(elapsed_ns_ptr: *u64) ScopeTimer {
        const std = @import("std");
        var start_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);
        return .{
            .start_ts = start_ts,
            .elapsed_ns_ptr = elapsed_ns_ptr,
        };
    }

    pub fn end(self: ScopeTimer) void {
        const std = @import("std");
        var end_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);
        const start_ns = @as(u64, @intCast(self.start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(self.start_ts.nsec));
        const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
        self.elapsed_ns_ptr.* = end_ns - start_ns;
    }
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}
