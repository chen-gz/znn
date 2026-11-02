const std = @import("std");
const nn = @import("../nn.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const checkpoint = @import("checkpoint.zig");

pub const SGDConfig = struct {
    lr: f32 = 0.01,
    momentum: f32 = 0.0,

    pub const default: SGDConfig = .{};
    pub fn defaultConfig() SGDConfig {
        return .{};
    }
};

/// 随机梯度下降优化器 (Stochastic Gradient Descent Optimizer, SGDOptimizer)
pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    velocities: ?[][]f32, // Only allocated if momentum > 0
    lr: f32,
    momentum: f32,
    step_count: u64 = 0,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !SGDOptimizer {
        return init(allocator, model, SGDConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: SGDConfig) !SGDOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const velocities: ?[][]f32 = if (config.momentum > 0.0) try checkpoint.allocZeroedStates(allocator, params) else null;

        return SGDOptimizer{
            .allocator = allocator,
            .params = params,
            .velocities = velocities,
            .lr = config.lr,
            .momentum = config.momentum,
            .step_count = 0,
        };
    }

    pub fn deinit(self: SGDOptimizer) void {
        if (self.velocities) |v_list| checkpoint.freeStates(self.allocator, v_list);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: SGDOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *SGDOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *SGDOptimizer, current_lr: f32) void {
        self.step_count += 1;
        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            if (self.velocities) |v_list| {
                const v = v_list[i];
                for (w, dw, v) |*weight, grad, *vel| {
                    vel.* = self.momentum * vel.* + current_lr * grad;
                    weight.* -= vel.*;
                }
            } else {
                for (w, dw) |*weight, grad| {
                    weight.* -= current_lr * grad;
                }
            }
            param.clearDeviceArray();
        }
    }

    pub fn step(self: *SGDOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: checkpoint.CheckpointHeader = .{ .step = self.step_count, .lr = self.lr };
        if (self.velocities) |v_list| {
            try checkpoint.writeCheckpoint(io, file_path, .sgd, header, self.params, &.{v_list});
        } else {
            try checkpoint.writeCheckpoint(io, file_path, .sgd, header, self.params, &.{});
        }
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try checkpoint.readCheckpoint(self.allocator, io, file_path, .sgd, self.params);

        // 提交阶段：以下操作均不会失败
        if (loaded.buffer_count == 1) {
            if (self.velocities) |*v_list| {
                checkpoint.replaceStates(self.allocator, v_list, loaded.states[0]);
            } else {
                self.velocities = loaded.states[0];
            }
        } else if (self.velocities) |v_list| {
            // 检查点不含动量缓冲：清零现有缓冲，使恢复后的状态与检查点一致
            for (v_list) |v| @memset(v, 0.0);
        }
        self.step_count = loaded.header.step;
        self.lr = loaded.header.lr;
    }
};
