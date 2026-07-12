const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    velocities: ?[][]f32, // Only allocated if momentum > 0
    lr: f32,
    momentum: f32,

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: struct { lr: f32, momentum: f32 = 0.0 }) !SGDOptimizer {
        const params = try nn.collectParameters(model, allocator);
        errdefer allocator.free(params);

        var velocities: ?[][]f32 = null;
        if (config.momentum > 0.0) {
            const v_list = try allocator.alloc([]f32, params.len);
            errdefer allocator.free(v_list);
            var initialized: usize = 0;
            errdefer {
                for (0..initialized) |j| allocator.free(v_list[j]);
            }
            for (params) |param| {
                v_list[initialized] = try allocator.alloc(f32, param.data.len);
                @memset(v_list[initialized], 0.0);
                initialized += 1;
            }
            velocities = v_list;
        }

        return SGDOptimizer{
            .allocator = allocator,
            .params = params,
            .velocities = velocities,
            .lr = config.lr,
            .momentum = config.momentum,
        };
    }

    pub fn deinit(self: SGDOptimizer) void {
        if (self.velocities) |v_list| {
            for (v_list) |v| self.allocator.free(v);
            self.allocator.free(v_list);
        }
        self.allocator.free(self.params);
    }

    pub fn step(self: SGDOptimizer) void {
        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            if (self.velocities) |v_list| {
                const v = v_list[i];
                for (w, dw, v) |*weight, grad, *vel| {
                    vel.* = self.momentum * vel.* + self.lr * grad;
                    weight.* -= vel.*;
                }
            } else {
                for (w, dw) |*weight, grad| {
                    weight.* -= self.lr * grad;
                }
            }
        }
    }
};

pub const AdamOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: f32, // Timestep

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: struct {
        lr: f32 = 0.001,
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        eps: f32 = 1e-8,
    }) !AdamOptimizer {
        const params = try nn.collectParameters(model, allocator);
        errdefer allocator.free(params);

        const m = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(m);
        const v = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(v);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(m[i]);
                allocator.free(v[i]);
            }
        }

        for (params) |param| {
            m[initialized] = try allocator.alloc(f32, param.data.len);
            v[initialized] = try allocator.alloc(f32, param.data.len);
            @memset(m[initialized], 0.0);
            @memset(v[initialized], 0.0);
            initialized += 1;
        }

        return AdamOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .lr = config.lr,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .t = 0.0,
        };
    }

    pub fn deinit(self: AdamOptimizer) void {
        for (0..self.params.len) |i| {
            self.allocator.free(self.m[i]);
            self.allocator.free(self.v[i]);
        }
        self.allocator.free(self.m);
        self.allocator.free(self.v);
        self.allocator.free(self.params);
    }

    pub fn step(self: *AdamOptimizer) void {
        self.t += 1.0;
        const correction1 = 1.0 - std.math.pow(f32, self.beta1, self.t);
        const correction2 = 1.0 - std.math.pow(f32, self.beta2, self.t);
        const lr_t = self.lr * @sqrt(correction2) / correction1;

        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            const m_t = self.m[i];
            const v_t = self.v[i];

            for (w, dw, m_t, v_t) |*weight, grad, *m_i, *v_i| {
                m_i.* = self.beta1 * m_i.* + (1.0 - self.beta1) * grad;
                v_i.* = self.beta2 * v_i.* + (1.0 - self.beta2) * grad * grad;
                weight.* -= lr_t * m_i.* / (@sqrt(v_i.*) + self.eps);
            }
        }
    }
};
