//! 有限差分梯度检查 (Finite-Difference Gradient Check) 测试：
//! 对每个可微算子比较反向传播得到的解析梯度与中心差分数值梯度 `(L(x + h) - L(x - h)) / 2h`。
//! 非标量输出先经确定性伪随机权重投影为标量损失 `L = sum(out ⊙ R)`，避免各输出分量的梯度互相抵消。

const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Graph = @import("graph.zig").Graph;

/// 梯度检查的单个输入张量描述
const Input = struct {
    shape: []const usize,
    /// 显式数据；为 null 时按 `fill` 规则生成
    data: ?[]const f32 = null,
    fill: Fill = .uniform,
    scale: f32 = 1.0,
    offset: f32 = 0.0,
    requires_grad: bool = true,

    const Fill = enum {
        /// 在 `[offset - scale, offset + scale]` 内均匀采样
        uniform,
        /// 互不相同且间隔为 `scale` 的取值 (`perm(i) * scale + offset`)，用于最大值池化、TopK 等分段可微算子
        distinct,
    };
};

const Options = struct {
    eps: f32 = 1e-2,
    rtol: f32 = 2e-2,
    atol: f32 = 2e-3,
};

fn projectionWeight(i: usize) f32 {
    return @sin(@as(f32, @floatFromInt(i)) * 0.731 + 0.25) + 0.1;
}

/// 把任意形状的输出投影为标量损失 `sum(out ⊙ R)`；单元素输出直接作为损失
fn projectToScalar(g: *Graph, out: *Tensor) !*Tensor {
    if (out.shape.numel() == 1) return out;
    const r = try g.tensorND(out.shape.dims[0..out.shape.len], false);
    for (r.data, 0..) |*v, i| v.* = projectionWeight(i);
    const prod = try g.mul(out, r);
    return g.sum(prod, null, false);
}

fn generateData(allocator: std.mem.Allocator, input: Input, seed: u64) ![]f32 {
    var numel: usize = 1;
    for (input.shape) |d| numel *= d;
    const buf = try allocator.alloc(f32, numel);
    if (input.data) |d| {
        std.debug.assert(d.len == numel);
        @memcpy(buf, d);
        return buf;
    }
    switch (input.fill) {
        .uniform => {
            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();
            for (buf) |*v| v.* = (rand.float(f32) * 2.0 - 1.0) * input.scale + input.offset;
        },
        .distinct => {
            for (buf, 0..) |*v, i| {
                const perm = (i * 7919 + 3) % numel;
                v.* = @as(f32, @floatFromInt(perm)) * input.scale + input.offset;
            }
        },
    }
    return buf;
}

fn buildInputs(g: *Graph, inputs: []const Input, values: []const []f32) ![]*Tensor {
    const ts = try g.arenaAllocator().alloc(*Tensor, inputs.len);
    for (inputs, values, 0..) |inp, vals, i| {
        ts[i] = try g.tensorNDWithData(inp.shape, vals, inp.requires_grad);
    }
    return ts;
}

fn evalLoss(allocator: std.mem.Allocator, comptime Builder: type, inputs: []const Input, values: []const []f32) !f32 {
    var g = Graph.initDefault(allocator);
    defer g.deinit();
    const ts = try buildInputs(&g, inputs, values);
    const out = try Builder.build(&g, ts);
    const loss = try projectToScalar(&g, out);
    return loss.data[0];
}

/// 比较 `Builder.build(graph, inputs)` 输出对每个 `requires_grad` 输入的解析梯度与中心差分数值梯度
fn expectGradientsMatch(comptime Builder: type, inputs: []const Input, options: Options) !void {
    const allocator = std.testing.allocator;

    const values = try allocator.alloc([]f32, inputs.len);
    defer allocator.free(values);
    var num_values: usize = 0;
    defer for (values[0..num_values]) |v| allocator.free(v);
    for (inputs, 0..) |inp, i| {
        values[i] = try generateData(allocator, inp, 0x5eed + i * 101);
        num_values += 1;
    }

    const analytic = try allocator.alloc([]f32, inputs.len);
    defer allocator.free(analytic);
    var num_analytic: usize = 0;
    defer for (analytic[0..num_analytic]) |a| allocator.free(a);
    {
        var g = Graph.initDefault(allocator);
        defer g.deinit();
        const ts = try buildInputs(&g, inputs, values);
        const out = try Builder.build(&g, ts);
        const loss = try projectToScalar(&g, out);
        try g.backward(loss);
        for (ts, 0..) |t, i| {
            analytic[i] = try allocator.alloc(f32, values[i].len);
            num_analytic += 1;
            if (t.requires_grad) @memcpy(analytic[i], t.grad) else @memset(analytic[i], 0.0);
        }
    }

    for (inputs, 0..) |inp, i| {
        if (!inp.requires_grad) continue;
        for (0..values[i].len) |j| {
            const orig = values[i][j];
            values[i][j] = orig + options.eps;
            const loss_plus = try evalLoss(allocator, Builder, inputs, values);
            values[i][j] = orig - options.eps;
            const loss_minus = try evalLoss(allocator, Builder, inputs, values);
            values[i][j] = orig;

            const numeric = (loss_plus - loss_minus) / (2.0 * options.eps);
            const ana = analytic[i][j];
            const tol = options.atol + options.rtol * @max(@abs(numeric), @abs(ana));
            if (!(@abs(numeric - ana) <= tol)) {
                std.debug.print(
                    "\n[gradcheck] {s}: input {d} element {d}: analytic={d:.6} numeric={d:.6}\n",
                    .{ @typeName(Builder), i, j, ana, numeric },
                );
                return error.GradientMismatch;
            }
        }
    }
}

// ============================================================================
// 逐元素与标量运算 (Element-wise & Scalar Ops)
// ============================================================================

test "gradcheck: broadcast add/sub/mul/div" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const s = try g.add(t[0], t[1]);
            const d = try g.sub(s, t[2]);
            const m = try g.mul(d, t[1]);
            return g.div(m, t[3]);
        }
    }, &.{
        .{ .shape = &.{ 2, 3, 4 } },
        .{ .shape = &.{ 3, 1 } },
        .{ .shape = &.{4} },
        .{ .shape = &.{ 1, 3, 4 }, .offset = 2.0, .scale = 0.5 },
    }, .{});
}

test "gradcheck: scalar ops" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const a = try g.mulScalar(t[0], 1.5);
            const b = try g.addScalar(a, 0.25);
            const c = try g.subScalar(b, 2.0);
            return g.divScalar(c, 4.0);
        }
    }, &.{.{ .shape = &.{ 3, 2 } }}, .{});
}

test "gradcheck: elementary math sqrt/exp/log/abs" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const s = try g.sqrt(t[0]);
            const l = try g.log(t[0]);
            const e = try g.exp(t[1]);
            const a = try g.abs(t[2]);
            const sl = try g.add(s, l);
            const ea = try g.mul(e, a);
            return g.add(sl, ea);
        }
    }, &.{
        .{ .shape = &.{ 2, 3 }, .offset = 2.0, .scale = 0.5 },
        .{ .shape = &.{ 2, 3 } },
        .{ .shape = &.{ 2, 3 }, .fill = .distinct, .scale = 0.3, .offset = -0.85 },
    }, .{});
}

// ============================================================================
// 矩阵乘法 (Matrix Multiplication)
// ============================================================================

test "gradcheck: matmul" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.matmul(t[0], t[1]);
        }
    }, &.{ .{ .shape = &.{ 3, 4 } }, .{ .shape = &.{ 4, 5 } } }, .{});
}

test "gradcheck: matmul with transposed view operands" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const a_t = try t[0].transposeView(0, 1, g.arenaAllocator());
            const b_t = try t[1].transposeView(0, 1, g.arenaAllocator());
            return g.matmul(a_t, b_t);
        }
    }, &.{ .{ .shape = &.{ 4, 3 } }, .{ .shape = &.{ 5, 4 } } }, .{});
    // 权重共享 (Weight Tying) 场景：y = x W^T，W 以零拷贝转置视图参与乘法
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const w_t = try t[1].transposeView(0, 1, g.arenaAllocator());
            return g.matmul(t[0], w_t);
        }
    }, &.{ .{ .shape = &.{ 2, 3 } }, .{ .shape = &.{ 4, 3 } } }, .{});
}

test "gradcheck: matmul with leading batch dimensions on the left operand" {
    // [B, T, K] x [K, N]：前导维度展平为行后与共享右操作数相乘
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.matmul(t[0], t[1]);
        }
    }, &.{ .{ .shape = &.{ 2, 3, 4 } }, .{ .shape = &.{ 4, 5 } } }, .{});
    // 四维左操作数，且右操作数为转置视图 (权重共享 (Weight Tying) 投影)
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const w_t = try t[1].transposeView(0, 1, g.arenaAllocator());
            return g.matmul(t[0], w_t);
        }
    }, &.{ .{ .shape = &.{ 2, 1, 3, 4 } }, .{ .shape = &.{ 3, 4 } } }, .{});
    // 非连续左操作数 (转置后的批次视图) 先物化再展平
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const x_t = try g.transposeND(t[0], 0, 1);
            return g.matmul(x_t, t[1]);
        }
    }, &.{ .{ .shape = &.{ 3, 2, 4 } }, .{ .shape = &.{ 4, 2 } } }, .{});
}

test "gradcheck: q4MatMul input gradient through frozen quantized weights" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const quant = tensor.quant;
            const in_f = 64;
            const out_f = 3;
            const raw = try g.arenaAllocator().alloc(f32, in_f * out_f);
            for (raw, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.37);
            const blocks = try g.arenaAllocator().alloc(quant.Q4Block, raw.len / quant.Q4Block.QK);
            quant.quantizeSlice(blocks, raw);
            return g.q4MatMul(t[0], blocks, in_f, out_f);
        }
    }, &.{.{ .shape = &.{ 2, 64 } }}, .{});
}

test "gradcheck: batchMatMul with transpose" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const k_t = try g.transposeND(t[1], 2, 3);
            return g.batchMatMul(t[0], k_t);
        }
    }, &.{ .{ .shape = &.{ 2, 2, 3, 4 } }, .{ .shape = &.{ 2, 2, 3, 4 } } }, .{});
}

// ============================================================================
// 形状与维度变换 (Shape & Dimension Transforms)
// ============================================================================

test "gradcheck: reshape/transpose/concat/split/repeatKV/slice" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const r = try g.reshape(t[0], &.{ 2, 2, 3, 2 });
            const tr = try g.transposeND(r, 1, 2);
            const parts = try g.split(tr, 3, 1);
            const cat = try g.concat(&.{ parts[2], parts[0] }, 1);
            const rep = try g.repeatKV(cat, 2);
            const sl = try g.slice(rep, &.{ .{}, .{ .start = 1, .step = 2 }, .{ .end = 1 }, .{} });
            const sq = try g.squeeze(sl, 2);
            return g.unsqueeze(sq, 0);
        }
    }, &.{.{ .shape = &.{ 4, 6 } }}, .{});
}

// ============================================================================
// 激活函数 (Activation Functions)
// ============================================================================

test "gradcheck: relu/leakyRelu/gelu/silu/sigmoid/tanh" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const r = try g.relu(t[0]);
            const lr = try g.leakyRelu(t[0], 0.1);
            const ge = try g.gelu(t[1]);
            const si = try g.silu(t[1]);
            const sg = try g.sigmoid(t[1]);
            const th = try g.tanh(t[1]);
            const a = try g.add(r, lr);
            const b = try g.add(ge, si);
            const c = try g.mul(sg, th);
            const ab = try g.add(a, b);
            return g.add(ab, c);
        }
    }, &.{
        .{ .shape = &.{ 2, 5 }, .fill = .distinct, .scale = 0.2, .offset = -0.9 },
        .{ .shape = &.{ 2, 5 }, .scale = 2.0 },
    }, .{});
}

test "gradcheck: softmax over last dimension" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.softmax(t[0]);
        }
    }, &.{.{ .shape = &.{ 2, 3, 4 }, .scale = 2.0 }}, .{});
}

// ============================================================================
// 归约与选择 (Reductions & Selection)
// ============================================================================

test "gradcheck: sum/mean/variance/stdDev" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const s0 = try g.sum(t[0], 1, true);
            const s1 = try g.mean(t[0], 0, false);
            const v = try g.variance(t[0], 2, false, 1);
            const sd = try g.stdDev(t[0], null, false, 0);
            const a = try g.sum(s0, null, false);
            const b = try g.sum(s1, null, false);
            const c = try g.sum(v, null, false);
            const ab = try g.add(a, b);
            const abc = try g.add(ab, c);
            return g.add(abc, sd);
        }
    }, &.{.{ .shape = &.{ 2, 3, 4 } }}, .{});
}

test "gradcheck: where and maskedFill" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const mask = try tensor.BoolTensor.fromSlice(g.arenaAllocator(), &.{ 2, 3 }, &.{ true, false, true, false, false, true });
            const cond = try tensor.BoolTensor.fromSlice(g.arenaAllocator(), &.{3}, &.{ false, true, true });
            const filled = try g.maskedFill(t[0], mask, -2.0);
            return g.where(cond, filled, t[1]);
        }
    }, &.{ .{ .shape = &.{ 2, 3 } }, .{ .shape = &.{ 1, 3 } } }, .{});
}

// ============================================================================
// 卷积与池化 (Convolution & Pooling)
// ============================================================================

test "gradcheck: conv1d with stride, padding and bias" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.conv1d(t[0], t[1], t[2], .{ .stride = 2, .padding = 1 });
        }
    }, &.{
        .{ .shape = &.{ 2, 2, 7 } },
        .{ .shape = &.{ 3, 2, 3 } },
        .{ .shape = &.{3} },
    }, .{});
}

test "gradcheck: conv2d with stride, padding and bias" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.conv2d(t[0], t[1], t[2], .{ .stride = 2, .padding = 1 });
        }
    }, &.{
        .{ .shape = &.{ 2, 2, 5, 4 } },
        .{ .shape = &.{ 3, 2, 3, 2 } },
        .{ .shape = &.{3} },
    }, .{});
}

test "gradcheck: conv2d on a non-contiguous input view" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const x = try t[0].transposeView(2, 3, g.arenaAllocator());
            return g.conv2d(x, t[1], null, .{ .padding = 1 });
        }
    }, &.{
        .{ .shape = &.{ 1, 2, 4, 3 } },
        .{ .shape = &.{ 2, 2, 2, 2 } },
    }, .{});
}

test "gradcheck: convTranspose1d and convTranspose2d" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.convTranspose1d(t[0], t[1], t[2], .{ .stride = 2, .padding = 1 });
        }
    }, &.{
        .{ .shape = &.{ 2, 2, 4 } },
        .{ .shape = &.{ 2, 3, 3 } },
        .{ .shape = &.{3} },
    }, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.convTranspose2d(t[0], t[1], t[2], .{ .stride = 2, .padding = 1 });
        }
    }, &.{
        .{ .shape = &.{ 1, 2, 3, 2 } },
        .{ .shape = &.{ 2, 2, 3, 3 } },
        .{ .shape = &.{2} },
    }, .{});
}

test "gradcheck: convTranspose without bias, unit stride and kernel wider than stride" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.convTranspose1d(t[0], t[1], null, .{});
        }
    }, &.{
        .{ .shape = &.{ 1, 3, 3 } },
        .{ .shape = &.{ 3, 2, 2 } },
    }, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.convTranspose2d(t[0], t[1], null, .{ .stride = 2, .padding = 2 });
        }
    }, &.{
        .{ .shape = &.{ 2, 1, 3, 3 } },
        .{ .shape = &.{ 1, 2, 4, 4 } },
    }, .{});
}

test "gradcheck: max/avg/adaptive pooling" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const mp = try g.maxpool1d(t[0], 3, .{ .stride = 2, .padding = 1 });
            const ap = try g.avgpool1d(t[0], 3, .{ .stride = 2, .padding = 1 });
            const ad = try g.adaptiveAvgPool1d(t[0], 3);
            const a = try g.add(mp, ap);
            const sa = try g.sum(a, null, false);
            const sd = try g.sum(ad, null, false);
            return g.add(sa, sd);
        }
    }, &.{.{ .shape = &.{ 2, 2, 7 }, .fill = .distinct, .scale = 0.1, .offset = -1.0 }}, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const mp = try g.maxpool2d(t[0], 2, .{ .stride = 2, .padding = 1 });
            const ap = try g.avgpool2d(t[0], 3, .{ .stride = 2, .padding = 1 });
            const ad = try g.adaptiveAvgPool2d(t[0], .{ 2, 3 });
            const sm = try g.sum(mp, null, false);
            const sa = try g.sum(ap, null, false);
            const sd = try g.sum(ad, null, false);
            const x = try g.add(sm, sa);
            return g.add(x, sd);
        }
    }, &.{.{ .shape = &.{ 1, 2, 5, 4 }, .fill = .distinct, .scale = 0.1, .offset = -2.0 }}, .{});
}

// ============================================================================
// 归一化 (Normalization)
// ============================================================================

test "gradcheck: rmsNorm and layerNorm" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const r = try g.rmsNorm(t[0], t[1], 1e-5);
            return g.layerNorm(r, t[2], t[3], 1e-5);
        }
    }, &.{
        .{ .shape = &.{ 3, 4 } },
        .{ .shape = &.{4}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{4}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{4} },
    }, .{});
}

test "gradcheck: batchNorm1d training and evaluation" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const rm = try g.zeros(&.{3}, false);
            const rv = try g.ones(&.{3}, false);
            return g.batchNorm1d(t[0], t[1], t[2], rm, rv, 1e-5, 0.1, true);
        }
    }, &.{
        .{ .shape = &.{ 4, 3, 2 } },
        .{ .shape = &.{3}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{3} },
    }, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const rm = try g.array(&.{3}, &.{ 0.1, -0.2, 0.3 }, false);
            const rv = try g.array(&.{3}, &.{ 0.5, 1.5, 2.0 }, false);
            return g.batchNorm1d(t[0], t[1], t[2], rm, rv, 1e-5, 0.1, false);
        }
    }, &.{
        .{ .shape = &.{ 4, 3 } },
        .{ .shape = &.{3}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{3} },
    }, .{});
}

test "gradcheck: batchNorm2d training and groupNorm" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const rm = try g.zeros(&.{2}, false);
            const rv = try g.ones(&.{2}, false);
            return g.batchNorm2d(t[0], t[1], t[2], rm, rv, 1e-5, 0.1, true);
        }
    }, &.{
        .{ .shape = &.{ 2, 2, 2, 3 } },
        .{ .shape = &.{2}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{2} },
    }, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.groupNorm(t[0], t[1], t[2], 2, 1e-5);
        }
    }, &.{
        .{ .shape = &.{ 2, 4, 3 } },
        .{ .shape = &.{4}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{4} },
    }, .{});
}

// ============================================================================
// 结构性算子 (Structural Ops)
// ============================================================================

test "gradcheck: dropout with a fixed seed" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            var prng = std.Random.DefaultPrng.init(7);
            return g.dropout(t[0], 0.3, prng.random());
        }
    }, &.{.{ .shape = &.{ 3, 4 } }}, .{});
}

test "gradcheck: rope interleaved and split_half with offset and partial rotation" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.rope(t[0], 3, .{ .rotary_offset = 1, .mode = .interleaved, .base = 100.0 });
        }
    }, &.{.{ .shape = &.{ 2, 3, 6 } }}, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.rope(t[0], 1, .{ .rotary_offset = 2, .mode = .split_half, .partial_rotary_factor = 0.5, .base = 100.0 });
        }
    }, &.{.{ .shape = &.{ 2, 3, 10 } }}, .{});
}

test "gradcheck: embedding" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const idx = try g.array(&.{ 2, 3 }, &.{ 0, 3, 1, 3, 3, 2 }, false);
            return g.embedding(t[0], idx);
        }
    }, &.{.{ .shape = &.{ 4, 3 } }}, .{});
}

test "gradcheck: topK values" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const res = try g.topK(t[0], 2);
            return res.values;
        }
    }, &.{.{ .shape = &.{ 3, 5 }, .fill = .distinct, .scale = 0.1 }}, .{});
}

test "gradcheck: gatherMatMul with shared and per-slot inputs" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const idx = try g.array(&.{ 3, 2 }, &.{ 0, 1, 1, 0, 1, 1 }, false);
            return g.gatherMatMul(t[0], idx, &.{ t[1], t[2] }, &.{ t[3], t[4] });
        }
    }, &.{
        .{ .shape = &.{ 3, 4 } },
        .{ .shape = &.{ 4, 2 } },
        .{ .shape = &.{ 4, 2 } },
        .{ .shape = &.{2} },
        .{ .shape = &.{2} },
    }, .{});
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const idx = try g.array(&.{ 2, 2 }, &.{ 1, 0, 0, 0 }, false);
            return g.gatherMatMul(t[0], idx, &.{ t[1], t[2] }, null);
        }
    }, &.{
        .{ .shape = &.{ 2, 2, 3 } },
        .{ .shape = &.{ 3, 2 } },
        .{ .shape = &.{ 3, 2 } },
    }, .{});
}

test "gradcheck: scaled dot-product attention with causal mask" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.scaledDotProductAttention(t[0], t[1], t[2], 0.5, true);
        }
    }, &.{
        .{ .shape = &.{ 1, 2, 3, 4 } },
        .{ .shape = &.{ 1, 2, 3, 4 } },
        .{ .shape = &.{ 1, 2, 3, 4 } },
    }, .{});
}

test "gradcheck: contiguous-layout kernels receive transposed view inputs" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const arena = g.arenaAllocator();
            const xt = try t[0].transposeView(0, 1, arena); // [3, 4]
            const sm = try g.softmax(xt);
            const th = try g.tanh(xt);
            const ln = try g.layerNorm(xt, t[1], t[2], 1e-5);
            const sc = try g.mulScalar(xt, 0.5);
            const cat = try g.concat(&.{ sm, th, ln, sc }, 0); // [12, 4]
            const parts = try g.split(xt, 2, 1);
            const ps = try g.sub(parts[0], parts[1]); // [3, 2]
            const mse = try g.mseLoss(ps, try g.zeros(&.{ 3, 2 }, false));

            const kt = try t[3].transposeView(2, 3, arena); // [1, 2, 3, 4]
            const att = try g.batchMatMul(t[4], kt); // [1, 2, 2, 4]
            const sa = try g.sum(att, null, false);
            const sc_sum = try g.sum(cat, null, false);
            const sq = try g.sum(try g.mul(cat, cat), null, false);
            const a = try g.add(sa, sc_sum);
            const b = try g.add(a, sq);
            return g.add(b, mse);
        }
    }, &.{
        .{ .shape = &.{ 4, 3 } },
        .{ .shape = &.{4}, .offset = 1.0, .scale = 0.3 },
        .{ .shape = &.{4} },
        .{ .shape = &.{ 1, 2, 4, 3 } },
        .{ .shape = &.{ 1, 2, 2, 3 } },
    }, .{});
}

// ============================================================================
// 损失函数 (Loss Functions)
// ============================================================================

test "gradcheck: mse/bce/bceWithLogits losses" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const mse = try g.mseLoss(t[0], t[1]);
            const probs = try g.sigmoid(t[0]);
            const bce = try g.bceLoss(probs, t[2], 1e-7);
            const bwl = try g.bceWithLogitsLoss(t[0], t[2]);
            const a = try g.add(mse, bce);
            return g.add(a, bwl);
        }
    }, &.{
        .{ .shape = &.{ 2, 3 } },
        .{ .shape = &.{ 2, 3 } },
        .{ .shape = &.{ 2, 3 }, .offset = 0.5, .scale = 0.4 },
    }, .{});
}

test "gradcheck: softmaxCrossEntropy and maskedCrossEntropyLoss" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const ce = try g.softmaxCrossEntropy(t[0], &[_]u8{ 2, 0, 1 });
            const mce = try g.maskedCrossEntropyLoss(t[0], &[_]u8{ 1, 1, 3 }, &.{ 1.0, 0.0, 0.5 });
            return g.add(ce, mce);
        }
    }, &.{.{ .shape = &.{ 3, 4 }, .scale = 2.0 }}, .{});
}

test "gradcheck: dpo and grpo losses" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.dpoLoss(t[0], t[1], &.{ -1.0, -2.0, 0.5 }, &.{ -1.5, -0.5, 0.0 }, 0.7);
        }
    }, &.{ .{ .shape = &.{3} }, .{ .shape = &.{3} } }, .{});
    // 比率 exp(new - old) 落在裁剪区间内部与外部各若干个，避开不可导的裁剪边界
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            const old = try g.array(&.{4}, &.{ 0.0, 0.0, 0.0, 0.0 }, false);
            return g.grpoLoss(old, t[0], &.{ 1.0, -0.5, 2.0, -1.5 }, &.{ 0.1, -0.2, 0.0, 0.3 }, 0.05, 0.2);
        }
    }, &.{.{ .shape = &.{4}, .data = &.{ 0.05, -0.1, 0.5, -0.6 } }}, .{});
}

test "gradcheck: l1/l2/elasticNet regularization" {
    try expectGradientsMatch(struct {
        fn build(g: *Graph, t: []const *Tensor) !*Tensor {
            return g.elasticNetLoss(t[0], t[1], t[2], 0.3, 0.4);
        }
    }, &.{
        .{ .shape = &.{ 2, 2 } },
        .{ .shape = &.{ 2, 2 } },
        .{ .shape = &.{ 2, 3 }, .fill = .distinct, .scale = 0.2, .offset = -0.5 },
    }, .{});
}
