const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const graph_mod = @import("graph.zig");
const Graph = graph_mod.Graph;

test "autodiff Graph enable_grad and setGradEnabled" {
    const allocator = std.testing.allocator;

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // 默认 enable_grad = true
    try std.testing.expect(graph.enable_grad);

    const x = try graph.tensorWithData(2, 2, &.{ 1.0, 2.0, 3.0, 4.0 }, false);
    const w = try graph.tensorWithData(2, 2, &.{ 0.5, -0.5, 1.0, 2.0 }, true);

    const y = try graph.matmul(x, w);
    try std.testing.expect(y.requires_grad);
    try std.testing.expectEqual(@as(usize, 4), y.grad.len);
    try std.testing.expect(y.creator != null);
    try std.testing.expectEqual(@as(usize, 1), graph.ops.items.len);

    // 关闭梯度：setGradEnabled(false)
    var graph_nograd = Graph.init(allocator);
    defer graph_nograd.deinit();
    graph_nograd.setGradEnabled(false);
    try std.testing.expect(!graph_nograd.enable_grad);

    const x2 = try graph_nograd.tensorWithData(2, 2, &.{ 1.0, 2.0, 3.0, 4.0 }, false);
    const w2 = try graph_nograd.tensorWithData(2, 2, &.{ 0.5, -0.5, 1.0, 2.0 }, true);
    // 即使 w2 传入了 requires_grad = true，在 graph_nograd 下也应被强制置为 false
    try std.testing.expect(!w2.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), w2.grad.len);

    const y2 = try graph_nograd.matmul(x2, w2);
    // 验证前向数值正确
    try std.testing.expectApproxEqAbs(y.data[0], y2.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[1], y2.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[2], y2.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[3], y2.data[3], 1e-5);

    // 验证无任何梯度内存与 Op 记录
    try std.testing.expect(!y2.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), y2.grad.len);
    try std.testing.expect(y2.creator == null);
    try std.testing.expectEqual(@as(usize, 0), graph_nograd.ops.items.len);
}

test "autodiff broadcasting add, sub, mul, div backward reduction" {
    const allocator = std.testing.allocator;

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // 1. Add broadcasting: A [2, 3] + B [1, 3] -> C [2, 3]
    const a = try graph.tensorWithData(2, 3, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, true);
    const b = try graph.tensorWithData(1, 3, &.{ 10.0, 20.0, 30.0 }, true);
    const out_c = try graph.add(a, b);

    // 设定输出的伪梯度 dC = all 1.0 (模拟 sum(C))
    @memset(out_c.grad, 1.0);
    try graph.backwardWithGrad(out_c);

    // dA 应为全部 1.0
    for (a.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-5);
    }
    // dB 应为沿第 0 维求和，即每列 1.0 + 1.0 = 2.0
    for (b.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), g, 1e-5);
    }

    // 2. Mul broadcasting: A2 [2, 2] * B2 [1, 2] -> C2 [2, 2]
    var graph2 = Graph.init(allocator);
    defer graph2.deinit();

    const a2 = try graph2.tensorWithData(2, 2, &.{ 1.0, 2.0, 3.0, 4.0 }, true);
    const b2 = try graph2.tensorWithData(1, 2, &.{ 10.0, 20.0 }, true);
    const out_c2 = try graph2.mul(a2, b2);

    @memset(out_c2.grad, 1.0);
    try graph2.backwardWithGrad(out_c2);

    // dA2: C.grad * B2.data -> [10.0, 20.0, 10.0, 20.0]
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), a2.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), a2.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), a2.grad[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), a2.grad[3], 1e-5);

    // dB2: 沿第 0 维求和 C.grad * A2.data -> [1.0 + 3.0, 2.0 + 4.0] = [4.0, 6.0]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), b2.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), b2.grad[1], 1e-5);

    // 3. Sub broadcasting: A3 [2, 2] - B3 [1, 2] -> C3 [2, 2]
    var graph3 = Graph.init(allocator);
    defer graph3.deinit();

    const a3 = try graph3.tensorWithData(2, 2, &.{ 10.0, 20.0, 30.0, 40.0 }, true);
    const b3 = try graph3.tensorWithData(1, 2, &.{ 1.0, 2.0 }, true);
    const out_c3 = try graph3.sub(a3, b3);

    @memset(out_c3.grad, 1.0);
    try graph3.backwardWithGrad(out_c3);

    // dA3 = +1.0
    for (a3.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-5);
    }
    // dB3 = - (1.0 + 1.0) = -2.0
    for (b3.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, -2.0), g, 1e-5);
    }

    // 4. Div broadcasting: A4 [2, 2] / B4 [1, 2] -> C4 [2, 2]
    var graph4 = Graph.init(allocator);
    defer graph4.deinit();

    const a4 = try graph4.tensorWithData(2, 2, &.{ 10.0, 20.0, 30.0, 40.0 }, true);
    const b4 = try graph4.tensorWithData(1, 2, &.{ 2.0, 4.0 }, true);
    const out_c4 = try graph4.div(a4, b4);

    @memset(out_c4.grad, 1.0);
    try graph4.backwardWithGrad(out_c4);

    // dA4: 1.0 / B4 -> [0.5, 0.25, 0.5, 0.25]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), a4.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), a4.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), a4.grad[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), a4.grad[3], 1e-5);

    // dB4: - sum(A4 / B4^2)
    // col 0: -(10 / 4 + 30 / 4) = -40 / 4 = -10.0
    // col 1: -(20 / 16 + 40 / 16) = -60 / 16 = -3.75
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), b4.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -3.75), b4.grad[1], 1e-5);
}

test "autodiff Graph.forward re-evaluates recorded ops consistently with initial forward" {
    const allocator = std.testing.allocator;

    var graph = Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensorWithData(2, 2, &.{ 1.0, -1.0, 2.0, 0.5 }, true);
    const w = try graph.tensorWithData(2, 2, &.{ 0.5, 1.5, -1.0, 2.0 }, true);
    const b = try graph.tensorWithData(1, 2, &.{ 0.1, -0.2 }, true);
    const g_scale = try graph.tensorWithData(1, 2, &.{ 1.0, 1.0 }, true);
    const g_shift = try graph.tensorWithData(1, 2, &.{ 0.0, 0.0 }, true);

    const mm = try graph.matmul(x, w);
    const added = try graph.add(mm, b);
    const act = try graph.silu(added);
    const normed = try graph.layerNorm(act, g_scale, g_shift, 1e-5);
    const labels = [_]usize{ 0, 1 };
    const ce = try graph.softmaxCrossEntropy(normed, &labels);
    const l2 = try graph.l2Loss(w, 0.1);
    const total = try graph.add(ce, l2);

    const initial_total = total.data[0];

    // Modify x in-place and re-run graph.forward(), then compare with a fresh graph built on the updated x
    x.data[0] = 3.0;
    x.data[3] = -2.0;
    try graph.forward();
    const reevaluated_total = total.data[0];
    try std.testing.expect(@abs(reevaluated_total - initial_total) > 1e-4);

    var fresh_graph = Graph.init(allocator);
    defer fresh_graph.deinit();
    const x2 = try fresh_graph.tensorWithData(2, 2, &.{ 3.0, -1.0, 2.0, -2.0 }, true);
    const w2 = try fresh_graph.tensorWithData(2, 2, &.{ 0.5, 1.5, -1.0, 2.0 }, true);
    const b2 = try fresh_graph.tensorWithData(1, 2, &.{ 0.1, -0.2 }, true);
    const g_scale2 = try fresh_graph.tensorWithData(1, 2, &.{ 1.0, 1.0 }, true);
    const g_shift2 = try fresh_graph.tensorWithData(1, 2, &.{ 0.0, 0.0 }, true);

    const mm2 = try fresh_graph.matmul(x2, w2);
    const added2 = try fresh_graph.add(mm2, b2);
    const act2 = try fresh_graph.silu(added2);
    const normed2 = try fresh_graph.layerNorm(act2, g_scale2, g_shift2, 1e-5);
    const ce2 = try fresh_graph.softmaxCrossEntropy(normed2, &labels);
    const l2_2 = try fresh_graph.l2Loss(w2, 0.1);
    const total2 = try fresh_graph.add(ce2, l2_2);

    try std.testing.expectApproxEqAbs(total2.data[0], reevaluated_total, 1e-6);
}

test "Autograd reductions, elementary math, where/maskedFill, and squeeze/unsqueeze/slice" {
    const allocator = std.testing.allocator;

    // 1. Elementary math (sqrt, exp, log, abs) + sum/mean backward
    {
        var graph = Graph.init(allocator);
        defer graph.deinit();

        const x = try graph.array(&.{2}, &.{ 4.0, -9.0 }, true);
        // abs(x) = [4.0, 9.0], sqrt(abs(x)) = [2.0, 3.0]
        const ax = try x.abs(allocator, &graph);
        const sx = try ax.sqrt(allocator, &graph);
        // log(exp(sx)) = sx = [2.0, 3.0]
        const ex = try sx.exp(allocator, &graph);
        const lx = try ex.log(allocator, &graph);
        // mean(lx) = 2.5
        const loss = try lx.mean(null, false, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), loss.data[0], 1e-5);

        try graph.backward(loss);
        // d(mean)/dx_i = 0.5 * 1.0 * sign(x_i) / (2 * sqrt(|x_i|))
        // x_0 = 4.0 -> 0.5 * 1 / (2 * 2) = 0.125
        // x_1 = -9.0 -> 0.5 * (-1) / (2 * 3) = -1/12 = -0.0833333
        try std.testing.expectApproxEqAbs(@as(f32, 0.125), x.grad[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, -1.0 / 12.0), x.grad[1], 1e-5);
    }

    // 2. Axis sum, variance, squeeze, unsqueeze, slice, where, and maskedFill backward
    {
        var graph = Graph.init(allocator);
        defer graph.deinit();

        // A: [2, 3] = [[1, 2, 3], [4, 5, 6]]
        const A = try graph.array(&.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, true);
        // Slice cols 1..3 -> [[2, 3], [5, 6]] (shape [2, 2])
        const S = try A.slice(&.{ .{ .start = 0, .end = 2 }, .{ .start = 1, .end = 3 } }, allocator, &graph);
        try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0, 5.0, 6.0 }, S.data);

        // Unsqueeze & squeeze roundtrip
        const U = try S.unsqueeze(1, allocator, &graph);
        try std.testing.expectEqual(@as(usize, 3), U.shape.len);
        const Sq = try U.squeeze(1, allocator, &graph);
        try std.testing.expectEqual(@as(usize, 2), Sq.shape.len);

        // BoolTensor mask on Sq: mask out first element [true, false; false, true]
        const mask = try tensor.BoolTensor.fromSlice(allocator, &.{ 2, 2 }, &.{ true, false, false, true });
        defer mask.deinit(allocator);

        // maskedFill(Sq, mask, 0.0) -> [[0, 3], [5, 0]]
        const MF = try Sq.maskedFill(mask, 0.0, allocator, &graph);
        try std.testing.expectEqualSlices(f32, &.{ 0.0, 3.0, 5.0, 0.0 }, MF.data);

        // where(mask, Sq, MF): where mask is true take Sq (2, 6), else take MF (3, 5) -> [[2, 3], [5, 6]]
        const W = try Tensor.where(mask, Sq, MF, allocator, &graph);
        try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0, 5.0, 6.0 }, W.data);

        // variance along axis 1 (ddof=0):
        // row 0: [2, 3], mean=2.5, var = ((2-2.5)^2 + (3-2.5)^2)/2 = 0.25
        // row 1: [5, 6], mean=5.5, var = 0.25
        const V = try W.variance(1, false, 0, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), V.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), V.data[1], 1e-5);

        const total = try V.sum(null, false, allocator, &graph);
        try graph.backward(total);

        // d(var)/dw_j = 2*(w_j - mean)/2 = w_j - mean
        // row 0: w_0=2 -> -0.5, w_1=3 -> +0.5
        // row 1: w_0=5 -> -0.5, w_1=6 -> +0.5
        // And since col 0 of A was sliced out, its gradient is 0.0!
        try std.testing.expectEqualSlices(f32, &.{
            0.0, -0.5, 0.5,
            0.0, -0.5, 0.5,
        }, A.grad);
    }
}

