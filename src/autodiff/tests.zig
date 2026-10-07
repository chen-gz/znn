const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const graph_mod = @import("graph.zig");
const Graph = graph_mod.Graph;
const autodiff = @import("../autodiff.zig");
const Shape = tensor.Shape;
const computeContiguousStrides = tensor.computeContiguousStrides;
const transposeShape = tensor.transposeShape;
const array = tensor.array;
const zeros = tensor.zeros;
const ones = tensor.ones;
const free = tensor.free;

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
        const ax = try graph.abs(x);
        const sx = try graph.sqrt(ax);
        // log(exp(sx)) = sx = [2.0, 3.0]
        const ex = try graph.exp(sx);
        const lx = try graph.log(ex);
        // mean(lx) = 2.5
        const loss = try graph.mean(lx, null, false);
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
        const S = try graph.slice(A, &.{ .{ .start = 0, .end = 2 }, .{ .start = 1, .end = 3 } });
        try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0, 5.0, 6.0 }, S.data);

        // Unsqueeze & squeeze roundtrip
        const U = try graph.unsqueeze(S, 1);
        try std.testing.expectEqual(@as(usize, 3), U.shape.len);
        const Sq = try graph.squeeze(U, 1);
        try std.testing.expectEqual(@as(usize, 2), Sq.shape.len);

        // BoolTensor mask on Sq: mask out first element [true, false; false, true]
        const mask = try tensor.BoolTensor.fromSlice(allocator, &.{ 2, 2 }, &.{ true, false, false, true });
        defer mask.deinit(allocator);

        // maskedFill(Sq, mask, 0.0) -> [[0, 3], [5, 0]]
        const MF = try graph.maskedFill(Sq, mask, 0.0);
        try std.testing.expectEqualSlices(f32, &.{ 0.0, 3.0, 5.0, 0.0 }, MF.data);

        // where(mask, Sq, MF): where mask is true take Sq (2, 6), else take MF (3, 5) -> [[2, 3], [5, 6]]
        const W = try graph.where(mask, Sq, MF);
        try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0, 5.0, 6.0 }, W.data);

        // variance along axis 1 (ddof=0):
        // row 0: [2, 3], mean=2.5, var = ((2-2.5)^2 + (3-2.5)^2)/2 = 0.25
        // row 1: [5, 6], mean=5.5, var = 0.25
        const V = try graph.variance(W, 1, false, 0);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), V.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), V.data[1], 1e-5);

        const total = try graph.sum(V, null, false);
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

test "Shape and strides helpers" {
    // Test Shape init & eq
    const s1 = Shape.init(&.{2, 3, 4});
    try std.testing.expectEqual(@as(usize, 3), s1.len);
    try std.testing.expectEqual(@as(usize, 2), s1.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), s1.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s1.dims[2]);

    const s2 = Shape.init(&.{2, 3, 4});
    try std.testing.expect(s1.eq(s2));

    const s3 = Shape.init(&.{2, 3, 5});
    try std.testing.expect(!s1.eq(s3));

    // Test computeContiguousStrides
    const strides1 = computeContiguousStrides(s1);
    try std.testing.expectEqual(@as(usize, 12), strides1.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), strides1.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), strides1.dims[2]);

    // Test transposeShape
    const s_trans = transposeShape(s1, 0, 1);
    try std.testing.expectEqual(@as(usize, 3), s_trans.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s_trans.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s_trans.dims[2]);
}

test "Tensor indexing and gradient operations" {
    const allocator = std.testing.allocator;
    const shape = Shape.init(&.{2, 3});
    const strides = computeContiguousStrides(shape);

    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    const grad = try allocator.alloc(f32, 6);
    defer allocator.free(grad);

    var t = Tensor{
        .data = data,
        .grad = grad,
        .shape = shape,
        .strides = strides,
        .requires_grad = true,
        .creator = null,
    };

    // Test indexing
    t.set(&.{0, 0}, 1.0);
    t.set(&.{0, 1}, 2.0);
    t.set(&.{0, 2}, 3.0);
    t.set(&.{1, 0}, 4.0);
    t.set(&.{1, 1}, 5.0);
    t.set(&.{1, 2}, 6.0);

    try std.testing.expectEqual(@as(f32, 1.0), t.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t.get(&.{1, 2}));
    try std.testing.expectEqual(@as(usize, 5), t.getFlatIndex(&.{1, 2}));

    // Test grad operations
    t.setGrad(&.{0, 1}, 10.0);
    try std.testing.expectEqual(@as(f32, 10.0), t.getGrad(&.{0, 1}));

    t.zeroGrad();
    try std.testing.expectEqual(@as(f32, 0.0), t.getGrad(&.{0, 1}));
}

test "NumPy-like raw tensor creation" {
    const allocator = std.testing.allocator;

    // Test array creation
    const t_arr = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t_arr);
    try std.testing.expectEqual(@as(f32, 1.0), t_arr.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t_arr.get(&.{1, 2}));

    // Test zeros creation
    const t_zeros = try zeros(allocator, &.{2, 2});
    defer free(allocator, t_zeros);
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{1, 1}));

    // Test ones creation
    const t_ones = try ones(allocator, &.{3, 1});
    defer free(allocator, t_ones);
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{2, 0}));
}

test "Direct tensor operations (eager and graph)" {
    const allocator = std.testing.allocator;

    // Eager Mode Test
    {
        const A = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
        defer free(allocator, A);
        const B = try array(allocator, &.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
        defer free(allocator, B);

        // Matmul
        const C = try A.matmul(B, allocator);
        defer free(allocator, C);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 6.4), C.get(&.{1, 1}), 1e-5);

        // AddBias
        const bias = try array(allocator, &.{1, 2}, &[_]f32{ 0.5, 1.0 });
        defer free(allocator, bias);
        const D = try C.addBias(bias, allocator);
        defer free(allocator, D);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 7.4), D.get(&.{1, 1}), 1e-5);

        // Relu
        const E = try D.relu(allocator);
        defer free(allocator, E);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), E.get(&.{0, 0}), 1e-5);

        // SoftmaxCrossEntropy
        const loss = try E.softmaxCrossEntropy(&[2]u8{ 0, 1 }, allocator);
        defer free(allocator, loss);
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try E.reshape(&.{1, 4}, allocator);
        defer free(allocator, F);
        try std.testing.expectEqualSlices(usize, &.{1, 4}, F.shape.dims[0..F.shape.len]);

        // Transpose
        const G = try F.transpose(0, 1, allocator);
        defer free(allocator, G);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }

    // Graph Mode Test
    {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const A = try graph.array(&.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 }, true);
        const B = try graph.array(&.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true);

        // Matmul
        const C = try graph.matmul(A, B);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);

        // AddBias
        const bias = try graph.array(&.{1, 2}, &[_]f32{ 0.5, 1.0 }, true);
        const D = try graph.addBias(C, bias);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);

        // Relu
        const E = try graph.relu(D);

        // SoftmaxCrossEntropy
        const loss = try graph.softmaxCrossEntropy(E, &[2]u8{ 0, 1 });
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try graph.reshape(E, &.{1, 4});

        // Transpose
        const G = try graph.transpose(F, 0, 1);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }
}

test "Tensor argmax and max reductions" {
    const allocator = std.testing.allocator;

    const A = try array(allocator, &.{2, 3}, &[_]f32{ 1.0, 5.0, 3.0, 9.0, 2.0, 6.0 });
    defer free(allocator, A);

    // Test argmax along dim 1
    const idx1 = try A.argmax(1, allocator);
    defer free(allocator, idx1);
    try std.testing.expectEqual(@as(f32, 1.0), idx1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx1.get(&.{1, 0}));

    // Test max along dim 1
    const val1 = try A.max(1, allocator);
    defer free(allocator, val1);
    try std.testing.expectEqual(@as(f32, 5.0), val1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 9.0), val1.get(&.{1, 0}));

    // Test argmax along dim 0
    const idx0 = try A.argmax(0, allocator);
    defer free(allocator, idx0);
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 2}));

    // Test max along dim 0
    const val0 = try A.max(0, allocator);
    defer free(allocator, val0);
    try std.testing.expectEqual(@as(f32, 9.0), val0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), val0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), val0.get(&.{0, 2}));
}

test "Tensor MSE loss forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const y_pred = try graph.array(&.{2, 1}, &[_]f32{ 1.5, 2.5 }, true);
    const y_true = try graph.array(&.{2, 1}, &[_]f32{ 1.0, 3.0 }, false);

    const loss = try graph.mseLoss(y_pred, y_true);
    // loss = 0.5 * ((1.5 - 1.0)^2 + (2.5 - 3.0)^2) = 0.5 * (0.25 + 0.25) = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), loss.data[0], 1e-5);

    try graph.backward(loss);

    // grad of y_pred = 2/N * (y_pred - y_true) = 2/2 * (y_pred - y_true) = y_pred - y_true
    // dy_pred_0 = 1.5 - 1.0 = 0.5
    // dy_pred_1 = 2.5 - 3.0 = -0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y_pred.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), y_pred.grad[1], 1e-5);
}

test "Tensor mulScalar and add autograd" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);

    // C = A.mulScalar(2.0)
    const C = try graph.mulScalar(A, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), C.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), C.get(&.{1, 1}), 1e-5);

    // D = C + B
    const D = try graph.add(C, B);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2.0 + 5.0 = 7.0
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 8.0 + 8.0 = 16.0

    // E = D.addScalar(10.0)
    const E = try graph.addScalar(D, 10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 17.0), E.get(&.{0, 0}), 1e-5); // 7.0 + 10.0 = 17.0
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), E.get(&.{1, 1}), 1e-5); // 16.0 + 10.0 = 26.0

    // Set gradients of E to 1.0 to backpropagate
    for (E.grad) |*g| {
        g.* = 1.0;
    }

    try graph.backward(E);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dB = 1 => B.grad = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[3], 1e-5);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dC = 1
    // Since C = A * 2, dC/dA = 2
    // By chain rule, dE/dA = 1 * 1 * 2 = 2.0 => A.grad = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[3], 1e-5);
}

test "Tensor static graph forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // 1. Build the static graph template once
    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);
    const C = try graph.mulScalar(A, 2.0);
    const D = try graph.add(C, B);

    // 2. First Run: set inputs
    A.data[0] = 1.0; A.data[1] = 2.0; A.data[2] = 3.0; A.data[3] = 4.0;
    B.data[0] = 5.0; B.data[1] = 6.0; B.data[2] = 7.0; B.data[3] = 8.0;

    // Execute forward pass
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2*1 + 5 = 7
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 2*4 + 8 = 16

    // Execute backward pass
    graph.zeroGrad(); // Clear all gradients in the graph!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);

    // 3. Second Run: change input data
    A.data[0] = 10.0; A.data[1] = 20.0; A.data[2] = 30.0; A.data[3] = 40.0;
    B.data[0] = 100.0; B.data[1] = 200.0; B.data[2] = 300.0; B.data[3] = 400.0;

    // Recompute forward pass on the exact same graph structure!
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), D.get(&.{0, 0}), 1e-5); // 2*10 + 100 = 120
    try std.testing.expectApproxEqAbs(@as(f32, 480.0), D.get(&.{1, 1}), 1e-5); // 2*40 + 400 = 480

    // Recompute backward pass
    graph.zeroGrad(); // Clear gradients again!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
}

test "Softmax forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Input shape [2, 3]
    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        1.0, 1.0, 1.0,
    }, true);

    const Y = try graph.softmax(X);

    try graph.forward();

    // Check forward
    // Row 0: exp(1), exp(2), exp(3) -> sum = 2.718 + 7.389 + 20.085 = 30.192
    // exp(1)/sum = 0.0900, exp(2)/sum = 0.2447, exp(3)/sum = 0.6652
    try std.testing.expectApproxEqAbs(@as(f32, 0.09003057), Y.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24472847), Y.get(&.{0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66524096), Y.get(&.{0, 2}), 1e-5);
    // Row 1: exp(1), exp(1), exp(1) -> 1/3, 1/3, 1/3
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333333), Y.get(&.{1, 0}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0); // dL/dY = 1.0
    // dX_i = Y_i * (dY_i - sum_j dY_j Y_j)
    // Since dY_j = 1.0, sum_j dY_j Y_j = sum_j Y_j = 1.0 (since softmax sums to 1)
    // So dX_i = Y_i * (1.0 - 1.0) = 0.0
    try graph.backward(Y);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[5], 1e-5);

    // Try another grad
    graph.zeroGrad();
    Y.grad[0] = 1.0;
    Y.grad[1] = 0.0;
    Y.grad[2] = 0.0;
    // Row 0: sum_dy_y = 1.0 * Y_0 = Y_0
    // dX_0 = Y_0 * (1.0 - Y_0) = Y_0 * (1 - Y_0)
    // dX_1 = Y_1 * (0.0 - Y_0) = - Y_1 * Y_0
    // dX_2 = Y_2 * (0.0 - Y_0) = - Y_2 * Y_0
    try graph.backward(Y);
    const y0 = Y.get(&.{0, 0});
    const y1 = Y.get(&.{0, 1});
    try std.testing.expectApproxEqAbs(y0 * (1.0 - y0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(-y1 * y0, X.grad[1], 1e-5);
}

test "RMSNorm forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }, true);
    const G = try graph.array(&.{3}, &[_]f32{ 1.0, 2.0, 3.0 }, true);

    const Y = try graph.rmsNorm(X, G, 1e-5);

    try graph.forward();

    // Row 0: mean(x^2) = (1+4+9)/3 = 14/3 = 4.666666
    // rms = sqrt(4.666666) = 2.1602468
    // Y_0 = 1 / rms * 1 = 0.46291
    // Y_1 = 2 / rms * 2 = 1.85164
    // Y_2 = 3 / rms * 3 = 4.16619
    try std.testing.expectApproxEqAbs(@as(f32, 0.46291), Y.get(&.{0, 0}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.85164), Y.get(&.{0, 1}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.16619), Y.get(&.{0, 2}), 1e-4);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // We can verify gradients numerically or just check they are non-zero and reasonable.
    // Let's verify G.grad: dG_j = sum_i (dY_i * X_i * scale)
    // Row 0 scale = 1/2.1602468 = 0.46291
    // Row 1: mean(x^2) = (16+25+36)/3 = 77/3 = 25.6666
    // Row 1 scale = 1/sqrt(25.6666) = 1/5.066228 = 0.197385
    // dG_0 = 1.0 * 1.0 * 0.46291 + 1.0 * 4.0 * 0.197385 = 0.46291 + 0.78954 = 1.25245
    try std.testing.expectApproxEqAbs(@as(f32, 1.25245), G.grad[0], 1e-4);
}

test "Embedding forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const W = try graph.array(&.{3, 4}, &[_]f32{
        0.1, 0.2, 0.3, 0.4,
        1.1, 1.2, 1.3, 1.4,
        2.1, 2.2, 2.3, 2.4,
    }, true);

    const X = try graph.array(&.{2, 2}, &[_]f32{
        0.0, 2.0,
        1.0, 0.0,
    }, false);

    const Y = try graph.embedding(W, X);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), Y.get(&.{0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.3), Y.get(&.{0, 1, 2}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), Y.get(&.{1, 0, 3}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // W.grad should accumulate gradients
    // X has:
    // (0,0) -> 0.0
    // (0,1) -> 2.0
    // (1,0) -> 1.0
    // (1,1) -> 0.0
    // So row 0 of W is selected twice, row 1 once, row 2 once.
    // Since dY is all 1.0, W.grad row 0 should be 2.0, row 1 should be 1.0, row 2 should be 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), W.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[8], 1e-5);
}

test "BatchMatMul forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Shape [2, 2, 2, 3]
    const A = try graph.array(&.{2, 2, 2, 3}, &[_]f32{
        // batch 0, head 0
        1, 2, 3,
        4, 5, 6,
        // batch 0, head 1
        1, 1, 1,
        2, 2, 2,
        // batch 1, head 0
        0, 1, 0,
        1, 0, 1,
        // batch 1, head 1
        2, 0, 2,
        0, 2, 0,
    }, true);

    // Shape [2, 2, 3, 2]
    const B = try graph.array(&.{2, 2, 3, 2}, &[_]f32{
        // batch 0, head 0
        1, 0,
        0, 1,
        1, 1,
        // batch 0, head 1
        2, 2,
        2, 2,
        2, 2,
        // batch 1, head 0
        1, 2,
        3, 4,
        5, 6,
        // batch 1, head 1
        1, 1,
        1, 1,
        1, 1,
    }, true);

    const C = try graph.batchMatMul(A, B);

    try graph.forward();

    // Check forward
    // Batch 0, Head 0:
    // [1, 2, 3]   [1, 0]   [4, 5]
    // [4, 5, 6] * [0, 1] = [10, 11]
    //             [1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.get(&.{0, 0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), C.get(&.{0, 0, 0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), C.get(&.{0, 0, 1, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), C.get(&.{0, 0, 1, 1}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // We can verify some gradients.
    // dA = dC * B^T
    // For Batch 0, Head 0:
    // dC_slice = [1, 1]
    //            [1, 1]
    // B_slice^T = [1, 0, 1]
    //             [0, 1, 1]
    // dA_slice = dC_slice * B_slice^T = [1, 1, 2]
    //                                   [1, 1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[2], 1e-5);
}

test "GELU forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try graph.gelu(A);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841345), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.954500), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients
    try std.testing.expectApproxEqAbs(@as(f32, -0.083316), A.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), A.grad[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.083316), A.grad[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.085232), A.grad[3], 1e-4);
}

test "Sigmoid forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try graph.sigmoid(A);

    try graph.forward();

    // Check forward: sigmoid(x) = 1 / (1 + exp(-x))
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, 1.0)))), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -1.0)))), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -2.0)))), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients: grad = C * (1 - C)
    for (A.grad, C.data) |g_val, c_val| {
        try std.testing.expectApproxEqAbs(c_val * (1.0 - c_val), g_val, 1e-5);
    }
}

test "SigmoidCrossEntropy forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const logits = try graph.tensorNDWithData(&.{3}, &.{ -1.0, 0.0, 2.0 }, true);
    const targets = try graph.tensorNDWithData(&.{3}, &.{ 0.0, 1.0, 1.0 }, false);
    const loss = try graph.sigmoidCrossEntropy(logits, targets);

    try graph.forward();

    // Check forward
    // x = -1, y = 0 -> loss = max(-1, 0) - 0 + log(1 + exp(-1)) = log(1 + e^-1) = log(1.367879) = 0.31326168
    // x = 0, y = 1 -> loss = max(0, 0) - 0 + log(1 + exp(0)) = log(2) = 0.69314718
    // x = 2, y = 1 -> loss = max(2, 0) - 2 + log(1 + exp(-2)) = log(1 + e^-2) = log(1.135335) = 0.126928
    // mean loss = (0.31326168 + 0.69314718 + 0.126928) / 3 = 1.13333686 / 3 = 0.37777895
    try std.testing.expectApproxEqAbs(@as(f32, 0.37777895), loss.data[0], 1e-5);

    // Backward
    graph.zeroGrad();
    loss.grad[0] = 1.0;
    try graph.backward(loss);

    // Check gradients:
    // grad = 1/3 * (sig(x) - y)
    // x = -1, y = 0 -> grad = 1/3 * (1/(1+e) - 0) = 1/3 * 0.268941 = 0.089647
    // x = 0, y = 1 -> grad = 1/3 * (0.5 - 1) = -1/6 = -0.166667
    // x = 2, y = 1 -> grad = 1/3 * (1/(1+e^-2) - 1) = 1/3 * (0.880797 - 1) = -0.039734
    try std.testing.expectApproxEqAbs(@as(f32, 0.089647), logits.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.166667), logits.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.039734), logits.grad[2], 1e-5);
}

test "RoPE forward, graph replay, and autograd backward for interleaved and split_half modes" {
    const allocator = std.testing.allocator;

    const modes = [_]tensor.RopeOptions.Mode{ .interleaved, .split_half };
    for (modes) |mode| {
        // 1. 验证 requires_grad 继承与梯度流传递
        {
            var graph = Graph.init(allocator);
            defer graph.deinit();

            const x = try graph.tensorNDWithData(&.{ 1, 2, 4 }, &.{
                1.0, 2.0, 3.0, 4.0,
                0.5, 1.5, -1.0, 2.5,
            }, true);
            const y = try graph.rope(x, 0, .{
                .mode = mode,
                .base = 10000.0,
                .partial_rotary_factor = 1.0,
            });

            try std.testing.expect(y.requires_grad);

            // 设置输出梯度 dY
            const dy_vals = [_]f32{ 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8 };
            @memcpy(y.grad, &dy_vals);

            try graph.backwardWithGrad(y);

            // 有限差分数值梯度检验 (Finite Difference Gradient Check)
            const eps: f32 = 1e-3;
            for (0..x.data.len) |idx| {
                const orig_val = x.data[idx];

                // f(x + eps)
                x.data[idx] = orig_val + eps;
                const y_pos = try x.rope(0, .{ .mode = mode, .base = 10000.0, .partial_rotary_factor = 1.0 }, allocator);
                defer tensor.free(allocator, y_pos);
                var loss_pos: f32 = 0.0;
                for (y_pos.data, dy_vals) |yp, dy| loss_pos += yp * dy;

                // f(x - eps)
                x.data[idx] = orig_val - eps;
                const y_neg = try x.rope(0, .{ .mode = mode, .base = 10000.0, .partial_rotary_factor = 1.0 }, allocator);
                defer tensor.free(allocator, y_neg);
                var loss_neg: f32 = 0.0;
                for (y_neg.data, dy_vals) |yn, dy| loss_neg += yn * dy;

                x.data[idx] = orig_val;

                const num_grad = (loss_pos - loss_neg) / (2.0 * eps);
                const ana_grad = x.grad[idx];
                try std.testing.expectApproxEqAbs(num_grad, ana_grad, 1e-3);
            }
        }

        // 2. 验证静态图重演 (Graph Replay: graph.forward()) 精度与数值一致性
        {
            var graph = Graph.init(allocator);
            defer graph.deinit();

            const x = try graph.tensorNDWithData(&.{ 1, 2, 4 }, &.{
                1.0, 2.0, 3.0, 4.0,
                0.5, 1.5, -1.0, 2.5,
            }, false);
            const y = try graph.rope(x, 1, .{
                .mode = mode,
                .base = 10000.0,
                .partial_rotary_factor = 1.0,
            });

            // 保存首次 eager 前向计算的数值结果
            const eager_res = try allocator.alloc(f32, y.data.len);
            defer allocator.free(eager_res);
            @memcpy(eager_res, y.data);

            // 修改输入张量数值并执行静态图重演
            const new_input = [_]f32{
                2.0, -1.0, 0.5, 3.0,
                -2.0, 1.0, 4.0, -0.5,
            };
            @memcpy(x.data, &new_input);
            try graph.forward();

            // 计算期望的参考输出
            const expected_y = try x.rope(1, .{
                .mode = mode,
                .base = 10000.0,
                .partial_rotary_factor = 1.0,
            }, allocator);
            defer tensor.free(allocator, expected_y);

            // 验证图重演更新后的结果与期望相符，且由于输入已改变，数值不应等于旧的 eager_res
            for (y.data, expected_y.data) |act, exp| {
                try std.testing.expectApproxEqAbs(exp, act, 1e-5);
            }
            try std.testing.expect(!std.mem.eql(f32, eager_res, y.data));
        }
    }

    // 3. 验证带 partial_rotary_factor (Proportional RoPE) 的反向传播与图重演
    {
        var graph = Graph.init(allocator);
        defer graph.deinit();

        // 8 维输入，half = 4，factor = 0.5 -> 仅前 2 个角度 (4 维) 旋转，后 4 维直通
        const x = try graph.tensorNDWithData(&.{ 1, 1, 8 }, &.{
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        }, true);
        const y = try graph.rope(x, 2, .{
            .mode = .split_half,
            .base = 10000.0,
            .partial_rotary_factor = 0.5,
        });

        try std.testing.expect(y.requires_grad);

        const dy_vals = [_]f32{ 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9 };
        @memcpy(y.grad, &dy_vals);

        try graph.backwardWithGrad(y);

        // 有限差分检验
        const eps: f32 = 1e-3;
        for (0..x.data.len) |idx| {
            const orig_val = x.data[idx];

            x.data[idx] = orig_val + eps;
            const y_pos = try x.rope(2, .{ .mode = .split_half, .base = 10000.0, .partial_rotary_factor = 0.5 }, allocator);
            defer tensor.free(allocator, y_pos);
            var loss_pos: f32 = 0.0;
            for (y_pos.data, dy_vals) |yp, dy| loss_pos += yp * dy;

            x.data[idx] = orig_val - eps;
            const y_neg = try x.rope(2, .{ .mode = .split_half, .base = 10000.0, .partial_rotary_factor = 0.5 }, allocator);
            defer tensor.free(allocator, y_neg);
            var loss_neg: f32 = 0.0;
            for (y_neg.data, dy_vals) |yn, dy| loss_neg += yn * dy;

            x.data[idx] = orig_val;

            const num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            const ana_grad = x.grad[idx];
            try std.testing.expectApproxEqAbs(num_grad, ana_grad, 1e-3);
        }
    }
}

test "Graph.getModuleFormula / setModuleFormula" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    // Verify null is returned for non-existent formula
    try std.testing.expect(graph.getModuleFormula("non.existent.path") == null);

    // Set a formula
    try graph.setModuleFormula("gpt.layers.0.attn", "A = softmax(QK^T / sqrt(d_k)) V");

    // Get the formula and verify it matches
    const formula = graph.getModuleFormula("gpt.layers.0.attn");
    try std.testing.expect(formula != null);
    try std.testing.expectEqualStrings("A = softmax(QK^T / sqrt(d_k)) V", formula.?);

    // Override the formula
    try graph.setModuleFormula("gpt.layers.0.attn", "A = V");
    const updated = graph.getModuleFormula("gpt.layers.0.attn");
    try std.testing.expect(updated != null);
    try std.testing.expectEqualStrings("A = V", updated.?);
}
