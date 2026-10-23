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
const BoolTensor = tensor.BoolTensor;
const GenericTensor = tensor.GenericTensor;
const IntTensor = tensor.IntTensor;
const mean = tensor.mean;
const slice = tensor.slice;
const sum = tensor.sum;
const where = tensor.where;

test "autodiff Graph enable_grad and setGradEnabled" {
    const allocator = std.testing.allocator;

    var graph = Graph.initDefault(allocator);
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
    var graph_nograd = Graph.initDefault(allocator);
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

    var graph = Graph.initDefault(allocator);
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
    var graph2 = Graph.initDefault(allocator);
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
    var graph3 = Graph.initDefault(allocator);
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
    var graph4 = Graph.initDefault(allocator);
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

    var graph = Graph.initDefault(allocator);
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

    var fresh_graph = Graph.initDefault(allocator);
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
        var graph = Graph.initDefault(allocator);
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
        var graph = Graph.initDefault(allocator);
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
        var graph = autodiff.Graph.initDefault(allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

    var graph = autodiff.Graph.initDefault(arena_allocator);
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

test "Graph.matmul broadcasts a 2-D right operand over leading batch dimensions" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    const a = try graph.tensorNDWithData(&.{ 2, 2, 3 }, &.{
        1, 2,  3,  4,  5,  6,
        7, 8,  9,  10, 11, 12,
    }, true);
    const w = try graph.tensorNDWithData(&.{ 3, 2 }, &.{ 1, 0, 0, 1, 1, 1 }, true);
    const c = try graph.matmul(a, w);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, c.shape.dims[0..c.shape.len]);
    // 每行 [x0, x1, x2] -> [x0 + x2, x1 + x2]
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 10, 11, 16, 17, 22, 23 }, c.data);

    // 反向传播：dA = dC W^T，dW = sum_rows(A^T dC)
    const loss = try graph.sum(c, null, false);
    try graph.backward(loss);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 }, a.grad);
    try std.testing.expectEqualSlices(f32, &.{ 22, 22, 26, 26, 30, 30 }, w.grad);

    // 前向重演：修改输入后 Graph.forward 重新计算批次输出
    @memset(a.data, 1.0);
    try graph.forward();
    for (c.data) |v| try std.testing.expectEqual(@as(f32, 2.0), v);

    // 右操作数必须是二维矩阵；内维必须一致
    const w3 = try graph.ones(&.{ 1, 3, 2 }, false);
    try std.testing.expectError(error.IncompatibleDimensions, graph.matmul(a, w3));
    const w_bad = try graph.ones(&.{ 4, 2 }, false);
    try std.testing.expectError(error.ShapeMismatch, graph.matmul(a, w_bad));
}

test "BatchMatMul forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.initDefault(arena_allocator);
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
    var graph = autodiff.Graph.initDefault(arena_allocator);
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
    var graph = autodiff.Graph.initDefault(arena_allocator);
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

test "sigmoidCrossEntropy records BceWithLogitsLoss forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.initDefault(arena_allocator);
    defer graph.deinit();

    const logits = try graph.tensorNDWithData(&.{3}, &.{ -1.0, 0.0, 2.0 }, true);
    const targets = try graph.tensorNDWithData(&.{3}, &.{ 0.0, 1.0, 1.0 }, false);
    const loss = try graph.sigmoidCrossEntropy(logits, targets);
    try std.testing.expectEqual(autodiff.OpType.BceWithLogitsLoss, autodiff.Op.creatorOf(loss).?.op_type);

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
            var graph = Graph.initDefault(allocator);
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
            var graph = Graph.initDefault(allocator);
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
        var graph = Graph.initDefault(allocator);
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
    var graph = Graph.initDefault(allocator);
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

test "Graph.split autograd backward executes once without gradient multiplication" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    // Input X of shape [2, 4] with requires_grad = true
    const x = try graph.tensor(2, 4, true);
    for (x.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i + 1));
    }

    // Split X into 2 parts along dim 1: each part is [2, 2]
    const splits = try graph.split(x, 2, 1);
    try std.testing.expectEqual(@as(usize, 2), splits.len);

    const s1 = splits[0]; // columns 0..2
    const s2 = splits[1]; // columns 2..4

    // Apply different operations to s1 and s2
    const z1 = try graph.mulScalar(s1, 2.0); // grad wrt s1 will be 2.0
    const z2 = try graph.mulScalar(s2, 3.0); // grad wrt s2 will be 3.0

    // Combine both outputs
    const z = try graph.add(z1, z2);
    const loss = try graph.sum(z, null, false);

    try graph.backward(loss);

    // Verify gradients on X:
    // Left two columns (from s1) should have grad = 2.0
    // Right two columns (from s2) should have grad = 3.0
    for (0..2) |r| {
        for (0..2) |c| {
            try std.testing.expectApproxEqAbs(@as(f32, 2.0), x.grad[r * 4 + c], 1e-5);
        }
        for (2..4) |c| {
            try std.testing.expectApproxEqAbs(@as(f32, 3.0), x.grad[r * 4 + c], 1e-5);
        }
    }
}

test "Strided non-contiguous views in binary ops, reshape, and autograd backward" {
    const allocator = std.testing.allocator;

    // 1. Eager binary op on equal-shape non-contiguous transposeView & slice
    {
        const mat = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
        defer free(allocator, mat);

        // mat^T view has shape [2, 2], strides [1, 2], logical [[1, 3], [2, 4]]
        const mat_t = try mat.transposeView(0, 1, allocator);
        defer free(allocator, mat_t);
        try std.testing.expect(!mat_t.isContiguous());

        // sum_mat = mat + mat_t => [[2, 5], [5, 8]]
        const sum_mat = try mat.add(mat_t, allocator);
        defer free(allocator, sum_mat);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), sum_mat.get(&.{ 0, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), sum_mat.get(&.{ 0, 1 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), sum_mat.get(&.{ 1, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 8.0), sum_mat.get(&.{ 1, 1 }), 1e-5);

        // Reshape on non-contiguous mat_t to [4] => [1, 3, 2, 4]
        const flat_t = try mat_t.reshape(&.{4}, allocator);
        defer free(allocator, flat_t);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 3.0, 2.0, 4.0 }, flat_t.data);
    }

    // 2. Graph reshape and binary ops backward on non-contiguous views
    {
        var graph = autodiff.Graph.initDefault(allocator);
        defer graph.deinit();

        const A = try graph.array(&.{ 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
        const A_t = try A.transposeView(0, 1, graph.arena.allocator());

        const B = try graph.array(&.{ 2, 2 }, &[_]f32{ 10.0, 20.0, 30.0, 40.0 }, true);
        // C = A_t * B => [[1*10, 3*20], [2*30, 4*40]] = [[10, 60], [60, 160]]
        const C = try graph.mul(A_t, B);
        try std.testing.expectApproxEqAbs(@as(f32, 10.0), C.get(&.{ 0, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 60.0), C.get(&.{ 0, 1 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 60.0), C.get(&.{ 1, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 160.0), C.get(&.{ 1, 1 }), 1e-5);

        // Also test Graph.reshape on non-contiguous A_t
        const R = try graph.reshape(A_t, &.{4});
        try std.testing.expect(!R.is_view);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 3.0, 2.0, 4.0 }, R.data);

        // Contiguous reshape sets is_view = true
        const C_flat = try graph.reshape(C, &.{4});
        try std.testing.expect(C_flat.is_view);

        const total = try graph.add(C_flat, R);
        @memset(total.grad, 1.0);
        try graph.backward(total);

        // d(total)/dA[0,1] (value 2.0): appears at A_t[1,0] (mul by B[1,0]=30) + R[2] (1.0) => 31.0
        // d(total)/dA[1,0] (value 3.0): appears at A_t[0,1] (mul by B[0,1]=20) + R[1] (1.0) => 21.0
        try std.testing.expectApproxEqAbs(@as(f32, 11.0), A.getGrad(&.{ 0, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 31.0), A.getGrad(&.{ 0, 1 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 21.0), A.getGrad(&.{ 1, 0 }), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 41.0), A.getGrad(&.{ 1, 1 }), 1e-5);
    }
}

test "GenericTensor core ops, BoolTensor where/maskedFill, and integer Embedding interoperability" {
    const allocator = std.testing.allocator;

    // 1. GenericTensor(i32) arithmetic, comparison, reshape, transposeView, sum, mean
    const a_int = try IntTensor.fromSlice(allocator, &.{ 2, 2 }, &[_]i32{ 1, 2, 3, 4 });
    defer a_int.deinit(allocator);
    const b_int = try IntTensor.fromSlice(allocator, &.{ 1, 2 }, &[_]i32{ 10, 20 });
    defer b_int.deinit(allocator);

    const sum_int = try a_int.add(b_int, allocator);
    defer sum_int.deinit(allocator);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 11, 22, 13, 24 }, sum_int.data);

    const gt_mask = try a_int.gt(b_int, allocator);
    defer gt_mask.deinit(allocator);
    try std.testing.expect(!gt_mask.any());

    const lt_mask = try a_int.lt(b_int, allocator);
    defer lt_mask.deinit(allocator);
    try std.testing.expect(lt_mask.all());

    const red_sum = try a_int.sum(0, false, allocator);
    defer red_sum.deinit(allocator);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 4, 6 }, red_sum.data);

    const red_mean = try a_int.mean(null, false, allocator);
    defer red_mean.deinit(allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), red_mean.data[0], 1e-5);

    // 2. Tensor.where and Tensor.maskedFill with BoolTensor
    const x_t = try array(allocator, &.{ 2, 2 }, &[_]f32{ -2.0, 3.0, -1.0, 5.0 });
    defer free(allocator, x_t);
    const y_t = try zeros(allocator, &.{ 2, 2 });
    defer free(allocator, y_t);

    const pos_mask = try x_t.gtScalar(0.0, allocator);
    defer pos_mask.deinit(allocator);
    try std.testing.expectEqualSlices(bool, &[_]bool{ false, true, false, true }, pos_mask.data);

    const w_out = try Tensor.where(pos_mask, x_t, y_t, allocator);
    defer free(allocator, w_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 3.0, 0.0, 5.0 }, w_out.data);

    const neg_mask = try x_t.ltScalar(0.0, allocator);
    defer neg_mask.deinit(allocator);
    const mf_out = try x_t.maskedFill(neg_mask, -99.0, allocator);
    defer free(allocator, mf_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -99.0, 3.0, -99.0, 5.0 }, mf_out.data);

    // 3. Embedding with IntTensor and integer slice in both eager and autograd modes
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const w_emb = try graph.array(&.{ 4, 2 }, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    }, true);

    const idx_int = try IntTensor.fromSlice(allocator, &.{ 1, 3 }, &[_]i32{ 3, 0, 3 });
    defer idx_int.deinit(allocator);

    const emb_out = try graph.embedding(w_emb, idx_int);
    try std.testing.expectEqual(@as(usize, 3), emb_out.shape.len);
    try std.testing.expectEqual(@as(usize, 1), emb_out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), emb_out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 2), emb_out.shape.dims[2]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7.0, 8.0, 1.0, 2.0, 7.0, 8.0 }, emb_out.data);

    @memset(emb_out.grad, 1.0);
    try graph.backwardWithGrad(emb_out);
    // Token 3 was looked up twice -> grad is 2.0; Token 0 once -> grad is 1.0
    try std.testing.expectEqualSlices(f32, &[_]f32{
        1.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
        2.0, 2.0,
    }, w_emb.grad);

    // Direct 1D integer slice lookup
    const slice_out = try w_emb.embedding(&[_]usize{ 2, 1 }, allocator);
    defer free(allocator, slice_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5.0, 6.0, 3.0, 4.0 }, slice_out.data);

    // 4. GenericTensor.transposeView verification and bounds check
    const rect_int = try IntTensor.fromSlice(allocator, &.{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 });
    defer rect_int.deinit(allocator);
    const trans_int = try rect_int.transposeView(0, 1, allocator);
    defer trans_int.deinit(allocator);
    try std.testing.expect(trans_int.is_view);
    try std.testing.expectEqual(@as(usize, 3), trans_int.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), trans_int.shape.dims[1]);
    try std.testing.expectEqual(@as(i32, 4), trans_int.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(i32, 3), trans_int.get(&.{ 2, 0 }));
    try std.testing.expectError(error.InvalidDimension, rect_int.transposeView(0, 2, allocator));
}

test "Graph.q4MatMul runs through the graph executor and matches serial execution bitwise" {
    const allocator = std.testing.allocator;
    const quant = tensor.quant;
    const in_f = 128;
    const out_f = 520;

    const raw_w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(raw_w);
    for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.29);
    const blocks = try allocator.alloc(quant.Q4Block, (in_f * out_f) / quant.Q4Block.QK);
    defer allocator.free(blocks);
    quant.quantizeSlice(blocks, raw_w);

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();

    // 单行解码输入 [1, in] 与三维预填充输入 [2, 5, in]
    const shapes = [_][]const usize{ &.{ 1, in_f }, &.{ 2, 5, in_f } };
    for (shapes) |shape| {
        var g_serial = Graph.init(allocator, .{ .enable_grad = false });
        defer g_serial.deinit();
        var g_threaded = Graph.init(allocator, .{ .enable_grad = false, .io = threaded.io(), .max_tasks = 4 });
        defer g_threaded.deinit();
        try std.testing.expect(g_threaded.executor().io != null);

        const x_s = try g_serial.zeros(shape, false);
        const x_t = try g_threaded.zeros(shape, false);
        for (x_s.data, x_t.data, 0..) |*a, *b, i| {
            const v = @cos(@as(f32, @floatFromInt(i)) * 0.17);
            a.* = v;
            b.* = v;
        }

        const y_s = try g_serial.q4MatMul(x_s, blocks, in_f, out_f);
        const y_t = try g_threaded.q4MatMul(x_t, blocks, in_f, out_f);
        try std.testing.expectEqual(shape.len, y_t.shape.len);
        try std.testing.expectEqual(out_f, y_t.shape.dims[shape.len - 1]);
        try std.testing.expectEqualSlices(f32, y_s.data, y_t.data);
    }
}

test "Graph MLX GPU accelerated matmul, rmsNorm, and scaledDotProductAttention" {
    const backend = @import("../backend.zig");
    if (!backend.mlx.has_mlx) return;

    const allocator = std.testing.allocator;

    var g_cpu = Graph.init(allocator, .{ .enable_grad = false, .device = .cpu });
    defer g_cpu.deinit();

    var g_gpu = Graph.init(allocator, .{ .enable_grad = false, .device = .gpu });
    defer g_gpu.deinit();

    // 1. 验证矩阵乘法 GPU 卸载 (MatMul)
    const a_cpu = try g_cpu.tensor(2, 3, false);
    const a_gpu = try g_gpu.tensor(2, 3, false);
    for (a_cpu.data, a_gpu.data, 0..) |*c, *g, i| {
        const val = @as(f32, @floatFromInt(i + 1));
        c.* = val;
        g.* = val;
    }
    const b_cpu = try g_cpu.tensor(3, 2, false);
    const b_gpu = try g_gpu.tensor(3, 2, false);
    for (b_cpu.data, b_gpu.data, 0..) |*c, *g, i| {
        const val = @as(f32, @floatFromInt((i + 1) * 2));
        c.* = val;
        g.* = val;
    }

    const c_cpu = try g_cpu.matmul(a_cpu, b_cpu);
    const c_gpu = try g_gpu.matmul(a_gpu, b_gpu);
    for (c_cpu.data, c_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-4);
    }

    // 2. 验证均方根层归一化 GPU 卸载 (RMSNorm)
    const x_cpu = try g_cpu.tensor(2, 4, false);
    const x_gpu = try g_gpu.tensor(2, 4, false);
    for (x_cpu.data, x_gpu.data, 0..) |*c, *g, i| {
        const val = @as(f32, @floatFromInt(i + 1)) * 0.5;
        c.* = val;
        g.* = val;
    }
    const w_cpu = try g_cpu.tensor(1, 4, false);
    const w_gpu = try g_gpu.tensor(1, 4, false);
    @memset(w_cpu.data, 1.0);
    @memset(w_gpu.data, 1.0);

    const norm_cpu = try g_cpu.rmsNorm(x_cpu, w_cpu, 1e-5);
    const norm_gpu = try g_gpu.rmsNorm(x_gpu, w_gpu, 1e-5);
    for (norm_cpu.data, norm_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-4);
    }

    // 3. 验证缩放点积注意力 GPU 卸载 (FlashAttention SDPA)
    const B: usize = 1;
    const H: usize = 2;
    const T: usize = 16;
    const D: usize = 64;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    const q_cpu = try g_cpu.tensorND(&.{ B, H, T, D }, false);
    const q_gpu = try g_gpu.tensorND(&.{ B, H, T, D }, false);
    const k_cpu = try g_cpu.tensorND(&.{ B, H, T, D }, false);
    const k_gpu = try g_gpu.tensorND(&.{ B, H, T, D }, false);
    const v_cpu = try g_cpu.tensorND(&.{ B, H, T, D }, false);
    const v_gpu = try g_gpu.tensorND(&.{ B, H, T, D }, false);

    for (0..B * H * T * D) |i| {
        const val = @sin(@as(f32, @floatFromInt(i)) * 0.2);
        q_cpu.data[i] = val;
        q_gpu.data[i] = val;
        k_cpu.data[i] = val * 0.8;
        k_gpu.data[i] = val * 0.8;
        v_cpu.data[i] = val * 1.2;
        v_gpu.data[i] = val * 1.2;
    }

    const sdpa_cpu = try g_cpu.scaledDotProductAttention(q_cpu, k_cpu, v_cpu, scale, true);
    const sdpa_gpu = try g_gpu.scaledDotProductAttention(q_gpu, k_gpu, v_gpu, scale, true);

    for (sdpa_cpu.data, sdpa_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
    }

    // 4. 验证旋转位置编码 GPU 卸载 (RoPE)
    const rope_cpu = try g_cpu.tensorND(&.{ B, H, T, D }, false);
    const rope_gpu = try g_gpu.tensorND(&.{ B, H, T, D }, false);
    for (rope_cpu.data, rope_gpu.data, 0..) |*c, *g, i| {
        const val = @sin(@as(f32, @floatFromInt(i)) * 0.15);
        c.* = val;
        g.* = val;
    }

    const r_out_cpu = try g_cpu.rope(rope_cpu, 0, .{ .mode = .split_half, .partial_rotary_factor = 1.0, .base = 10000.0 });
    const r_out_gpu = try g_gpu.rope(rope_gpu, 0, .{ .mode = .split_half, .partial_rotary_factor = 1.0, .base = 10000.0 });
    for (r_out_cpu.data, r_out_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
    }

    const r_part_cpu = try g_cpu.rope(rope_cpu, 0, .{ .mode = .split_half, .partial_rotary_factor = 0.5, .base = 10000.0 });
    const r_part_gpu = try g_gpu.rope(rope_gpu, 0, .{ .mode = .split_half, .partial_rotary_factor = 0.5, .base = 10000.0 });
    for (r_part_cpu.data, r_part_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
    }

    const r_intl_cpu = try g_cpu.rope(rope_cpu, 0, .{ .mode = .interleaved, .partial_rotary_factor = 1.0, .base = 10000.0 });
    const r_intl_gpu = try g_gpu.rope(rope_gpu, 0, .{ .mode = .interleaved, .partial_rotary_factor = 1.0, .base = 10000.0 });
    for (r_intl_cpu.data, r_intl_gpu.data) |c_val, g_val| {
        try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
    }
}

test "Graph.backward called repeatedly accumulates parameter gradients without re-propagating stale intermediate gradients" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    // 共享主干 h = x * w，两个损失头 l1 = sum(2h)、l2 = sum(3h)
    const x = try graph.array(&.{ 1, 2 }, &.{ 1.0, 2.0 }, false);
    const w = try graph.array(&.{ 2, 1 }, &.{ 0.5, -0.5 }, true);
    const h = try graph.matmul(x, w);
    const l1 = try graph.sum(try graph.mulScalar(h, 2.0), null, false);
    const l2 = try graph.sum(try graph.mulScalar(h, 3.0), null, false);

    try graph.backward(l1);
    try graph.backward(l2);

    // dL1/dw + dL2/dw = (2 + 3) * x^T = [5, 10]
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), w.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), w.grad[1], 1e-5);

    // 同一损失重复反向两次：参数梯度恰好翻倍
    graph.zeroGrad();
    try graph.backward(l1);
    try graph.backward(l1);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), w.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), w.grad[1], 1e-5);
}

test "Graph.backward skips frozen subgraphs that do not require gradients" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    // 冻结的均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)：输入与缩放参数都不需要梯度
    const x = try graph.array(&.{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, false);
    const gamma = try graph.array(&.{4}, &.{ 1, 1, 1, 1 }, false);
    const y = try graph.rmsNorm(x, gamma, 1e-6);
    try std.testing.expect(!y.requires_grad);

    const w = try graph.zeros(&.{ 4, 3 }, true);
    @memset(w.data, 0.1);
    const logits = try graph.matmul(y, w);
    const loss = try graph.softmaxCrossEntropy(logits, &[_]u8{ 0, 2 });
    try graph.backward(loss);

    var grad_norm: f32 = 0.0;
    for (w.grad) |g| grad_norm += @abs(g);
    try std.testing.expect(grad_norm > 1e-4);
}

test "Graph.backward rejects a root tensor that does not require gradients" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.array(&.{ 1, 2 }, &.{ 1.0, 2.0 }, false);
    const s = try graph.sum(x, null, false);
    try std.testing.expectError(error.TensorDoesNotRequireGrad, graph.backward(s));
    try std.testing.expectError(error.TensorDoesNotRequireGrad, graph.backwardWithGrad(s));
}

test "Graph.backward on a persistent graph does not grow the arena" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    var h = try graph.zeros(&.{ 4, 16 }, true);
    for (0..64) |_| h = try graph.tanh(h);
    const loss = try graph.mean(h, null, false);

    try graph.backward(loss);
    const capacity_before = graph.arena.queryCapacity();
    for (0..32) |_| {
        graph.zeroGrad();
        try graph.forward();
        try graph.backward(loss);
    }
    try std.testing.expectEqual(capacity_before, graph.arena.queryCapacity());
}

test "Graph.forward replays training-mode batch normalization with fresh batch statistics" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.array(&.{ 4, 1 }, &.{ 1, 2, 3, 4 }, true);
    const gamma = try graph.array(&.{1}, &.{1.5}, true);
    const beta = try graph.array(&.{1}, &.{0.5}, true);
    const rm = try graph.array(&.{1}, &.{0}, false);
    const rv = try graph.array(&.{1}, &.{1}, false);
    const y = try graph.batchNorm1d(x, gamma, beta, rm, rv, 1e-5, 0.1, true);
    const w = try graph.array(&.{ 4, 1 }, &.{ 1.0, -2.0, 0.5, 3.0 }, false);
    const loss = try graph.sum(try graph.mul(y, w), null, false);

    // 输入整体平移并缩放后重演前向：训练模式输出只取决于批内标准化结果
    for (x.data, 0..) |*v, i| v.* = 100.0 + 2.0 * @as(f32, @floatFromInt(i));
    const running_mean_before = rm.data[0];
    try graph.forward();
    graph.zeroGrad();
    try graph.backward(loss);

    var fresh = Graph.initDefault(allocator);
    defer fresh.deinit();
    const x2 = try fresh.array(&.{ 4, 1 }, x.data, true);
    const gamma2 = try fresh.array(&.{1}, &.{1.5}, true);
    const beta2 = try fresh.array(&.{1}, &.{0.5}, true);
    const rm2 = try fresh.array(&.{1}, &.{0}, false);
    const rv2 = try fresh.array(&.{1}, &.{1}, false);
    const y2 = try fresh.batchNorm1d(x2, gamma2, beta2, rm2, rv2, 1e-5, 0.1, true);
    const w2 = try fresh.array(&.{ 4, 1 }, &.{ 1.0, -2.0, 0.5, 3.0 }, false);
    const loss2 = try fresh.sum(try fresh.mul(y2, w2), null, false);
    try fresh.backward(loss2);

    for (y.data, y2.data) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
    for (x.grad, x2.grad) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
    try std.testing.expectApproxEqAbs(gamma2.grad[0], gamma.grad[0], 1e-4);
    // 重演同样按动量更新运行均值
    try std.testing.expect(rm.data[0] > running_mean_before);
}

test "softmaxCrossEntropy uses log-sum-exp for confidently wrong predictions" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();

    // 目标类别的概率 exp(-100) 在单精度下溢为 0，旧实现会把损失截断为 -log(1e-15) ≈ 34.5
    const logits = try graph.array(&.{ 2, 2 }, &.{ 100.0, 0.0, 0.0, 1.0 }, true);
    const loss = try graph.softmaxCrossEntropy(logits, &[_]u8{ 1, 1 });
    const masked = try graph.maskedCrossEntropyLoss(logits, &[_]u8{ 1, 1 }, &.{ 1.0, 1.0 });
    const row1 = @log(1.0 + @exp(@as(f32, -1.0)));
    const expected = (100.0 + row1) / 2.0;
    try std.testing.expectApproxEqAbs(expected, loss.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(expected, masked.data[0], 1e-3);

    try graph.backward(loss);
    // d/dz = (softmax(z) - onehot) / B：第一行 ≈ [0.5, -0.5]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), logits.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), logits.grad[1], 1e-5);

    // 非二维 logits 必须先 reshape
    const logits3 = try graph.zeros(&.{ 1, 2, 3 }, true);
    try std.testing.expectError(error.IncompatibleDimensions, graph.softmaxCrossEntropy(logits3, &[_]u8{0}));
}

test "Graph.q4MatMul frozen backbone trains the head above it" {
    const allocator = std.testing.allocator;
    const quant = tensor.quant;
    const in_f = 32;
    const out_f = 8;
    const raw_w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(raw_w);
    for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.29);
    const blocks = try allocator.alloc(quant.Q4Block, (in_f * out_f) / quant.Q4Block.QK);
    defer allocator.free(blocks);
    quant.quantizeSlice(blocks, raw_w);

    var graph = Graph.initDefault(allocator);
    defer graph.deinit();
    const x = try graph.zeros(&.{ 2, in_f }, false);
    for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)));
    const h = try graph.q4MatMul(x, blocks, in_f, out_f);
    try std.testing.expect(!h.requires_grad);
    const w_head = try graph.zeros(&.{ out_f, 3 }, true);
    @memset(w_head.data, 0.1);
    const logits = try graph.matmul(h, w_head);
    const loss = try graph.softmaxCrossEntropy(logits, &[_]u8{ 0, 2 });
    try graph.backward(loss);

    var grad_norm: f32 = 0.0;
    for (w_head.grad) |g| grad_norm += @abs(g);
    try std.testing.expect(grad_norm > 1e-4);
}

test "Graph.dropout rejects probabilities outside [0, 1]" {
    const allocator = std.testing.allocator;
    var graph = Graph.initDefault(allocator);
    defer graph.deinit();
    var prng = std.Random.DefaultPrng.init(1);
    const x = try graph.ones(&.{ 2, 2 }, true);
    try std.testing.expectError(error.InvalidProbability, graph.dropout(x, -0.1, prng.random()));
    try std.testing.expectError(error.InvalidProbability, graph.dropout(x, 1.5, prng.random()));
    try std.testing.expectError(error.InvalidProbability, graph.dropout(x, std.math.nan(f32), prng.random()));
}
