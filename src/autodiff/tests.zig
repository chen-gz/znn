const std = @import("std");
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



