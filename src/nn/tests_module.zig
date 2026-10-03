//! 模块协议测试：字段遍历、PyTorch 风格的参数查询 / 冻结，以及 `Module(T)` 的元组前向调用
const std = @import("std");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const testing_init = @import("testing_init.zig");
const Tensor = @import("../tensor.zig").Tensor;

test "namedParameters reports field paths matching the serialization keys" {
    const allocator = std.testing.allocator;
    var gru = try nn.GRU.init(allocator, 3, 4);
    defer nn.deinitModel(&gru, allocator);

    var named = try nn.namedParameters(&gru, allocator);
    defer named.deinit();

    const expected = [_][]const u8{
        "cell.w_ih_r.weight", "cell.w_ih_r.bias", "cell.w_hh_r.weight", "cell.w_hh_r.bias",
        "cell.w_ih_z.weight", "cell.w_ih_z.bias", "cell.w_hh_z.weight", "cell.w_hh_z.bias",
        "cell.w_ih_h.weight", "cell.w_ih_h.bias", "cell.w_hh_h.weight", "cell.w_hh_h.bias",
    };
    try std.testing.expectEqual(expected.len, named.items.len);
    for (expected, named.items) |name, p| try std.testing.expectEqualStrings(name, p.name);
    try std.testing.expect(named.items[0].tensor == gru.cell.w_ih_r.weight);

    const params = try nn.parameters(&gru, allocator);
    defer allocator.free(params);
    try std.testing.expectEqual(expected.len, params.len);

    // 3 个输入门 [3, 4] + 3 个隐状态门 [4, 4] + 6 个偏置 [1, 4]
    try std.testing.expectEqual(@as(usize, 3 * 12 + 3 * 16 + 6 * 4), nn.numParameters(&gru));
}

test "walk visits slices and arrays of sub-modules and skips comptime tuple fields" {
    const allocator = std.testing.allocator;
    const Model = struct {
        blocks: []nn.Linear,
        pair: [2]nn.Linear,
        head: nn.Sequential(@TypeOf(.{ nn.Linear.init(allocator, 1, 1) catch unreachable, nn.ReLU{} })),
    };
    const blocks = try allocator.alloc(nn.Linear, 2);
    blocks[0] = try nn.Linear.init(allocator, 2, 2);
    blocks[1] = try nn.Linear.init(allocator, 2, 2);
    var model = Model{
        .blocks = blocks,
        .pair = .{ try nn.Linear.init(allocator, 2, 3), try nn.Linear.init(allocator, 3, 2) },
        .head = nn.sequential(.{ try nn.Linear.init(allocator, 2, 1), nn.ReLU{} }),
    };
    defer nn.deinitModel(&model, allocator);

    var named = try nn.namedParameters(&model, allocator);
    defer named.deinit();
    const expected = [_][]const u8{
        "blocks.0.weight", "blocks.0.bias", "blocks.1.weight", "blocks.1.bias",
        "pair.0.weight",   "pair.0.bias",   "pair.1.weight",   "pair.1.bias",
        "head.layers.0.weight", "head.layers.0.bias",
    };
    try std.testing.expectEqual(expected.len, named.items.len);
    for (expected, named.items) |name, p| try std.testing.expectEqualStrings(name, p.name);

    nn.zeroGradModel(&model);
    nn.evalModel(&model);
}

test "setRequiresGrad freezes and unfreezes sub-modules" {
    const allocator = std.testing.allocator;
    var block = try nn.TransformerBlock.init(allocator, 8, 2);
    defer nn.deinitModel(&block, allocator);

    const total = nn.numParameters(&block);
    const attn_params = nn.numParameters(&block.attn);
    try nn.setRequiresGrad(&block.attn, allocator, false);
    try std.testing.expectEqual(total - attn_params, nn.numParameters(&block));
    try std.testing.expect(!block.attn.q_attn.weight.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), block.attn.q_attn.weight.grad.len);

    const params = try nn.parameters(&block, allocator);
    defer allocator.free(params);
    for (params) |p| try std.testing.expect(p.requires_grad);

    try nn.setRequiresGrad(&block.attn, allocator, true);
    try std.testing.expectEqual(total, nn.numParameters(&block));
    try std.testing.expectEqual(block.attn.q_attn.weight.data.len, block.attn.q_attn.weight.grad.len);
}

test "Module forwards a single input or a tuple of inputs to the wrapped model" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    var gru = nn.Module(nn.GRU).init(allocator, try nn.GRU.init(allocator, 3, 4));
    defer gru.deinit();
    try testing_init.initRecurrent(&gru.inner, allocator, random);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    const x0 = try graph.ones(&.{ 2, 3 }, false);
    const x1 = try graph.ones(&.{ 2, 3 }, false);
    const res = try gru.forward(&graph, .{ &[_]*Tensor{ x0, x1 }, null });
    try std.testing.expectEqual(@as(usize, 2), res.outputs.len);
    try std.testing.expectEqual(@as(usize, 4), res.h_n.shape.dims[1]);

    var mlp = nn.Module(nn.Linear).init(allocator, try nn.Linear.init(allocator, 3, 2));
    defer mlp.deinit();
    try mlp.initParametersWithSample(random, x0);
    const y = try mlp.forward(&graph, x0);
    try std.testing.expectEqual(@as(usize, 2), y.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3 * 2 + 2), mlp.numParameters());

    var named = try gru.namedParameters(allocator);
    defer named.deinit();
    try std.testing.expectEqualStrings("cell.w_ih_r.weight", named.items[0].name);
}
