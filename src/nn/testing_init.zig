//! 测试与基准内部辅助：以全 1 样本输入建立一次前向计算图，再调用 `nn.initModel` 按计算图初始化模型参数。
//! 循环神经网络 (Recurrent Neural Network, RNN) 系列的前向签名需要额外的隐藏状态参数，由 `initRecurrent` 统一构造。
const std = @import("std");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const recurrent = @import("recurrent.zig");

/// 以形状为 `shape` 的全 1 样本输入执行 `model.forward(graph, x)`，然后按计算图初始化参数
pub fn initFromOnes(model: anytype, allocator: std.mem.Allocator, random: std.Random, shape: []const usize) !void {
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    const x = try graph.ones(shape, false);
    _ = try model.forward(&graph, x);
    try core.initModel(model, &graph, random);
}

/// 以批大小为 1、序列长度为 1 的全 1 样本输入执行循环网络前向，然后按计算图初始化参数
pub fn initRecurrent(model: anytype, allocator: std.mem.Allocator, random: std.Random) !void {
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    const T = @TypeOf(model.*);
    const x = try graph.ones(&.{ 1, model.input_dim }, false);
    const h = try graph.zeros(&.{ 1, model.hidden_dim }, false);
    if (T == recurrent.RNNCell or T == recurrent.GRUCell) {
        _ = try model.forward(&graph, x, h);
    } else if (T == recurrent.LSTMCell) {
        const c = try graph.zeros(&.{ 1, model.hidden_dim }, false);
        _ = try model.forward(&graph, x, h, c);
    } else if (T == recurrent.RNN or T == recurrent.GRU) {
        _ = try model.forward(&graph, &.{x}, null);
    } else if (T == recurrent.LSTM) {
        _ = try model.forward(&graph, &.{x}, null, null);
    } else if (T == recurrent.StackedLSTM) {
        _ = try model.forwardSequence(&graph, &.{x}, null, null);
    } else {
        @compileError("initRecurrent does not support " ++ @typeName(T));
    }
    try core.initModel(model, &graph, random);
}
