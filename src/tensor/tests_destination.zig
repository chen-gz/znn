const std = @import("std");
const tensor = @import("../tensor.zig");
const backend = @import("../backend.zig");
const Tensor = tensor.Tensor;
const expect = std.testing.expect;
const expectError = std.testing.expectError;
const expectEqualSlices = std.testing.expectEqualSlices;

fn borrowed(data: []f32, dims: []const usize) Tensor {
    const shape = tensor.Shape.init(dims);
    return .{
        .data = data,
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
}

test "destination arithmetic broadcasts into preallocated graph outputs" {
    // All storage and tensor metadata are on the stack; no allocator is used.
    var a_data = [_]f32{ 1, 2 };
    var b_data = [_]f32{ 10, 20, 30 };
    var output: [8]f32 = @splat(99);
    var gradient: [6]f32 = @splat(7);
    const a = borrowed(&a_data, &.{ 2, 1 });
    const b = borrowed(&b_data, &.{ 1, 3 });
    var dest = borrowed(&output, &.{ 2, 3 });
    dest.grad = &gradient;
    dest.requires_grad = true;
    dest.is_graph_owned = true;
    const original = dest.data.ptr;
    for (0..100) |_| {
        try tensor.addInto(&dest, &a, &b);
        try expectEqualSlices(f32, &.{ 11, 21, 31, 12, 22, 32, 99, 99 }, &output);
        try tensor.mulInto(&dest, &a, &b);
        try expectEqualSlices(f32, &.{ 10, 20, 30, 20, 40, 60, 99, 99 }, &output);
    }
    try expect(original == dest.data.ptr);
    try expectEqualSlices(f32, &.{ 7, 7, 7, 7, 7, 7 }, &gradient);
    try expectEqualSlices(f32, &.{ 1, 2 }, &a_data);
    try expectEqualSlices(f32, &.{ 10, 20, 30 }, &b_data);
}

test "destination arithmetic supports strided inputs scalars and empty outputs" {
    var values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var scalar_data = [_]f32{2};
    var output: [6]f32 = @splat(0);
    var a = borrowed(&values, &.{ 3, 2 });
    a.strides = tensor.Shape.init(&.{ 1, 3 });
    const b = borrowed(&scalar_data, &.{});
    var dest = borrowed(&output, &.{ 3, 2 });
    try tensor.addInto(&dest, &a, &b);
    try expectEqualSlices(f32, &.{ 3, 6, 4, 7, 5, 8 }, &output);
    try tensor.mulInto(&dest, &a, &b);
    try expectEqualSlices(f32, &.{ 2, 8, 4, 10, 6, 12 }, &output);

    var scalar_out = borrowed(output[0..1], &.{});
    try tensor.addInto(&scalar_out, &b, &b);
    try std.testing.expectEqual(@as(f32, 4), output[0]);
    try tensor.mulInto(&scalar_out, &b, &b);
    try std.testing.expectEqual(@as(f32, 4), output[0]);

    const empty = borrowed(values[0..0], &.{ 2, 0, 3 });
    var empty_out = borrowed(output[0..0], &.{ 2, 0, 3 });
    const before = output;
    try tensor.addInto(&empty_out, &empty, &b);
    try tensor.mulInto(&empty_out, &empty, &b);
    try expectEqualSlices(f32, &before, &output);
}

test "destination arithmetic validates shape layout storage and devices before writing" {
    var source = [_]f32{ 1, 2, 3, 4 };
    var scalar_data = [_]f32{2};
    var output: [4]f32 = @splat(99);
    const a = borrowed(&source, &.{ 2, 2 });
    const b = borrowed(&scalar_data, &.{});
    const good = borrowed(&output, &.{ 2, 2 });
    inline for (.{ tensor.addInto, tensor.mulInto }) |op| {
        var dest = good;
        dest.shape = tensor.Shape.init(&.{4});
        dest.strides = tensor.Shape.init(&.{1});
        try expectError(error.ShapeMismatch, op(&dest, &a, &b));
        dest = good;
        dest.strides = tensor.Shape.init(&.{ 1, 2 });
        try expectError(error.NonContiguousLayout, op(&dest, &a, &b));
        dest = good;
        dest.data = output[0..3];
        try expectError(error.InvalidStorage, op(&dest, &a, &b));
        dest = good;
        const incompatible = borrowed(source[0..3], &.{3});
        try expectError(error.ShapeMismatch, op(&dest, &a, &incompatible));
        var bad = a;
        bad.shape = tensor.Shape.init(&.{ 2, 3 });
        bad.strides = tensor.Shape.init(&.{ 3, 1 });
        try expectError(error.InvalidStorage, op(&dest, &bad, &b));
        bad = a;
        bad.strides.len = 1;
        try expectError(error.StrideMismatch, op(&dest, &bad, &b));
        bad = a;
        bad.shape.len = 9;
        try expectError(error.DimensionOutOfBounds, op(&dest, &bad, &b));
        bad = a;
        bad.strides.dims[0] = std.math.maxInt(usize);
        try expectError(error.StrideMismatch, op(&dest, &bad, &b));
        bad = a;
        bad.shape.dims[0] = std.math.maxInt(usize);
        bad.strides = tensor.Shape.init(&.{ 0, 0 });
        try expectError(error.InvalidStorage, op(&dest, &bad, &b));
        bad = a;
        bad.storage.device = backend.Device.metal;
        try expectError(error.DeviceMismatch, op(&dest, &bad, &b));
        bad = a;
        bad.storage.typed_owner = .{ .mlx = undefined };
        try expectError(error.DeviceMismatch, op(&dest, &bad, &b));
        dest.storage.device = backend.Device.cuda;
        try expectError(error.DeviceMismatch, op(&dest, &a, &b));
        dest = good;
        dest.storage.typed_owner = .{ .mlx = undefined };
        try expectError(error.DeviceMismatch, op(&dest, &a, &b));
        dest = good;
        dest.is_graph_view = true;
        try expectError(error.InPlaceOpOnGraphTensor, op(&dest, &a, &b));
        try expectEqualSlices(f32, &.{ 99, 99, 99, 99 }, &output);
    }
}

test "destination arithmetic rejects exact and partial storage aliases atomically" {
    var shared = [_]f32{ 1, 2, 3, 4, 5 };
    var other_data = [_]f32{ 10, 20, 30, 40 };
    var a = borrowed(shared[0..4], &.{4});
    const b = borrowed(&other_data, &.{4});
    inline for (.{ tensor.addInto, tensor.mulInto }) |op| {
        var dest = borrowed(shared[1..5], &.{4});
        try expectError(error.AliasedOutput, op(&dest, &a, &b));
        try expectError(error.AliasedOutput, op(&dest, &b, &a));
        try expectError(error.AliasedOutput, op(&a, &a, &b));
        a.requires_grad = true;
        try expectError(error.InPlaceOpOnGraphTensor, op(&dest, &a, &b));
        a.requires_grad = false;
        dest.requires_grad = true;
        try expectError(error.InPlaceOpOnGraphTensor, op(&dest, &a, &b));
        try expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5 }, &shared);
        try expectEqualSlices(f32, &.{ 10, 20, 30, 40 }, &other_data);
    }
}

test "destination arithmetic handles zero strides adjacency and conservative overlap" {
    var shared = [_]f32{ 1, 2, 3, 4, 99, 99 };
    var scalar_data = [_]f32{2};
    const scalar = borrowed(&scalar_data, &.{});
    var strided = borrowed(shared[0..4], &.{2});
    strided.strides = tensor.Shape.init(&.{3});
    var gap = borrowed(shared[1..3], &.{2});
    inline for (.{ tensor.addInto, tensor.mulInto }) |op| {
        try expectError(error.AliasedOutput, op(&gap, &strided, &scalar));
        try expectEqualSlices(f32, &.{ 1, 2, 3, 4, 99, 99 }, &shared);
    }

    // The source span ends exactly where this independent output starts.
    var adjacent = borrowed(shared[4..6], &.{2});
    var creator_token: u8 = 0;
    adjacent.creator = @ptrCast(&creator_token);
    try tensor.addInto(&adjacent, &strided, &scalar);
    try expectEqualSlices(f32, &.{ 1, 2, 3, 4, 3, 6 }, &shared);
    try tensor.mulInto(&adjacent, &strided, &scalar);
    try expectEqualSlices(f32, &.{ 1, 2, 3, 4, 2, 8 }, &shared);

    var repeated_data = [_]f32{7};
    var repeated = borrowed(&repeated_data, &.{2});
    repeated.strides = tensor.Shape.init(&.{0});
    try tensor.addInto(&adjacent, &repeated, &scalar);
    try expectEqualSlices(f32, &.{ 9, 9 }, adjacent.data);
    try tensor.mulInto(&adjacent, &repeated, &scalar);
    try expectEqualSlices(f32, &.{ 14, 14 }, adjacent.data);
    try expectEqualSlices(f32, &.{7}, &repeated_data);
}

fn expectClose(actual: []const f32, expected: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (actual, expected) |a, e| try std.testing.expectApproxEqAbs(e, a, 0.00001);
}

test "destination neural operations match eager kernels with strided inputs" {
    var data = [_]f32{ -2, 1, 0, 3, -1, 2 };
    var x = borrowed(&data, &.{ 3, 2 });
    x.strides = tensor.Shape.init(&.{ 1, 3 });
    var weight_data = [_]f32{ 2, 99, 3 };
    var w = borrowed(&weight_data, &.{2});
    w.strides.dims[0] = 2;
    var output: [8]f32 = @splat(77);
    var dest = borrowed(&output, &.{ 3, 2 });
    const allocator = std.testing.allocator;
    inline for (.{ backend.GeluApproximation.none, backend.GeluApproximation.tanh }) |approx| {
        const eager = try x.gelu(approx, allocator);
        defer eager.deinit(allocator);
        for (0..10) |_| try tensor.geluInto(&dest, &x, approx);
        try expectClose(output[0..6], eager.data);
    }
    const soft = try x.softmax(allocator);
    defer soft.deinit(allocator);
    try tensor.softmaxInto(&dest, &x, null);
    try expectClose(output[0..6], soft.data);
    const options: tensor.RmsNormOptions = .{ .eps = 0.01, .weight_offset = 1 };
    const norm = try x.rmsNorm(&w, options, null, allocator);
    defer norm.deinit(allocator);
    try tensor.rmsNormInto(&dest, &x, &w, options);
    try expectClose(output[0..6], norm.data);
    const unweighted = try x.rmsNorm(null, options, null, allocator);
    defer unweighted.deinit(allocator);
    try tensor.rmsNormInto(&dest, &x, null, options);
    try expectClose(output[0..6], unweighted.data);
    try expectEqualSlices(f32, &.{ 77, 77 }, output[6..]);
}

test "destination matrix multiply handles strides and zero contraction" {
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 2, 3, 4, 5, 6, 7 };
    var a = borrowed(&a_data, &.{ 2, 3 });
    a.strides = tensor.Shape.init(&.{ 1, 2 });
    var b = borrowed(&b_data, &.{ 3, 2 });
    b.strides = tensor.Shape.init(&.{ 1, 3 });
    var out = [_]f32{ 99, 99, 99, 99, 99 };
    var dest = borrowed(&out, &.{ 2, 2 });
    const eager = try a.matmul(&b, std.testing.allocator);
    defer eager.deinit(std.testing.allocator);
    try tensor.matmulInto(&dest, &a, &b);
    try expectClose(out[0..4], eager.data);
    var empty_a = borrowed(&.{}, &.{ 2, 0 });
    var empty_b = borrowed(&.{}, &.{ 0, 2 });
    try tensor.matmulInto(&dest, &empty_a, &empty_b);
    try expectEqualSlices(f32, &.{ 0, 0, 0, 0, 99 }, &out);
    var no_rows = borrowed(&.{}, &.{ 0, 3 });
    var no_output = borrowed(&.{}, &.{ 0, 2 });
    try tensor.matmulInto(&no_output, &no_rows, &b);
}

test "destination softmax arbitrary axes have independent normalized groups" {
    var data = [_]f32{ 0, 0, 1, 2, 2, 4, 3, 6 };
    var x = borrowed(&data, &.{ 2, 2, 2 });
    var out: [8]f32 = undefined;
    var dest = borrowed(&out, &.{ 2, 2, 2 });
    try tensor.softmaxInto(&dest, &x, 1);
    const p: f32 = 1 / (1 + @exp(@as(f32, 1)));
    const q: f32 = 1 / (1 + @exp(@as(f32, 2)));
    try expectClose(&out, &.{ p, q, 1 - p, 1 - q, p, q, 1 - p, 1 - q });
    try tensor.softmaxInto(&dest, &x, 0);
    const r: f32 = 1 / (1 + @exp(@as(f32, 4)));
    try expectClose(&out, &.{ q, r, q, r, 1 - q, 1 - r, 1 - q, 1 - r });
    x.strides = tensor.Shape.init(&.{ 0, 0, 0 });
    try tensor.softmaxInto(&dest, &x, 0);
    try expectEqualSlices(f32, &@as([8]f32, @splat(0.5)), &out);
}

test "destination normalization distinguishes empty outer dimensions and axes" {
    var empty = borrowed(&.{}, &.{ 0, 3 });
    var dest = borrowed(&.{}, &.{ 0, 3 });
    try tensor.softmaxInto(&dest, &empty, null);
    try tensor.rmsNormInto(&dest, &empty, null, .{});
    try tensor.geluInto(&dest, &empty, .none);
    try expectError(error.DivisionByZero, tensor.softmaxInto(&dest, &empty, 0));
    empty.shape = tensor.Shape.init(&.{ 3, 0 });
    dest.shape = empty.shape;
    try expectError(error.DivisionByZero, tensor.softmaxInto(&dest, &empty, null));
    try expectError(error.DivisionByZero, tensor.rmsNormInto(&dest, &empty, null, .{}));
    var value = [_]f32{2};
    var result = [_]f32{99};
    var scalar = borrowed(&value, &.{});
    var scalar_dest = borrowed(&result, &.{});
    try expectError(error.IncompatibleDimensions, tensor.softmaxInto(&scalar_dest, &scalar, null));
    try expectError(error.IncompatibleDimensions, tensor.rmsNormInto(&scalar_dest, &scalar, null, .{}));
    try tensor.geluInto(&scalar_dest, &scalar, .none);
    try expectClose(&result, &.{backend.activation.gelu(.none, 2)});
}

const NeuralOp = enum { matmul, rms, softmax, gelu };
fn neuralCall(op: NeuralOp, dest: *Tensor, x: *Tensor, other: *Tensor) tensor.TensorError!void {
    return switch (op) {
        .matmul => tensor.matmulInto(dest, x, other),
        .rms => tensor.rmsNormInto(dest, x, other, .{}),
        .softmax => tensor.softmaxInto(dest, x, null),
        .gelu => tensor.geluInto(dest, x, .tanh),
    };
}

test "destination neural validation is atomic for layouts devices shapes and aliases" {
    var data = [_]f32{ 1, 2, 3, 4 };
    var other_data = [_]f32{ 5, 6, 7, 8 };
    inline for (std.meta.tags(NeuralOp)) |op| {
        var output: [4]f32 = @splat(99);
        var x = borrowed(&data, &.{ 2, 2 });
        var other = borrowed(&other_data, if (op == .rms) &.{2} else &.{ 2, 2 });
        var dest = borrowed(&output, &.{ 2, 2 });
        var bad = x;
        bad.shape.len = 9;
        try expectError(error.DimensionOutOfBounds, neuralCall(op, &dest, &bad, &other));
        bad = x;
        bad.strides.len = 1;
        try expectError(error.StrideMismatch, neuralCall(op, &dest, &bad, &other));
        bad = x;
        bad.data = data[0..3];
        try expectError(error.InvalidStorage, neuralCall(op, &dest, &bad, &other));
        bad = x;
        bad.storage.device = backend.Device.metal;
        try expectError(error.DeviceMismatch, neuralCall(op, &dest, &bad, &other));
        var bad_dest = dest;
        bad_dest.strides = tensor.Shape.init(&.{ 1, 2 });
        try expectError(error.NonContiguousLayout, neuralCall(op, &bad_dest, &x, &other));
        bad_dest = dest;
        bad_dest.shape = tensor.Shape.init(&.{ 1, 4 });
        bad_dest.strides = tensor.computeContiguousStrides(bad_dest.shape);
        try expectError(error.ShapeMismatch, neuralCall(op, &bad_dest, &x, &other));
        bad_dest = dest;
        bad_dest.is_graph_view = true;
        try expectError(error.InPlaceOpOnGraphTensor, neuralCall(op, &bad_dest, &x, &other));
        bad = x;
        bad.data = &output;
        try expectError(error.AliasedOutput, neuralCall(op, &dest, &bad, &other));
        bad.requires_grad = true;
        try expectError(error.InPlaceOpOnGraphTensor, neuralCall(op, &dest, &bad, &other));
        if (op == .matmul or op == .rms) {
            var bad_other = other;
            bad_other.data = &output;
            try expectError(error.AliasedOutput, neuralCall(op, &dest, &x, &bad_other));
            bad_other.requires_grad = true;
            try expectError(error.InPlaceOpOnGraphTensor, neuralCall(op, &dest, &x, &bad_other));
            bad_other = other;
            bad_other.storage.device = backend.Device.metal;
            try expectError(error.DeviceMismatch, neuralCall(op, &dest, &x, &bad_other));
        }
        try expectEqualSlices(f32, &@as([4]f32, @splat(99)), &output);
    }
}

test "destination neural operation-specific validation and partial aliases" {
    var input = [_]f32{ 1, 2, 3, 4 };
    var storage: [6]f32 = @splat(99);
    var x = borrowed(&input, &.{ 2, 2 });
    var dest = borrowed(storage[0..4], &.{ 2, 2 });
    var invalid_weight = borrowed(&input, &.{ 1, 2 });
    try expectError(error.ShapeMismatch, tensor.rmsNormInto(&dest, &x, &invalid_weight, .{}));
    try expectError(error.DimensionOutOfBounds, tensor.softmaxInto(&dest, &x, 2));
    var rank_one = borrowed(&input, &.{4});
    try expectError(error.IncompatibleDimensions, tensor.matmulInto(&dest, &rank_one, &x));
    var wrong_k = borrowed(&input, &.{ 1, 2 });
    try expectError(error.ShapeMismatch, tensor.matmulInto(&dest, &x, &wrong_k));
    var overlap = borrowed(storage[1..5], &.{ 2, 2 });
    try expectError(error.AliasedOutput, tensor.matmulInto(&dest, &x, &overlap));
    try expectError(error.AliasedOutput, tensor.geluInto(&dest, &overlap, .none));
    try expectError(error.AliasedOutput, tensor.softmaxInto(&dest, &overlap, null));
    try expectError(error.AliasedOutput, tensor.rmsNormInto(&dest, &overlap, null, .{}));
    var weight = borrowed(storage[3..5], &.{2});
    try expectError(error.AliasedOutput, tensor.rmsNormInto(&dest, &x, &weight, .{}));
    try expectEqualSlices(f32, &@as([6]f32, @splat(99)), &storage);
    // Independent outputs may already belong to a graph, without changing gradients.
    var grad = [_]f32{ 10, 11, 12, 13 };
    dest.requires_grad = true;
    dest.grad = &grad;
    try tensor.matmulInto(&dest, &x, &x);
    try tensor.geluInto(&dest, &x, .tanh);
    try tensor.softmaxInto(&dest, &x, null);
    try tensor.rmsNormInto(&dest, &x, null, .{});
    try expectEqualSlices(f32, &.{ 10, 11, 12, 13 }, &grad);
    try expectEqualSlices(f32, &.{ 99, 99 }, storage[4..]);
    // Numeric options retain eager IEEE behavior, without extra validation.
    try tensor.rmsNormInto(&dest, &x, null, .{ .eps = -100 });
    for (storage[0..4]) |v| try expect(std.math.isNan(v));
}

test "destination softmax preserves eager nonfinite behavior on every axis" {
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    for ([_][2]f32{ .{ nan, 1 }, .{ inf, 1 }, .{ -inf, -inf }, .{ -inf, 1 } }) |values| {
        var input = values;
        var row = borrowed(&input, &.{2});
        const eager = try row.softmax(std.testing.allocator);
        defer eager.deinit(std.testing.allocator);
        var data = [_]f32{ values[0], values[0], values[1], values[1] };
        var x = borrowed(&data, &.{ 2, 2 });
        var out: [4]f32 = undefined;
        var dest = borrowed(&out, &.{ 2, 2 });
        for ([_]usize{ 0, 1 }) |axis| {
            x.strides = if (axis == 0) tensor.Shape.init(&.{ 2, 1 }) else tensor.Shape.init(&.{ 1, 2 });
            try tensor.softmaxInto(&dest, &x, axis);
            for (out, 0..) |actual, i| {
                const expected = eager.data[if (axis == 0) i / 2 else i % 2];
                if (std.math.isNan(expected)) {
                    try expect(std.math.isNan(actual));
                } else {
                    try std.testing.expectEqual(expected, actual);
                }
            }
        }
    }
}
