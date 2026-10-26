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
        bad.device = backend.Device.metal;
        try expectError(error.DeviceMismatch, op(&dest, &bad, &b));
        bad = a;
        bad.device_array = .{ .mlx = undefined };
        try expectError(error.DeviceMismatch, op(&dest, &bad, &b));
        dest.device = backend.Device.cuda;
        try expectError(error.DeviceMismatch, op(&dest, &a, &b));
        dest = good;
        dest.device_array = .{ .mlx = undefined };
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
