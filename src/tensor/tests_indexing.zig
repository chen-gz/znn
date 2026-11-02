//! Advanced indexing and strided view tests for Milestone 11.

const std = @import("std");
const testing = std.testing;
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const SliceRange = tensor.SliceRange;
const zeros = tensor.zeros;
const array = tensor.array;
const free = tensor.free;

test "indexing: negative step strided slicing and reverse materialization" {
    const allocator = testing.allocator;

    // 1. 1D reverse: [1, 2, 3, 4, 5] -> [5, 4, 3, 2, 1]
    const t1 = try array(allocator, &.{5}, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 });
    defer free(allocator, t1);

    const rev1 = try t1.slice(&.{.{ .step = -1 }}, allocator);
    defer free(allocator, rev1);

    try testing.expect(!rev1.is_view); // 显式逆序物化拷贝
    try testing.expectEqual(@as(usize, 5), rev1.data.len);
    try testing.expectEqualSlices(f32, &[_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0 }, rev1.data);

    // 2. Positive step preserves zero-copy view
    const fwd1 = try t1.slice(&.{.{ .step = 1 }}, allocator);
    defer free(allocator, fwd1);
    try testing.expect(fwd1.is_view);

    // 3. 1D reverse with explicit start and end: [1, 2, 3, 4, 5], start=3, end=0, step=-1 -> [4, 3, 2]
    const rev_part = try t1.slice(&.{.{ .start = 3, .end = 0, .step = -1 }}, allocator);
    defer free(allocator, rev_part);
    try testing.expect(!rev_part.is_view);
    try testing.expectEqualSlices(f32, &[_]f32{ 4.0, 3.0, 2.0 }, rev_part.data);

    // 4. 1D reverse with step = -2: [1, 2, 3, 4, 5] -> [5, 3, 1]
    const rev_step2 = try t1.slice(&.{.{ .step = -2 }}, allocator);
    defer free(allocator, rev_step2);
    try testing.expectEqualSlices(f32, &[_]f32{ 5.0, 3.0, 1.0 }, rev_step2.data);

    // 5. 2D reverse along dimension 0:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // -> [[4, 5, 6],
    //     [1, 2, 3]]
    const t2 = try array(allocator, &.{ 2, 3 }, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer free(allocator, t2);

    const rev2d_d0 = try t2.slice(&.{ .{ .step = -1 }, .{} }, allocator);
    defer free(allocator, rev2d_d0);
    try testing.expectEqualSlices(f32, &[_]f32{ 4.0, 5.0, 6.0, 1.0, 2.0, 3.0 }, rev2d_d0.data);

    // 6. 2D reverse along dimension 1:
    // -> [[3, 2, 1],
    //     [6, 5, 4]]
    const rev2d_d1 = try t2.slice(&.{ .{}, .{ .step = -1 } }, allocator);
    defer free(allocator, rev2d_d1);
    try testing.expectEqualSlices(f32, &[_]f32{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 }, rev2d_d1.data);

    // 7. Error cases
    try testing.expectError(error.InvalidStep, t1.slice(&.{.{ .step = 0 }}, allocator));
    try testing.expectError(error.IndexOutOfBounds, t1.slice(&.{.{ .start = 10, .step = -1 }}, allocator));
    try testing.expectError(error.InvalidSliceRange, t1.slice(&.{.{ .start = 1, .end = 3, .step = -1 }}, allocator));
}

test "indexing: 2D and 3D gather and gatherInto" {
    const allocator = testing.allocator;

    // PyTorch reference:
    // x = torch.tensor([[1., 2.], [3., 4.]])
    // index = torch.tensor([[0, 0], [1, 0]])
    // torch.gather(x, 0, index) -> [[1., 2.], [3., 2.]]
    const x = try array(allocator, &.{ 2, 2 }, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer free(allocator, x);

    const index0 = try array(allocator, &.{ 2, 2 }, &[_]f32{
        0.0, 0.0,
        1.0, 0.0,
    });
    defer free(allocator, index0);

    const g0 = try x.gather(0, index0, allocator);
    defer free(allocator, g0);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 2.0 }, g0.data);

    // torch.gather(x, 1, index) with:
    // x = [[10., 20., 30.], [40., 50., 60.]]
    // index = [[1, 0], [2, 1]]
    // result -> [[20., 10.], [60., 50.]]
    const x2 = try array(allocator, &.{ 2, 3 }, &[_]f32{
        10.0, 20.0, 30.0,
        40.0, 50.0, 60.0,
    });
    defer free(allocator, x2);

    const index1 = try array(allocator, &.{ 2, 2 }, &[_]f32{
        1.0, 0.0,
        2.0, 1.0,
    });
    defer free(allocator, index1);

    const g1 = try x2.gather(1, index1, allocator);
    defer free(allocator, g1);
    try testing.expectEqualSlices(f32, &[_]f32{ 20.0, 10.0, 60.0, 50.0 }, g1.data);

    // Destination-driven gatherInto
    const dest = try zeros(allocator, &.{ 2, 2 });
    defer free(allocator, dest);
    try x2.gatherInto(dest, 1, index1);
    try testing.expectEqualSlices(f32, &[_]f32{ 20.0, 10.0, 60.0, 50.0 }, dest.data);

    // IndexOutOfBounds error defense
    const bad_index = try array(allocator, &.{ 2, 2 }, &[_]f32{
        1.0, 3.0, // 3 is out of bounds for dim_size 3 (valid: 0, 1, 2)
        0.0, 0.0,
    });
    defer free(allocator, bad_index);
    try testing.expectError(error.IndexOutOfBounds, x2.gather(1, bad_index, allocator));
}

test "indexing: scatter and scatterInto" {
    const allocator = testing.allocator;

    // x = zeros(2, 3)
    // src = [[1., 2.], [3., 4.]]
    // index = [[0, 2], [1, 0]]
    // scatter along dim 1:
    // row 0: dest[0, 0] = 1, dest[0, 2] = 2
    // row 1: dest[1, 1] = 3, dest[1, 0] = 4
    // result:
    // [[1, 0, 2],
    //  [4, 3, 0]]
    const x = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, x);

    const src = try array(allocator, &.{ 2, 2 }, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer free(allocator, src);

    const index = try array(allocator, &.{ 2, 2 }, &[_]f32{
        0.0, 2.0,
        1.0, 0.0,
    });
    defer free(allocator, index);

    const sc = try x.scatter(1, index, src, allocator);
    defer free(allocator, sc);
    try testing.expectEqualSlices(f32, &[_]f32{
        1.0, 0.0, 2.0,
        4.0, 3.0, 0.0,
    }, sc.data);

    // scatterInto with base copy:
    // base = [[9., 9., 9.], [9., 9., 9.]]
    // result:
    // [[1, 9, 2],
    //  [4, 3, 9]]
    const base = try array(allocator, &.{ 2, 3 }, &[_]f32{
        9.0, 9.0, 9.0,
        9.0, 9.0, 9.0,
    });
    defer free(allocator, base);

    const dest = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, dest);
    try Tensor.scatterInto(dest, base, 1, index, src);
    try testing.expectEqualSlices(f32, &[_]f32{
        1.0, 9.0, 2.0,
        4.0, 3.0, 9.0,
    }, dest.data);
}

test "indexing: scatterAdd and scatterAddInto with overlapping indices" {
    const allocator = testing.allocator;

    // x = zeros(2, 3)
    // src = [[1., 2., 3.], [4., 5., 6.]]
    // index = [[0, 1, 0], [2, 2, 1]]
    // scatterAdd along dim 1:
    // row 0: dest[0, 0] += 1, dest[0, 1] += 2, dest[0, 0] += 3 -> [4, 2, 0]
    // row 1: dest[1, 2] += 4, dest[1, 2] += 5, dest[1, 1] += 6 -> [0, 6, 9]
    const x = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, x);

    const src = try array(allocator, &.{ 2, 3 }, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer free(allocator, src);

    const index = try array(allocator, &.{ 2, 3 }, &[_]f32{
        0.0, 1.0, 0.0,
        2.0, 2.0, 1.0,
    });
    defer free(allocator, index);

    const sc_add = try x.scatterAdd(1, index, src, allocator);
    defer free(allocator, sc_add);
    try testing.expectEqualSlices(f32, &[_]f32{
        4.0, 2.0, 0.0,
        0.0, 6.0, 9.0,
    }, sc_add.data);

    // 4-parameter in-place scatterAddInto
    const dest = try array(allocator, &.{ 2, 3 }, &[_]f32{
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0,
    });
    defer free(allocator, dest);

    try dest.scatterAddInto(1, index, src);
    try testing.expectEqualSlices(f32, &[_]f32{
        14.0, 12.0, 10.0,
        10.0, 16.0, 19.0,
    }, dest.data);
}

test "indexing: indexSelect and indexSelectInto" {
    const allocator = testing.allocator;

    // x = [[1, 2], [3, 4], [5, 6], [7, 8]] (4 x 2)
    // index = [0, 2, 0]
    // indexSelect along dim 0:
    // output shape: [3, 2]
    // row 0 = x[0] = [1, 2]
    // row 1 = x[2] = [5, 6]
    // row 2 = x[0] = [1, 2]
    const x = try array(allocator, &.{ 4, 2 }, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    });
    defer free(allocator, x);

    const index = try array(allocator, &.{3}, &[_]f32{ 0.0, 2.0, 0.0 });
    defer free(allocator, index);

    const sel = try x.indexSelect(0, index, allocator);
    defer free(allocator, sel);
    try testing.expectEqual(@as(usize, 2), sel.shape.len);
    try testing.expectEqual(@as(usize, 3), sel.shape.dims[0]);
    try testing.expectEqual(@as(usize, 2), sel.shape.dims[1]);
    try testing.expectEqualSlices(f32, &[_]f32{
        1.0, 2.0,
        5.0, 6.0,
        1.0, 2.0,
    }, sel.data);

    // gatherSlice convenience helper with []const usize
    const sel_slice = try tensor.gatherSlice(x, 0, &.{ 0, 2, 0 }, allocator);
    defer free(allocator, sel_slice);
    try testing.expectEqualSlices(f32, sel.data, sel_slice.data);
}

test "indexing: indexAdd and indexAddInto" {
    const allocator = testing.allocator;

    // x = zeros(3, 2)
    // index = [0, 2]
    // src = [[1, 2], [3, 4]]
    // indexAdd along dim 0:
    // dest[0] += [1, 2]
    // dest[2] += [3, 4]
    // dest[1] remains 0
    const x = try zeros(allocator, &.{ 3, 2 });
    defer free(allocator, x);

    const index = try array(allocator, &.{2}, &[_]f32{ 0.0, 2.0 });
    defer free(allocator, index);

    const src = try array(allocator, &.{ 2, 2 }, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer free(allocator, src);

    const added = try x.indexAdd(0, index, src, allocator);
    defer free(allocator, added);
    try testing.expectEqualSlices(f32, &[_]f32{
        1.0, 2.0,
        0.0, 0.0,
        3.0, 4.0,
    }, added.data);

    // Repeated indices: index = [1, 1], accumulates into dest[1]
    const rep_idx = try array(allocator, &.{2}, &[_]f32{ 1.0, 1.0 });
    defer free(allocator, rep_idx);

    const rep_dest = try zeros(allocator, &.{ 3, 2 });
    defer free(allocator, rep_dest);

    try rep_dest.indexAddInto(0, rep_idx, src);
    try testing.expectEqualSlices(f32, &[_]f32{
        0.0, 0.0,
        4.0, 6.0, // [1, 2] + [3, 4] = [4, 6]
        0.0, 0.0,
    }, rep_dest.data);
}

test "indexing: maskedSelect 1D compression" {
    const allocator = testing.allocator;

    // x = [[1, 2, 3], [4, 5, 6]]
    // mask = [[1, 0, 1], [0, 1, 0]]
    // maskedSelect -> [1, 3, 5]
    const x = try array(allocator, &.{ 2, 3 }, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer free(allocator, x);

    const mask = try array(allocator, &.{ 2, 3 }, &[_]f32{
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
    });
    defer free(allocator, mask);

    const selected = try x.maskedSelect(mask, allocator);
    defer free(allocator, selected);

    try testing.expectEqual(@as(usize, 1), selected.shape.len);
    try testing.expectEqual(@as(usize, 3), selected.shape.dims[0]);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 3.0, 5.0 }, selected.data);

    // Empty mask (all zeros) -> empty 1D tensor
    const zero_mask = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, zero_mask);

    const empty_sel = try x.maskedSelect(zero_mask, allocator);
    defer free(allocator, empty_sel);
    try testing.expectEqual(@as(usize, 0), empty_sel.shape.dims[0]);
    try testing.expectEqual(@as(usize, 0), empty_sel.data.len);
}
