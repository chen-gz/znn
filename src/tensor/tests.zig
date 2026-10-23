const std = @import("std");
const tensor = @import("../tensor.zig");
const Shape = tensor.Shape;
const broadcastShapes = tensor.broadcastShapes;
const DType = tensor.DType;
const bf16 = tensor.bf16;
const convertScalar = tensor.convertScalar;
const GenericTensor = tensor.GenericTensor;
const IntTensor = tensor.IntTensor;
const BoolTensor = tensor.BoolTensor;
const BFloat16Tensor = tensor.BFloat16Tensor;
const Tensor = tensor.Tensor;
const array = tensor.array;
const zeros = tensor.zeros;
const ones = tensor.ones;
const full = tensor.full;
const free = tensor.free;
const slice = tensor.slice;
const clip = tensor.clip;
const sort = tensor.sort;
const argsort = tensor.argsort;
const nonzero = tensor.nonzero;
const concat = tensor.concat;
const split = tensor.split;
const unsqueeze = tensor.unsqueeze;
const squeeze = tensor.squeeze;
const sum = tensor.sum;
const mean = tensor.mean;
const variance = tensor.variance;
const stdDev = tensor.stdDev;
const where = tensor.where;
const arange = tensor.arange;
const linspace = tensor.linspace;
const eye = tensor.eye;
const identity = tensor.identity;
const stack = tensor.stack;
const repeat = tensor.repeat;
const tile = tensor.tile;
const sqrt = tensor.sqrt;
const exp = tensor.exp;
const log = tensor.log;
const abs = tensor.abs;
const applyRoPE = tensor.applyRoPE;
const solveLinearSystem = tensor.solveLinearSystem;


test "applyRoPE rotation properties" {
    var data = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // seq_len=2, n_head=1, head_dim=2
    applyRoPE(&data, 2, 1, 2, 10000.0);
    // m = 0: theta^0 = 1, freq = 0 -> cos(0)=1, sin(0)=0 -> x0=1.0, x1=0.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 1e-5);

    // m = 1: freq = 1.0 -> cos(1), sin(1) for [0.0, 1.0] -> x2 = 0*cos(1) - 1*sin(1) = -sin(1), x3 = 0*sin(1) + 1*cos(1) = cos(1)
    const expected_x2 = -@sin(@as(f32, 1.0));
    const expected_x3 = @cos(@as(f32, 1.0));
    try std.testing.expectApproxEqAbs(expected_x2, data[2], 1e-5);
    try std.testing.expectApproxEqAbs(expected_x3, data[3], 1e-5);
}

test "broadcastShapes inference" {
    // 1. Same shapes
    const s1 = Shape.init(&.{ 2, 3 });
    const s2 = Shape.init(&.{ 2, 3 });
    const out1 = try broadcastShapes(s1, s2);
    try std.testing.expect(out1.eq(Shape.init(&.{ 2, 3 })));

    // 2. Trailing dimensions with 1s
    const s3 = Shape.init(&.{ 4, 1, 5 });
    const s4 = Shape.init(&.{ 3, 5 });
    const out2 = try broadcastShapes(s3, s4);
    try std.testing.expect(out2.eq(Shape.init(&.{ 4, 3, 5 })));

    // 3. Different rank multi-dim broadcasting
    const s5 = Shape.init(&.{ 2, 1, 4, 1 });
    const s6 = Shape.init(&.{ 3, 1, 5 });
    const out3 = try broadcastShapes(s5, s6);
    try std.testing.expect(out3.eq(Shape.init(&.{ 2, 3, 4, 5 })));

    // 4. Incompatible shapes
    const s7 = Shape.init(&.{ 3, 4 });
    const s8 = Shape.init(&.{ 2, 4 });
    try std.testing.expectError(error.IncompatibleBroadcastShapes, broadcastShapes(s7, s8));
}

test "tensor eager broadcasting operations (add, sub, mul, div)" {
    const allocator = std.testing.allocator;

    // A: 2x3 matrix
    const a = try array(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer free(allocator, a);

    // B: 1x3 row vector
    const b = try array(allocator, &.{ 1, 3 }, &.{ 10.0, 20.0, 30.0 });
    defer free(allocator, b);

    // C: 2x1 col vector
    const c_vec = try array(allocator, &.{ 2, 1 }, &.{ 100.0, 200.0 });
    defer free(allocator, c_vec);

    // 1. A + B -> 2x3
    const a_add_b = try a.add(b, allocator);
    defer free(allocator, a_add_b);
    try std.testing.expect(a_add_b.shape.eq(Shape.init(&.{ 2, 3 })));
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), a_add_b.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), a_add_b.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), a_add_b.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), a_add_b.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), a_add_b.data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), a_add_b.data[5], 1e-5);

    // 2. A * C -> 2x3
    const a_mul_c = try a.mul(c_vec, allocator);
    defer free(allocator, a_mul_c);
    try std.testing.expect(a_mul_c.shape.eq(Shape.init(&.{ 2, 3 })));
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), a_mul_c.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), a_mul_c.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), a_mul_c.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 800.0), a_mul_c.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1000.0), a_mul_c.data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1200.0), a_mul_c.data[5], 1e-5);

    // 3. B - A -> 2x3
    const b_sub_a = try b.sub(a, allocator);
    defer free(allocator, b_sub_a);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), b_sub_a.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), b_sub_a.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), b_sub_a.data[2], 1e-5);

    // 4. B / A -> 2x3
    const b_div_a = try b.div(a, allocator);
    defer free(allocator, b_div_a);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[2], 1e-5);

    // 5. 4D Broadcasting: [2, 1, 3, 1] + [1, 2, 1, 4] -> [2, 2, 3, 4] (Total 48 elements)
    const t4d_1 = try ones(allocator, &.{ 2, 1, 3, 1 });
    defer free(allocator, t4d_1);
    const t4d_2 = try array(allocator, &.{ 1, 2, 1, 4 }, &.{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    });
    defer free(allocator, t4d_2);

    const t4d_out = try t4d_1.add(t4d_2, allocator);
    defer free(allocator, t4d_out);
    try std.testing.expect(t4d_out.shape.eq(Shape.init(&.{ 2, 2, 3, 4 })));
    try std.testing.expectEqual(@as(usize, 48), t4d_out.data.len);
    // Elements should be 1.0 + t4d_2 values
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), t4d_out.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), t4d_out.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), t4d_out.data[12], 1e-5);

    // 6. subScalar and divScalar
    const s_sub = try a.subScalar(1.0, allocator);
    defer free(allocator, s_sub);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s_sub.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), s_sub.data[5], 1e-5);

    const s_div = try a.divScalar(2.0, allocator);
    defer free(allocator, s_div);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s_div.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), s_div.data[5], 1e-5);
}

test "tensor argmax, argmin, max, min multi-dimension support and edge cases" {
    const allocator = std.testing.allocator;

    // 2x3 matrix with all negative values
    var t = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, t);

    // Row 0: [-10.0, -2.0, -5.0] -> max index is 1 (-2.0), min index is 0 (-10.0)
    // Row 1: [-1.0, -8.0, -4.0]  -> max index is 0 (-1.0), min index is 1 (-8.0)
    t.set(&.{ 0, 0 }, -10.0);
    t.set(&.{ 0, 1 }, -2.0);
    t.set(&.{ 0, 2 }, -5.0);
    t.set(&.{ 1, 0 }, -1.0);
    t.set(&.{ 1, 1 }, -8.0);
    t.set(&.{ 1, 2 }, -4.0);

    // 1. argmax / argmin / max / min dim=1
    const idx_col = try t.argmax(1, allocator);
    defer free(allocator, idx_col);
    try std.testing.expectEqual(@as(f32, 1.0), idx_col.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_col.data[1]);

    const min_idx_col = try t.argmin(1, allocator);
    defer free(allocator, min_idx_col);
    try std.testing.expectEqual(@as(f32, 0.0), min_idx_col.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), min_idx_col.data[1]);

    const max_val_col = try t.max(1, allocator);
    defer free(allocator, max_val_col);
    try std.testing.expectEqual(@as(f32, -2.0), max_val_col.data[0]);
    try std.testing.expectEqual(@as(f32, -1.0), max_val_col.data[1]);

    const min_val_col = try t.min(1, allocator);
    defer free(allocator, min_val_col);
    try std.testing.expectEqual(@as(f32, -10.0), min_val_col.data[0]);
    try std.testing.expectEqual(@as(f32, -8.0), min_val_col.data[1]);

    // 2. argmax / argmin dim=0
    // Col 0: -10 vs -1 -> max idx 1, min idx 0
    // Col 1: -2 vs -8  -> max idx 0, min idx 1
    // Col 2: -5 vs -4  -> max idx 1, min idx 0
    const idx_row = try t.argmax(0, allocator);
    defer free(allocator, idx_row);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_row.data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[2]);

    const min_idx_row = try t.argmin(0, allocator);
    defer free(allocator, min_idx_row);
    try std.testing.expectEqual(@as(f32, 0.0), min_idx_row.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), min_idx_row.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), min_idx_row.data[2]);

    // 3. 1D vector test
    var v = try zeros(allocator, &.{4});
    defer free(allocator, v);
    @memcpy(v.data, &[_]f32{ 3.0, 7.0, 1.0, 5.0 });

    const v_argmax = try v.argmax(0, allocator);
    defer free(allocator, v_argmax);
    try std.testing.expectEqual(@as(f32, 1.0), v_argmax.data[0]);

    const v_argmin = try v.argmin(0, allocator);
    defer free(allocator, v_argmin);
    try std.testing.expectEqual(@as(f32, 2.0), v_argmin.data[0]);

    // 4. 3D tensor: shape [2, 2, 3] along dim 2
    var t_3d = try zeros(allocator, &.{ 2, 2, 3 });
    defer free(allocator, t_3d);
    t_3d.data[0] = 5.0; // [0, 0]: max at 0
    t_3d.data[1] = 1.0;
    t_3d.data[2] = 2.0;

    t_3d.data[3] = 1.0;
    t_3d.data[4] = 9.0; // [0, 1]: max at 1
    t_3d.data[5] = 4.0;

    t_3d.data[6] = 2.0;
    t_3d.data[7] = 3.0;
    t_3d.data[8] = 8.0; // [1, 0]: max at 2

    t_3d.data[9] = 10.0; // [1, 1]: max at 0
    t_3d.data[10] = 0.0;
    t_3d.data[11] = 1.0;

    const res_3d = try t_3d.argmax(2, allocator);
    defer free(allocator, res_3d);
    try std.testing.expectEqual(@as(usize, 3), res_3d.shape.len);
    try std.testing.expectEqual(@as(usize, 2), res_3d.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), res_3d.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), res_3d.shape.dims[2]);
    try std.testing.expectEqual(@as(f32, 0.0), res_3d.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), res_3d.data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), res_3d.data[2]);
    try std.testing.expectEqual(@as(f32, 0.0), res_3d.data[3]);
}

test "solveLinearSystem singular matrix error and n=1 scalar" {
    const allocator = std.testing.allocator;

    // 1. n = 1 non-zero
    const A1 = [_]f32{4.0};
    const b1 = [_]f32{8.0};
    var x1 = [_]f32{0.0};
    try solveLinearSystem(allocator, &A1, &b1, 1, &x1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x1[0], 1e-6);

    // 2. n = 1 singular (zero)
    const A_zero = [_]f32{0.0};
    try std.testing.expectError(error.SingularMatrix, solveLinearSystem(allocator, &A_zero, &b1, 1, &x1));

    // 3. n = 2 singular matrix (linearly dependent rows: [1, 2; 2, 4])
    const A_sing = [_]f32{ 1.0, 2.0, 2.0, 4.0 };
    const b2 = [_]f32{ 3.0, 6.0 };
    var x2 = [_]f32{ 0.0, 0.0 };
    try std.testing.expectError(error.SingularMatrix, solveLinearSystem(allocator, &A_sing, &b2, 2, &x2));
}

test "tensor typed error handling and boundary validation" {
    const allocator = std.testing.allocator;

    // 1. Shape.fromSlice and creation functions exceeding max dimensions (8)
    const nine_dims = [_]usize{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    try std.testing.expectError(error.MaxDimensionsExceeded, Shape.fromSlice(&nine_dims));
    try std.testing.expectError(error.MaxDimensionsExceeded, zeros(allocator, &nine_dims));
    try std.testing.expectError(error.MaxDimensionsExceeded, ones(allocator, &nine_dims));

    // 2. array data length mismatch
    const data_3 = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.ShapeMismatch, array(allocator, &.{ 2, 2 }, &data_3));

    // 3. getFlatIndexChecked and safe getChecked/setChecked
    var t2x2 = try zeros(allocator, &.{ 2, 2 });
    defer free(allocator, t2x2);
    try std.testing.expectError(error.DimensionMismatch, t2x2.getFlatIndexChecked(&.{ 0, 0, 0 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.getFlatIndexChecked(&.{ 2, 0 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.getChecked(&.{ 0, 3 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.setChecked(&.{ 3, 0 }, 1.0));
    try t2x2.setChecked(&.{ 1, 1 }, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), try t2x2.getChecked(&.{ 1, 1 }));

    // 4. matmul error conditions
    const t1d = try zeros(allocator, &.{4});
    defer free(allocator, t1d);
    var t2x3 = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, t2x3);
    const t4x2 = try zeros(allocator, &.{ 4, 2 });
    defer free(allocator, t4x2);
    // Non-2D inputs
    try std.testing.expectError(error.IncompatibleDimensions, t2x2.matmul(t1d, allocator));
    // Inner dimension mismatch (2x3 cannot multiply 4x2)
    try std.testing.expectError(error.ShapeMismatch, t2x3.matmul(t4x2, allocator));

    // 5. batchMatMul error conditions
    var t4d_a = try zeros(allocator, &.{ 1, 2, 3, 4 });
    defer free(allocator, t4d_a);
    const t4d_b_bad = try zeros(allocator, &.{ 1, 2, 5, 6 }); // K mismatch (4 != 5)
    defer free(allocator, t4d_b_bad);
    try std.testing.expectError(error.IncompatibleDimensions, t4d_a.batchMatMul(t2x2, allocator));
    try std.testing.expectError(error.ShapeMismatch, t4d_a.batchMatMul(t4d_b_bad, allocator));

    // 6. reshape element count mismatch
    try std.testing.expectError(error.ShapeMismatch, t2x2.reshape(&.{ 3, 3 }, allocator));

    // 7. transpose dimension out of bounds
    try std.testing.expectError(error.DimensionOutOfBounds, t2x2.transpose(0, 3, allocator));

    // 8. conv1d and conv2d error conditions
    const t3d_a = try zeros(allocator, &.{ 1, 2, 4 });
    defer free(allocator, t3d_a);
    const w1d_bad_c = try zeros(allocator, &.{ 2, 3, 2 });
    defer free(allocator, w1d_bad_c);
    try std.testing.expectError(error.ShapeMismatch, t3d_a.conv1d(w1d_bad_c, null, .{}, allocator));
    const w1d_too_big = try zeros(allocator, &.{ 2, 2, 10 });
    defer free(allocator, w1d_too_big);
    try std.testing.expectError(error.KernelBiggerThanInput, t3d_a.conv1d(w1d_too_big, null, .{}, allocator));

    const w_bad_c = try zeros(allocator, &.{ 2, 3, 2, 2 }); // C_in mismatch with t4d_a (C_in is 2, weight has 3)
    defer free(allocator, w_bad_c);
    try std.testing.expectError(error.ShapeMismatch, t4d_a.conv2d(w_bad_c, null, .{}, allocator));
    const w_too_big = try zeros(allocator, &.{ 2, 2, 5, 5 }); // KH/KW > H/W (5 > 3 or 4)
    defer free(allocator, w_too_big);
    try std.testing.expectError(error.KernelBiggerThanInput, t4d_a.conv2d(w_too_big, null, .{}, allocator));

    // 9. concat error conditions
    try std.testing.expectError(error.EmptyInputs, concat(allocator, &.{}, 0));
    const inputs_dim_out = [_]*Tensor{t2x2};
    try std.testing.expectError(error.DimensionOutOfBounds, concat(allocator, &inputs_dim_out, 3));
    const inputs_mismatch = [_]*Tensor{ t2x2, t2x3 };
    try std.testing.expectError(error.ShapeMismatch, concat(allocator, &inputs_mismatch, 0));

    // 10. split error conditions
    try std.testing.expectError(error.InvalidSplitCount, split(allocator, t2x2, 0, 0));
    try std.testing.expectError(error.DimensionOutOfBounds, split(allocator, t2x2, 2, 5));
    try std.testing.expectError(error.UnevenSplit, split(allocator, t2x3, 2, 1)); // dim 1 has size 3, not divisible by 2

    // 11. solveLinearSystem slice length mismatch
    const bad_A = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{1.0};
    var x = [_]f32{0.0};
    try std.testing.expectError(error.ShapeMismatch, solveLinearSystem(allocator, &bad_A, &b, 1, &x));
}

test "Tensor multi-axis reductions (sum, mean, variance, stdDev)" {
    const allocator = std.testing.allocator;

    // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    const t = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t);

    // 1. sum over all elements (axis = null)
    {
        const s_all = try t.sum(null, false, allocator);
        defer free(allocator, s_all);
        try std.testing.expectEqual(@as(usize, 1), s_all.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s_all.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 21.0), s_all.data[0]);

        const s_all_kd = try t.sum(null, true, allocator);
        defer free(allocator, s_all_kd);
        try std.testing.expectEqual(@as(usize, 2), s_all_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s_all_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 1), s_all_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 21.0), s_all_kd.data[0]);
    }

    // 2. sum over axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    {
        const s0 = try sum(t, 0, false, allocator);
        defer free(allocator, s0);
        try std.testing.expectEqual(@as(usize, 1), s0.shape.len);
        try std.testing.expectEqual(@as(usize, 3), s0.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 5.0), s0.data[0]);
        try std.testing.expectEqual(@as(f32, 7.0), s0.data[1]);
        try std.testing.expectEqual(@as(f32, 9.0), s0.data[2]);

        const s0_kd = try sum(t, 0, true, allocator);
        defer free(allocator, s0_kd);
        try std.testing.expectEqual(@as(usize, 2), s0_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s0_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), s0_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 5.0), s0_kd.data[0]);
    }

    // 3. sum over axis 1: [1+2+3, 4+5+6] = [6, 15]
    {
        const s1 = try sum(t, 1, false, allocator);
        defer free(allocator, s1);
        try std.testing.expectEqual(@as(usize, 1), s1.shape.len);
        try std.testing.expectEqual(@as(usize, 2), s1.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 6.0), s1.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s1.data[1]);

        const s1_kd = try sum(t, 1, true, allocator);
        defer free(allocator, s1_kd);
        try std.testing.expectEqual(@as(usize, 2), s1_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 2), s1_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 1), s1_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 6.0), s1_kd.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s1_kd.data[1]);
    }

    // 4. Non-contiguous sum test: custom strided view
    {
        var t_strided = Tensor{
            .data = t.data,
            .grad = &.{},
            .shape = Shape.init(&.{ 3, 2 }),
            .strides = Shape.init(&.{ 1, 3 }), // transposed strides!
            .requires_grad = false,
            .creator = null,
        };
        try std.testing.expect(!t_strided.isContiguous());
        const s_strided = try t_strided.sum(0, false, allocator);
        defer free(allocator, s_strided);
        // r=0: [1, 4], r=1: [2, 5], r=2: [3, 6]
        // sum along axis 0 gives [1+2+3, 4+5+6] = [6, 15]
        try std.testing.expectEqual(@as(f32, 6.0), s_strided.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s_strided.data[1]);
    }

    // 5. mean
    {
        const m_all = try mean(t, null, false, allocator);
        defer free(allocator, m_all);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), m_all.data[0], 1e-5);

        const m0 = try mean(t, 0, false, allocator);
        defer free(allocator, m0);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), m0.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), m0.data[1], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 4.5), m0.data[2], 1e-5);

        const m1 = try mean(t, 1, false, allocator);
        defer free(allocator, m1);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), m1.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), m1.data[1], 1e-5);
    }

    // 6. variance and stdDev
    {
        // variance with ddof=0: 17.5 / 6 = 2.9166667
        const v_all = try variance(t, null, false, 0, allocator);
        defer free(allocator, v_all);
        try std.testing.expectApproxEqAbs(@as(f32, 2.9166667), v_all.data[0], 1e-5);

        // variance with ddof=1: 17.5 / 5 = 3.5
        const v_sample = try variance(t, null, false, 1, allocator);
        defer free(allocator, v_sample);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), v_sample.data[0], 1e-5);

        // stdDev with ddof=0: sqrt(2.9166667) ~= 1.7078251
        const sd_all = try stdDev(t, null, false, 0, allocator);
        defer free(allocator, sd_all);
        try std.testing.expectApproxEqAbs(@as(f32, 1.7078251), sd_all.data[0], 1e-5);

        // stdDev along axis 1:
        // row 0: [1, 2, 3], mean = 2, sq_diff sum = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2.
        // var(ddof=0) = 2/3, stdDev = sqrt(2/3) ~= 0.8164966
        const sd1 = try stdDev(t, 1, false, 0, allocator);
        defer free(allocator, sd1);
        try std.testing.expectApproxEqAbs(@as(f32, 0.8164966), sd1.data[0], 1e-5);
    }

    // 7. Error handling
    try std.testing.expectError(error.DimensionOutOfBounds, t.sum(5, false, allocator));
    try std.testing.expectError(error.DimensionOutOfBounds, t.mean(2, false, allocator));
    try std.testing.expectError(error.InvalidDDOF, t.variance(null, false, 6, allocator));
    try std.testing.expectError(error.InvalidDDOF, t.stdDev(null, false, 10, allocator));
}

test "Tensor where condition and masking operations" {
    const allocator = std.testing.allocator;

    // 1. where with identical shapes
    const cond = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    defer free(allocator, cond);
    const x = try array(allocator, &.{ 2, 2 }, &[_]f32{ 10.0, 20.0, 30.0, 40.0 });
    defer free(allocator, x);
    const y = try array(allocator, &.{ 2, 2 }, &[_]f32{ -1.0, -2.0, -3.0, -4.0 });
    defer free(allocator, y);

    const out = try where(cond, x, y, allocator);
    defer free(allocator, out);
    try std.testing.expectEqual(@as(f32, 10.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, -2.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, -3.0), out.data[2]);
    try std.testing.expectEqual(@as(f32, 40.0), out.data[3]);

    // 2. where with broadcast condition
    // cond shape [2, 1], x shape [2, 2], y shape [2, 2]
    const cond_bc = try array(allocator, &.{ 2, 1 }, &[_]f32{ 1.0, 0.0 });
    defer free(allocator, cond_bc);
    const out_bc = try where(cond_bc, x, y, allocator);
    defer free(allocator, out_bc);
    // Row 0 selects x: [10.0, 20.0]; Row 1 selects y: [-3.0, -4.0]
    try std.testing.expectEqual(@as(f32, 10.0), out_bc.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), out_bc.data[1]);
    try std.testing.expectEqual(@as(f32, -3.0), out_bc.data[2]);
    try std.testing.expectEqual(@as(f32, -4.0), out_bc.data[3]);

    // 3. maskedFill (out of place)
    const mask = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 0.0, 1.0, 0.0 });
    defer free(allocator, mask);
    const filled = try x.maskedFill(mask, -999.0, allocator);
    defer free(allocator, filled);
    try std.testing.expectEqual(@as(f32, -999.0), filled.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), filled.data[1]);
    try std.testing.expectEqual(@as(f32, -999.0), filled.data[2]);
    try std.testing.expectEqual(@as(f32, 40.0), filled.data[3]);
    // Original x should remain unchanged
    try std.testing.expectEqual(@as(f32, 10.0), x.data[0]);

    // 4. maskedFill_ (in place)
    var x_mut = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer free(allocator, x_mut);
    _ = try x_mut.maskedFill_(mask, 0.0);
    try std.testing.expectEqual(@as(f32, 0.0), x_mut.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), x_mut.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), x_mut.data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), x_mut.data[3]);

    // In-place protection for graph tensor
    x_mut.requires_grad = true;
    try std.testing.expectError(error.InPlaceOpOnGraphTensor, x_mut.maskedFill_(mask, 1.0));

    // Shape mismatch
    const bad_mask = try zeros(allocator, &.{3});
    defer free(allocator, bad_mask);
    try std.testing.expectError(error.ShapeMismatch, x.maskedFill(bad_mask, 0.0, allocator));
}

test "Tensor squeeze and unsqueeze" {
    const allocator = std.testing.allocator;

    // 1. Squeeze
    const t_4d = try zeros(allocator, &.{ 1, 2, 1, 3 });
    defer free(allocator, t_4d);

    // Squeeze all size-1 dims
    const t_sq_all = try squeeze(t_4d, null, allocator);
    defer free(allocator, t_sq_all);
    try std.testing.expectEqual(@as(usize, 2), t_sq_all.shape.len);
    try std.testing.expectEqual(@as(usize, 2), t_sq_all.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_all.shape.dims[1]);

    // Squeeze specific dim 0
    const t_sq_0 = try squeeze(t_4d, 0, allocator);
    defer free(allocator, t_sq_0);
    try std.testing.expectEqual(@as(usize, 3), t_sq_0.shape.len);
    try std.testing.expectEqual(@as(usize, 2), t_sq_0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), t_sq_0.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_0.shape.dims[2]);

    // Squeeze specific dim 2
    const t_sq_2 = try squeeze(t_4d, 2, allocator);
    defer free(allocator, t_sq_2);
    try std.testing.expectEqual(@as(usize, 3), t_sq_2.shape.len);
    try std.testing.expectEqual(@as(usize, 1), t_sq_2.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), t_sq_2.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_2.shape.dims[2]);

    // Cannot squeeze non-unit dimension
    try std.testing.expectError(error.CannotSqueezeDimension, t_4d.squeeze(1, allocator));
    try std.testing.expectError(error.DimensionOutOfBounds, t_4d.squeeze(5, allocator));

    // Squeeze on tensor where all dimensions are 1
    const t_1x1 = try zeros(allocator, &.{ 1, 1 });
    defer free(allocator, t_1x1);
    const t_sq_scalar = try t_1x1.squeeze(null, allocator);
    defer free(allocator, t_sq_scalar);
    try std.testing.expectEqual(@as(usize, 1), t_sq_scalar.shape.len);
    try std.testing.expectEqual(@as(usize, 1), t_sq_scalar.shape.dims[0]);

    // 2. Unsqueeze
    const t_2d = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t_2d);

    // Insert at dim 0: [1, 2, 3]
    const u_dim0 = try unsqueeze(t_2d, 0, allocator);
    defer free(allocator, u_dim0);
    try std.testing.expectEqual(@as(usize, 3), u_dim0.shape.len);
    try std.testing.expectEqual(@as(usize, 1), u_dim0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), u_dim0.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), u_dim0.shape.dims[2]);
    try std.testing.expectEqual(@as(f32, 1.0), u_dim0.data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), u_dim0.data[5]);

    // Insert at dim 1: [2, 1, 3]
    const u_dim1 = try unsqueeze(t_2d, 1, allocator);
    defer free(allocator, u_dim1);
    try std.testing.expectEqual(@as(usize, 3), u_dim1.shape.len);
    try std.testing.expectEqual(@as(usize, 2), u_dim1.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), u_dim1.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), u_dim1.shape.dims[2]);

    // Insert at dim 2 (end): [2, 3, 1]
    const u_dim2 = try unsqueeze(t_2d, 2, allocator);
    defer free(allocator, u_dim2);
    try std.testing.expectEqual(@as(usize, 3), u_dim2.shape.len);
    try std.testing.expectEqual(@as(usize, 2), u_dim2.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), u_dim2.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), u_dim2.shape.dims[2]);

    // Out of bounds
    try std.testing.expectError(error.DimensionOutOfBounds, t_2d.unsqueeze(4, allocator));
}

test "DType, bf16, and scalar type conversion" {
    // 1. DType sizeOf
    try std.testing.expectEqual(@as(usize, 4), DType.f32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.sizeOf());
    try std.testing.expectEqual(@as(usize, 4), DType.i32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.sizeOf());
    try std.testing.expectEqual(@as(usize, 1), DType.u8.sizeOf());
    try std.testing.expectEqual(@as(usize, 1), DType.bool.sizeOf());

    // 2. bf16 conversions
    const b0 = bf16.fromF32(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), b0.toF32());

    const b1 = bf16.fromF32(1.0);
    try std.testing.expectEqual(@as(f32, 1.0), b1.toF32());

    const bm1 = bf16.fromF32(-1.0);
    try std.testing.expectEqual(@as(f32, -1.0), bm1.toF32());

    const b2_5 = bf16.fromF32(2.5);
    try std.testing.expectEqual(@as(f32, 2.5), b2_5.toF32());

    const b_pi = bf16.fromF32(3.14159);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14159), b_pi.toF32(), 1e-2);

    // 3. convertScalar
    try std.testing.expectEqual(@as(f64, 1.5), convertScalar(f64, f32, 1.5));
    try std.testing.expectEqual(@as(i32, 42), convertScalar(i32, f32, 42.0));
    try std.testing.expectEqual(true, convertScalar(bool, i32, 1));
    try std.testing.expectEqual(false, convertScalar(bool, i32, 0));
    try std.testing.expectEqual(@as(f32, 2.0), convertScalar(f32, bf16, bf16.fromF32(2.0)));
    try std.testing.expectEqual(@as(f32, 2.0), convertScalar(bf16, f32, 2.0).toF32());
}

test "GenericTensor and multi-type tensor manipulation" {
    const allocator = std.testing.allocator;

    // 1. GenericTensor(i32)
    const t_i32 = try GenericTensor(i32).fromSlice(allocator, &.{ 2, 2 }, &[_]i32{ 1, 2, 3, 4 });
    defer t_i32.deinit(allocator);
    try std.testing.expectEqual(@as(i32, 1), t_i32.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(i32, 3), t_i32.get(&.{ 1, 0 }));
    t_i32.set(&.{ 1, 0 }, 30);
    try std.testing.expectEqual(@as(i32, 30), t_i32.get(&.{ 1, 0 }));

    // 2. GenericTensor(bool)
    const t_b = try BoolTensor.fromSlice(allocator, &.{3}, &[_]bool{ true, false, true });
    defer t_b.deinit(allocator);
    try std.testing.expectEqual(true, t_b.get(&.{0}));
    try std.testing.expectEqual(false, t_b.get(&.{1}));
    try std.testing.expectEqual(true, t_b.get(&.{2}));

    // 3. GenericTensor to conversion
    const t_f32_conv = try t_i32.to(f32, allocator);
    defer t_f32_conv.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 1.0), t_f32_conv.data[0]);
    try std.testing.expectEqual(@as(f32, 30.0), t_f32_conv.data[2]);

    // 4. BFloat16Tensor
    const t_bf16 = try BFloat16Tensor.init(allocator, &.{2}, bf16.fromF32(3.5));
    defer t_bf16.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 3.5), t_bf16.data[0].toF32());

    // 5. Bidirectional Tensor <-> GenericTensor
    const t_orig = try array(allocator, &.{2}, &[_]f32{ 10.0, 20.0 });
    defer free(allocator, t_orig);
    const t_int_gen = try t_orig.to(i32, allocator);
    defer t_int_gen.deinit(allocator);
    try std.testing.expectEqual(@as(i32, 10), t_int_gen.data[0]);
    try std.testing.expectEqual(@as(i32, 20), t_int_gen.data[1]);

    const t_back = try Tensor.fromGeneric(i32, t_int_gen, allocator);
    defer free(allocator, t_back);
    try std.testing.expectEqual(@as(f32, 10.0), t_back.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), t_back.data[1]);
}

test "Tensor strided view slicing, contiguous, clip, sort, argsort, nonzero" {
    const allocator = std.testing.allocator;

    // 1. Slicing on 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var t = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t);

    // Extract submatrix rows 0..2, cols 1..3 -> [[2, 3], [5, 6]]
    const s = try slice(t, &.{ .{ .start = 0, .end = 2 }, .{ .start = 1, .end = 3 } }, allocator);
    defer free(allocator, s);
    try std.testing.expect(s.is_view);
    try std.testing.expectEqual(@as(usize, 2), s.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s.shape.dims[1]);
    try std.testing.expectEqual(@as(f32, 2.0), s.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 3.0), s.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 5.0), s.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 6.0), s.get(&.{ 1, 1 }));

    // Zero-copy mutation: mutating slice modifies parent
    s.set(&.{ 0, 0 }, 99.0);
    try std.testing.expectEqual(@as(f32, 99.0), t.get(&.{ 0, 1 }));
    s.set(&.{ 0, 0 }, 2.0); // restore

    // Contiguous copy of non-contiguous slice
    const c_contig = try s.contiguous(allocator);
    defer free(allocator, c_contig);
    try std.testing.expect(!c_contig.is_view);
    try std.testing.expect(c_contig.isContiguous());
    try std.testing.expectEqual(@as(f32, 2.0), c_contig.data[0]);
    try std.testing.expectEqual(@as(f32, 3.0), c_contig.data[1]);
    try std.testing.expectEqual(@as(f32, 5.0), c_contig.data[2]);
    try std.testing.expectEqual(@as(f32, 6.0), c_contig.data[3]);

    // Slice error conditions
    try std.testing.expectError(error.DimensionOutOfBounds, t.slice(&.{ .{}, .{}, .{} }, allocator));
    try std.testing.expectError(error.IndexOutOfBounds, t.slice(&.{ .{ .start = 10 } }, allocator));
    try std.testing.expectError(error.InvalidSliceRange, t.slice(&.{ .{ .start = 2, .end = 1 } }, allocator));
    try std.testing.expectError(error.InvalidStep, t.slice(&.{ .{ .step = 0 } }, allocator));

    // 2. clip and clip_
    var t_clip = try array(allocator, &.{4}, &[_]f32{ -5.0, 0.5, 3.0, 10.0 });
    defer free(allocator, t_clip);
    const clipped = try clip(t_clip, 0.0, 5.0, allocator);
    defer free(allocator, clipped);
    try std.testing.expectEqual(@as(f32, 0.0), clipped.data[0]);
    try std.testing.expectEqual(@as(f32, 0.5), clipped.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), clipped.data[2]);
    try std.testing.expectEqual(@as(f32, 5.0), clipped.data[3]);

    _ = try t_clip.clip_(0.0, 5.0);
    try std.testing.expectEqual(@as(f32, 0.0), t_clip.data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), t_clip.data[3]);
    try std.testing.expectError(error.InvalidRange, t_clip.clip(5.0, 2.0, allocator));

    // 3. sort and argsort
    const t_unsorted = try array(allocator, &.{4}, &[_]f32{ 3.0, 1.0, 4.0, 2.0 });
    defer free(allocator, t_unsorted);

    const t_sorted = try sort(t_unsorted, 0, true, allocator);
    defer free(allocator, t_sorted);
    try std.testing.expectEqual(@as(f32, 1.0), t_sorted.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), t_sorted.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), t_sorted.data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), t_sorted.data[3]);

    const t_desc = try sort(t_unsorted, 0, false, allocator);
    defer free(allocator, t_desc);
    try std.testing.expectEqual(@as(f32, 4.0), t_desc.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), t_desc.data[3]);

    const t_idxs = try argsort(t_unsorted, 0, true, allocator);
    defer t_idxs.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), t_idxs.data[0]);
    try std.testing.expectEqual(@as(usize, 3), t_idxs.data[1]);
    try std.testing.expectEqual(@as(usize, 0), t_idxs.data[2]);
    try std.testing.expectEqual(@as(usize, 2), t_idxs.data[3]);

    // 4. nonzero
    const t_sparse = try array(allocator, &.{ 2, 3 }, &[_]f32{ 0.0, 5.0, 0.0, 1.0, 0.0, 2.0 });
    defer free(allocator, t_sparse);
    const nz = try nonzero(t_sparse.*, allocator);
    defer nz.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), nz.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), nz.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 0), nz.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(usize, 0), nz.get(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 2, 0 }));
    try std.testing.expectEqual(@as(usize, 2), nz.get(&.{ 2, 1 }));
}

test "NumPy Tier 1 creation routines: arange, linspace, eye, identity, full" {
    const allocator = std.testing.allocator;

    // 1. arange
    const a1 = try arange(allocator, 5.0, null, null); // 0, 1, 2, 3, 4
    defer free(allocator, a1);
    try std.testing.expectEqual(@as(usize, 5), a1.shape.dims[0]);
    try std.testing.expectEqual(@as(f32, 0.0), a1.data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), a1.data[4]);

    const a2 = try arange(allocator, 2.0, 7.0, 1.5); // 2.0, 3.5, 5.0, 6.5
    defer free(allocator, a2);
    try std.testing.expectEqual(@as(usize, 4), a2.shape.dims[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a2.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), a2.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), a2.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), a2.data[3], 1e-5);

    // 2. linspace
    const l1 = try linspace(allocator, 0.0, 1.0, 5); // 0.0, 0.25, 0.5, 0.75, 1.0
    defer free(allocator, l1);
    try std.testing.expectEqual(@as(usize, 5), l1.shape.dims[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), l1.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), l1.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), l1.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), l1.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l1.data[4], 1e-5);

    // 3. eye and identity
    const e1 = try eye(allocator, 3, null, null); // 3x3 identity
    defer free(allocator, e1);
    try std.testing.expectEqual(@as(usize, 3), e1.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), e1.shape.dims[1]);
    try std.testing.expectEqual(@as(f32, 1.0), e1.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 0.0), e1.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), e1.get(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), e1.get(&.{ 2, 2 }));

    const e2 = try eye(allocator, 2, 4, 1); // 2x4 with diagonal offset +1
    defer free(allocator, e2);
    try std.testing.expectEqual(@as(f32, 1.0), e2.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), e2.get(&.{ 1, 2 }));
    try std.testing.expectEqual(@as(f32, 0.0), e2.get(&.{ 0, 0 }));

    const id = try identity(allocator, 2);
    defer free(allocator, id);
    try std.testing.expectEqual(@as(f32, 1.0), id.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 0.0), id.get(&.{ 0, 1 }));

    // 4. full
    const f1 = try full(allocator, &.{ 2, 2 }, 3.14);
    defer free(allocator, f1);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), f1.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), f1.data[3], 1e-5);
}

test "NumPy Tier 1 dimension operations: stack, repeat, tile" {
    const allocator = std.testing.allocator;

    // 1. stack
    const t1 = try array(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer free(allocator, t1);
    const t2 = try array(allocator, &.{ 2, 3 }, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    defer free(allocator, t2);

    // Stack along axis 0 -> [2, 2, 3]
    const s0 = try stack(allocator, &.{ t1, t2 }, 0);
    defer free(allocator, s0);
    try std.testing.expectEqual(@as(usize, 3), s0.shape.len);
    try std.testing.expectEqual(@as(usize, 2), s0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s0.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), s0.shape.dims[2]);
    try std.testing.expectEqual(@as(f32, 1.0), s0.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), s0.data[6]);

    // Stack along axis 1 -> [2, 2, 3]
    const s1 = try stack(allocator, &.{ t1, t2 }, 1);
    defer free(allocator, s1);
    try std.testing.expectEqual(@as(usize, 2), s1.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s1.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), s1.shape.dims[2]);

    // 2. repeat
    const r_flat = try repeat(t1, 2, null, allocator); // flatten then repeat: [1, 1, 2, 2, ...]
    defer free(allocator, r_flat);
    try std.testing.expectEqual(@as(usize, 12), r_flat.shape.dims[0]);
    try std.testing.expectEqual(@as(f32, 1.0), r_flat.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), r_flat.data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), r_flat.data[2]);
    try std.testing.expectEqual(@as(f32, 2.0), r_flat.data[3]);

    // Repeat along axis 0 -> [4, 3]
    const r_ax0 = try t1.repeat(2, 0, allocator);
    defer free(allocator, r_ax0);
    try std.testing.expectEqual(@as(usize, 4), r_ax0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), r_ax0.shape.dims[1]);
    // Row 0 and Row 1 are both [1, 2, 3]
    try std.testing.expectEqual(@as(f32, 1.0), r_ax0.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), r_ax0.get(&.{ 1, 0 }));
    // Row 2 and Row 3 are both [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), r_ax0.get(&.{ 2, 0 }));
    try std.testing.expectEqual(@as(f32, 4.0), r_ax0.get(&.{ 3, 0 }));

    // 3. tile
    const tiled = try t1.tile(&.{ 2, 1 }, allocator); // [4, 3]
    defer free(allocator, tiled);
    try std.testing.expectEqual(@as(usize, 4), tiled.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), tiled.shape.dims[1]);
    try std.testing.expectEqual(@as(f32, 1.0), tiled.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 4.0), tiled.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), tiled.get(&.{ 2, 0 }));
    try std.testing.expectEqual(@as(f32, 4.0), tiled.get(&.{ 3, 0 }));
}

test "NumPy Tier 1 element-wise ufuncs: sqrt, exp, log, abs" {
    const allocator = std.testing.allocator;

    const t = try array(allocator, &.{4}, &.{ -4.0, 0.0, 4.0, 9.0 });
    defer free(allocator, t);

    // 1. abs
    const t_abs = try abs(t, allocator);
    defer free(allocator, t_abs);
    try std.testing.expectEqual(@as(f32, 4.0), t_abs.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), t_abs.data[1]);
    try std.testing.expectEqual(@as(f32, 4.0), t_abs.data[2]);
    try std.testing.expectEqual(@as(f32, 9.0), t_abs.data[3]);

    // 2. sqrt
    const t_pos = try array(allocator, &.{3}, &.{ 1.0, 4.0, 9.0 });
    defer free(allocator, t_pos);
    const t_sqrt = try sqrt(t_pos, allocator);
    defer free(allocator, t_sqrt);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), t_sqrt.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), t_sqrt.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), t_sqrt.data[2], 1e-5);

    // 3. exp and log
    const t_zero = try zeros(allocator, &.{1});
    defer free(allocator, t_zero);
    const t_exp = try exp(t_zero, allocator); // exp(0) = 1
    defer free(allocator, t_exp);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), t_exp.data[0], 1e-5);

    const t_log = try log(t_exp, allocator); // log(1) = 0
    defer free(allocator, t_log);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t_log.data[0], 1e-5);
}

test "StaticTensor compile-time shaped arithmetic, matmul, transpose, reshape, and dynamic conversion" {
    const allocator = std.testing.allocator;
    const StaticTensor = @import("static.zig").StaticTensor;

    const A = StaticTensor(f32, &.{ 2, 3 }).fromSlice(&.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    const B = StaticTensor(f32, &.{ 3, 2 }).fromSlice(&.{
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0,
    });

    const C = A.matmul(B);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(C).shape[0]);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(C).shape[1]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31.0, 19.0, 85.0, 55.0 }, &C.data);

    const AT = A.transpose();
    try std.testing.expectEqual(@as(usize, 3), @TypeOf(AT).shape[0]);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(AT).shape[1]);
    try std.testing.expectEqual(@as(f32, 4.0), AT.get(.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 3.0), AT.get(.{ 2, 0 }));

    const flat = A.reshape(.{6});
    try std.testing.expectEqual(@as(usize, 6), @TypeOf(flat).numel());
    try std.testing.expectEqual(@as(f32, 6.0), flat.get(.{5}));

    const sum_ab = C.add(StaticTensor(f32, .{ 2, 2 }).ones());
    try std.testing.expectEqualSlices(f32, &[_]f32{ 32.0, 20.0, 86.0, 56.0 }, &sum_ab.data);

    const dyn_t = try C.toDynamic(allocator);
    defer free(allocator, dyn_t);
    try std.testing.expectEqual(@as(usize, 2), dyn_t.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), dyn_t.shape.dims[1]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31.0, 19.0, 85.0, 55.0 }, dyn_t.data);
}

test "Tensor slice views and non-contiguous inputs to ops" {
    const allocator = std.testing.allocator;

    // Create a 4x4 matrix
    const parent = try zeros(allocator, &.{ 4, 4 });
    defer free(allocator, parent);
    for (parent.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) - 5.0; // some negative, some positive
    }

    // Slice rows 1..3 (2 rows, 4 cols)
    // In this view, self.data.len = 16 - 4 = 12, but shape.numel() = 8!
    const s = try parent.slice(&.{ .{ .start = 1, .end = 3 } }, allocator);
    defer free(allocator, s);
    try std.testing.expectEqual(@as(usize, 8), s.shape.numel());
    try std.testing.expect(s.data.len > s.shape.numel());

    // Test relu on slice view
    const r = try s.relu(allocator);
    defer free(allocator, r);
    try std.testing.expectEqual(@as(usize, 8), r.shape.numel());
    try std.testing.expectEqual(@as(usize, 8), r.data.len);

    // Test mulScalar on slice view
    const m = try s.mulScalar(2.0, allocator);
    defer free(allocator, m);
    try std.testing.expectEqual(@as(usize, 8), m.shape.numel());

    // Test softmax on slice view
    const sm = try s.softmax(allocator);
    defer free(allocator, sm);
    try std.testing.expectEqual(@as(usize, 8), sm.shape.numel());

    // Test matmul with transposed (non-contiguous) matrix
    // A: 2x3, B: 3x2 (transposed from 2x3)
    const A = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, A);
    @memcpy(A.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    const B_orig = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, B_orig);
    @memcpy(B_orig.data, &[_]f32{ 7.0, 9.0, 2.0, 8.0, 1.0, 3.0 });

    const B_T = try B_orig.transposeView(0, 1, allocator);
    defer free(allocator, B_T);
    try std.testing.expect(!B_T.isContiguous());

    // A is 2x3, B_T is 3x2
    const C = try A.matmul(B_T, allocator);
    defer free(allocator, C);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31.0, 19.0, 85.0, 55.0 }, C.data);

    // Test split on non-contiguous tensor
    const splits = try split(allocator, B_T, 3, 0); // split 3x2 into three 1x2 tensors
    defer {
        for (splits) |sp| free(allocator, sp);
        allocator.free(splits);
    }
    try std.testing.expectEqual(@as(usize, 3), splits.len);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7.0, 8.0 }, splits[0].data);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 1.0 }, splits[1].data);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 3.0 }, splits[2].data);
}

test "Tensor linear algebra: Householder QR decomposition" {
    const allocator = std.testing.allocator;

    // 1. Tall rectangular matrix (4x3)
    const A = try zeros(allocator, &.{ 4, 3 });
    defer free(allocator, A);
    @memcpy(A.data, &[_]f32{
        1.0, 2.0, 4.0,
        3.0, 8.0, 14.0,
        2.0, 6.0, 13.0,
        1.0, 4.0, 7.0,
    });

    const qr_res = try A.qr(allocator);
    defer qr_res.deinit(allocator);

    // Q: [4, 3], R: [3, 3]
    try std.testing.expectEqual(@as(usize, 2), qr_res.Q.shape.len);
    try std.testing.expectEqual(@as(usize, 4), qr_res.Q.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), qr_res.Q.shape.dims[1]);

    try std.testing.expectEqual(@as(usize, 2), qr_res.R.shape.len);
    try std.testing.expectEqual(@as(usize, 3), qr_res.R.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), qr_res.R.shape.dims[1]);

    // Check R is upper triangular with non-negative diagonal
    for (0..3) |i| {
        try std.testing.expect(qr_res.R.data[i * 3 + i] >= -1e-6);
        for (0..i) |j| {
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), qr_res.R.data[i * 3 + j], 1e-4);
        }
    }

    // Check Q^T * Q = I_3
    const QT = try qr_res.Q.transpose(0, 1, allocator);
    defer free(allocator, QT);
    const QTQ = try QT.matmul(qr_res.Q, allocator);
    defer free(allocator, QTQ);

    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, QTQ.data[i * 3 + j], 1e-4);
        }
    }

    // Check A_rec = Q * R == A
    const A_rec = try qr_res.Q.matmul(qr_res.R, allocator);
    defer free(allocator, A_rec);
    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(A.data[i], A_rec.data[i], 1e-4);
    }
}

test "Tensor linear algebra: Jacobi real symmetric eigenvalue decomposition (symeig)" {
    const allocator = std.testing.allocator;

    // Real symmetric matrix 3x3
    const A = try zeros(allocator, &.{ 3, 3 });
    defer free(allocator, A);
    @memcpy(A.data, &[_]f32{
        4.0,  1.0, -2.0,
        1.0,  2.0,  0.0,
       -2.0,  0.0,  3.0,
    });

    const res = try A.symeig(allocator);
    defer res.deinit(allocator);

    // Eigenvalues should be 1D [3] and sorted in ascending order
    try std.testing.expectEqual(@as(usize, 1), res.eigenvalues.shape.len);
    try std.testing.expectEqual(@as(usize, 3), res.eigenvalues.shape.dims[0]);
    try std.testing.expect(res.eigenvalues.data[0] <= res.eigenvalues.data[1] + 1e-6);
    try std.testing.expect(res.eigenvalues.data[1] <= res.eigenvalues.data[2] + 1e-6);

    // Eigenvectors should be [3, 3] and orthonormal: V^T * V = I_3
    const VT = try res.eigenvectors.transpose(0, 1, allocator);
    defer free(allocator, VT);
    const VTV = try VT.matmul(res.eigenvectors, allocator);
    defer free(allocator, VTV);

    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, VTV.data[i * 3 + j], 1e-4);
        }
    }

    // Check A * v[:, i] = lambda_i * v[:, i]
    const AV = try A.matmul(res.eigenvectors, allocator);
    defer free(allocator, AV);
    for (0..3) |col| {
        const lambda = res.eigenvalues.data[col];
        for (0..3) |row| {
            const expected = lambda * res.eigenvectors.data[row * 3 + col];
            try std.testing.expectApproxEqAbs(expected, AV.data[row * 3 + col], 1e-4);
        }
    }
}

test "Tensor linear algebra: One-Sided Jacobi SVD decomposition" {
    const allocator = std.testing.allocator;

    // 1. Tall matrix 4x2
    {
        const A = try zeros(allocator, &.{ 4, 2 });
        defer free(allocator, A);
        @memcpy(A.data, &[_]f32{
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        });

        const res = try A.svd(allocator);
        defer res.deinit(allocator);

        // U: [4, 2], S: [2], Vt: [2, 2]
        try std.testing.expectEqual(@as(usize, 4), res.U.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), res.U.shape.dims[1]);
        try std.testing.expectEqual(@as(usize, 2), res.S.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), res.Vt.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), res.Vt.shape.dims[1]);

        // S non-negative and descending
        try std.testing.expect(res.S.data[0] >= res.S.data[1]);
        try std.testing.expect(res.S.data[1] >= 0.0);

        // U^T * U = I_2
        const UT = try res.U.transpose(0, 1, allocator);
        defer free(allocator, UT);
        const UTU = try UT.matmul(res.U, allocator);
        defer free(allocator, UTU);
        for (0..2) |i| {
            for (0..2) |j| {
                const exp_val: f32 = if (i == j) 1.0 else 0.0;
                try std.testing.expectApproxEqAbs(exp_val, UTU.data[i * 2 + j], 1e-4);
            }
        }

        // Reconstruct A_rec = U * diag(S) * Vt
        const US = try zeros(allocator, &.{ 4, 2 });
        defer free(allocator, US);
        for (0..4) |r| {
            for (0..2) |c| {
                US.data[r * 2 + c] = res.U.data[r * 2 + c] * res.S.data[c];
            }
        }
        const A_rec = try US.matmul(res.Vt, allocator);
        defer free(allocator, A_rec);

        for (0..8) |i| {
            try std.testing.expectApproxEqAbs(A.data[i], A_rec.data[i], 1e-3);
        }
    }

    // 2. Wide matrix 2x3
    {
        const B = try zeros(allocator, &.{ 2, 3 });
        defer free(allocator, B);
        @memcpy(B.data, &[_]f32{
            3.0, 2.0, 2.0,
            2.0, 3.0, -2.0,
        });

        const res = try B.svd(allocator);
        defer res.deinit(allocator);

        // U: [2, 2], S: [2], Vt: [2, 3]
        try std.testing.expectEqual(@as(usize, 2), res.U.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), res.U.shape.dims[1]);
        try std.testing.expectEqual(@as(usize, 2), res.S.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), res.Vt.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), res.Vt.shape.dims[1]);

        try std.testing.expect(res.S.data[0] >= res.S.data[1]);
        try std.testing.expect(res.S.data[1] >= 0.0);

        // Reconstruct B_rec = U * diag(S) * Vt
        const US = try zeros(allocator, &.{ 2, 2 });
        defer free(allocator, US);
        for (0..2) |r| {
            for (0..2) |c| {
                US.data[r * 2 + c] = res.U.data[r * 2 + c] * res.S.data[c];
            }
        }
        const B_rec = try US.matmul(res.Vt, allocator);
        defer free(allocator, B_rec);

        for (0..6) |i| {
            try std.testing.expectApproxEqAbs(B.data[i], B_rec.data[i], 1e-3);
        }
    }
}

test "Tensor to MLX Array bridge zero-copy and conversion" {
    const backend = @import("../backend.zig");
    if (!backend.mlx.has_mlx) return;

    const allocator = std.testing.allocator;

    var t = try tensor.zeros(allocator, &.{ 2, 3 });
    defer tensor.free(allocator, t);
    for (t.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i + 1));
    }

    // 1. 验证常规复制转换为 MLX 数组 (toMlxArray)
    var arr = try t.toMlxArray();
    defer arr.deinit();
    try std.testing.expectEqual(@as(usize, 6), arr.size());
    try std.testing.expectEqual(@as(usize, 2), arr.ndim());
    try std.testing.expectEqualSlices(f32, t.data, arr.dataFloat32());

    // 2. 验证从 MLX 数组新建分配张量 (fromMlxArray)
    const t_restored = try Tensor.fromMlxArray(allocator, arr);
    defer tensor.free(allocator, t_restored);
    try std.testing.expectEqualSlices(usize, t.shape.dims[0..t.shape.len], t_restored.shape.dims[0..t_restored.shape.len]);
    try std.testing.expectEqualSlices(f32, t.data, t_restored.data);

    // 3. 验证零拷贝映射与回写 (toMlxArrayZeroCopy & copyFromMlxArray)
    var arr_zc = try t.toMlxArrayZeroCopy();
    defer arr_zc.deinit();
    try std.testing.expectEqual(@as(usize, 6), arr_zc.size());

    // 对原张量底层数据原地修改，统一内存下由 MLX 阵列可见
    t.data[0] = 999.0;
    try std.testing.expectApproxEqAbs(@as(f32, 999.0), arr_zc.dataFloat32()[0], 1e-4);

    // 验证 copyFromMlxArray 与 backend.mlx.Array.copyToTensor
    t_restored.data[0] = 0.0;
    try t_restored.copyFromMlxArray(arr_zc);
    try std.testing.expectApproxEqAbs(@as(f32, 999.0), t_restored.data[0], 1e-4);

    // 4. 验证直接从 backend.mlx.Array 静态方法调用 (Array.fromTensor / Array.fromTensorZeroCopy / arr.copyToTensor)
    var arr_direct = try backend.mlx.Array.fromTensorZeroCopy(t);
    defer arr_direct.deinit();
    try std.testing.expectEqual(@as(usize, 6), arr_direct.size());
    t.data[1] = 777.0;
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), arr_direct.dataFloat32()[1], 1e-4);
    try arr_direct.copyToTensor(t_restored);
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), t_restored.data[1], 1e-4);
}


// ============================================================================
// 审计回归测试 (Audit Regression Tests)：内存安全、数值稳定性、视图语义与内核等价性
// ============================================================================

/// 测试辅助：用确定性伪随机数填充张量
fn fillDeterministic(t: *Tensor, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (t.data) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

/// 测试辅助：按逻辑坐标读取任意跨步张量的元素
fn atCoord(t: *const Tensor, coord: []const usize) f32 {
    var idx: usize = 0;
    for (coord, 0..) |c_, d| idx += c_ * t.strides.dims[d];
    return t.data[idx];
}

fn allocFailConcatStackSplit(allocator: std.mem.Allocator) !void {
    const X = try array(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer free(allocator, X);
    const XT = try X.transposeView(0, 1, allocator);
    defer free(allocator, XT);

    const c_ = try concat(allocator, &.{ X, XT, X }, 1);
    defer free(allocator, c_);
    const st = try stack(allocator, &.{ X, XT }, 0);
    defer free(allocator, st);
    const parts = try split(allocator, XT, 2, 1);
    defer {
        for (parts) |p| free(allocator, p);
        allocator.free(parts);
    }
    const cl = try X.clone(allocator);
    defer free(allocator, cl);
    const rs = try XT.reshape(&.{4}, allocator);
    defer free(allocator, rs);
    const tr = try X.transpose(0, 1, allocator);
    defer free(allocator, tr);
    const rp = try XT.repeat(2, null, allocator);
    defer free(allocator, rp);
    const tl = try XT.tile(&.{ 2, 1 }, allocator);
    defer free(allocator, tl);
}

test "audit: concat/stack/split/clone/reshape/repeat/tile free everything on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocFailConcatStackSplit, .{});
}

fn allocFailLosses(allocator: std.mem.Allocator) !void {
    const X = try array(allocator, &.{ 2, 2 }, &.{ 0.2, 0.8, 0.6, 0.4 });
    defer free(allocator, X);
    const XT = try X.transposeView(0, 1, allocator);
    defer free(allocator, XT);
    const Y = try array(allocator, &.{ 2, 2 }, &.{ 0, 1, 1, 0 });
    defer free(allocator, Y);

    const l1 = try XT.bceWithLogitsLoss(Y, allocator);
    defer free(allocator, l1);
    const l2 = try XT.bceLoss(Y, 1e-7, allocator);
    defer free(allocator, l2);
    const l3 = try XT.softmaxCrossEntropy(&.{ 0, 1 }, allocator);
    defer free(allocator, l3);
    const l4 = try XT.mseLoss(Y, allocator);
    defer free(allocator, l4);
    const l5 = try XT.l2Loss(0.1, allocator);
    defer free(allocator, l5);
    const l6 = try XT.l1Loss(0.1, allocator);
    defer free(allocator, l6);
}

test "audit: loss kernels free everything on allocation failure and reject bad labels without leaking" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocFailLosses, .{});
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, X);
    try std.testing.expectError(error.IndexOutOfBounds, X.softmaxCrossEntropy(&.{ 0, 3 }, allocator));
    const Y = try zeros(allocator, &.{5});
    defer free(allocator, Y);
    try std.testing.expectError(error.ShapeMismatch, X.bceLoss(Y, 1e-7, allocator));
    try std.testing.expectError(error.ShapeMismatch, X.bceWithLogitsLoss(Y, allocator));
}

fn allocFailLinalg(allocator: std.mem.Allocator) !void {
    const A = try array(allocator, &.{ 3, 2 }, &.{ 2, 1, 1, 3, 0.5, -1 });
    defer free(allocator, A);
    const q = try A.qr(allocator);
    defer q.deinit(allocator);
    const s = try A.svd(allocator);
    defer s.deinit(allocator);
    const At = try A.transpose(0, 1, allocator);
    defer free(allocator, At);
    const s2 = try At.svd(allocator);
    defer s2.deinit(allocator);
    const S = try array(allocator, &.{ 2, 2 }, &.{ 2, 1, 1, 3 });
    defer free(allocator, S);
    const e = try S.symeig(allocator);
    defer e.deinit(allocator);
}

test "audit: qr/svd/symeig free partial results on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocFailLinalg, .{});
}

test "audit: softmaxCrossEntropy uses log-sum-exp instead of clipping the probability" {
    const allocator = std.testing.allocator;
    const X = try array(allocator, &.{ 2, 2 }, &.{ 0.0, 200.0, 1.0, 2.0 });
    defer free(allocator, X);
    const loss = try X.softmaxCrossEntropy(&.{ 0, 1 }, allocator);
    defer free(allocator, loss);
    // 样本 0：log(1 + e^-200) + 200 - 0 = 200；样本 1：log(1 + e^-1) = 0.3132617
    const expected = (200.0 + @log(1.0 + @exp(@as(f32, -1.0)))) / 2.0;
    try std.testing.expectApproxEqRel(expected, loss.data[0], 1e-5);
}

test "audit: bf16 conversion keeps NaN as NaN" {
    const nan_payload: f32 = @bitCast(@as(u32, 0x7F800001));
    try std.testing.expect(std.math.isNan(nan_payload));
    try std.testing.expect(std.math.isNan(bf16.fromF32(nan_payload).toF32()));
    const neg_nan: f32 = @bitCast(@as(u32, 0xFF800001));
    try std.testing.expect(std.math.isNan(bf16.fromF32(neg_nan).toF32()));
    try std.testing.expect(std.math.isInf(bf16.fromF32(std.math.inf(f32)).toF32()));
    try std.testing.expectEqual(@as(f32, 1.0), bf16.fromF32(1.0).toF32());
}

test "audit: rank-0 scalar tensors have one element" {
    const allocator = std.testing.allocator;
    const s = try full(allocator, &.{}, 3.5);
    defer free(allocator, s);
    try std.testing.expectEqual(@as(usize, 1), s.shape.numel());
    try std.testing.expectEqual(@as(usize, 1), s.data.len);
    const total = try s.sum(null, false, allocator);
    defer free(allocator, total);
    try std.testing.expectEqual(@as(f32, 3.5), total.data[0]);
    const doubled = try s.mulScalar(2.0, allocator);
    defer free(allocator, doubled);
    try std.testing.expectEqual(@as(f32, 7.0), doubled.data[0]);
}

test "audit: empty slices at the end of every dimension do not index past the buffer" {
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, X);
    const v = try X.slice(&.{ .{ .start = 2 }, .{ .start = 3 } }, allocator);
    defer free(allocator, v);
    try std.testing.expectEqual(@as(usize, 0), v.shape.numel());
    const c_ = try v.contiguous(allocator);
    defer free(allocator, c_);
    try std.testing.expectEqual(@as(usize, 0), c_.data.len);

    const G = try GenericTensor(i32).zeros(allocator, &.{ 2, 3 });
    defer G.deinit(allocator);
    const gv = try G.slice(&.{ .{ .start = 2 }, .{ .start = 3 } }, allocator);
    defer gv.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), gv.numel());
}

test "audit: repeat without axis only repeats the visible elements of a view" {
    const allocator = std.testing.allocator;
    const X = try array(allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, X);
    const v = try X.slice(&.{.{ .start = 1, .end = 2 }}, allocator);
    defer free(allocator, v);
    const r = try v.repeat(2, null, allocator);
    defer free(allocator, r);
    try std.testing.expectEqualSlices(f32, &.{ 3, 3, 4, 4 }, r.data);

    const r0 = try X.repeat(0, 1, allocator);
    defer free(allocator, r0);
    try std.testing.expectEqual(@as(usize, 3), r0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 0), r0.shape.dims[1]);
}

test "audit: in-place ops on views only touch the visible elements" {
    const allocator = std.testing.allocator;
    const X = try array(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, X);

    // 第 1 列 (跨步视图) 乘以 10
    const col = try X.slice(&.{ .{}, .{ .start = 1, .end = 2 } }, allocator);
    defer free(allocator, col);
    _ = col.mulScalar_(10.0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 20, 3, 4, 50, 6 }, X.data);
    _ = col.addScalar_(1.0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 21, 3, 4, 51, 6 }, X.data);
    _ = try col.clip_(0.0, 30.0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 21, 3, 4, 30, 6 }, X.data);

    // 第 0 行 (连续视图，底层 data 更长) 原地加上另一张量
    const row = try X.slice(&.{.{ .start = 0, .end = 1 }}, allocator);
    defer free(allocator, row);
    const other = try array(allocator, &.{ 1, 3 }, &.{ 100, 100, 100 });
    defer free(allocator, other);
    _ = try row.add_(other);
    try std.testing.expectEqualSlices(f32, &.{ 101, 121, 103, 4, 30, 6 }, X.data);
    const bad = try zeros(allocator, &.{3});
    defer free(allocator, bad);
    try std.testing.expectError(error.ShapeMismatch, row.add_(bad));
}

test "audit: contiguous returns a plain data copy without gradient buffers" {
    const allocator = std.testing.allocator;
    const X = try array(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer free(allocator, X);
    X.grad = try allocator.alloc(f32, 4);
    @memset(X.grad, 7.0);
    X.requires_grad = true;

    const c_ = try X.contiguous(allocator);
    defer free(allocator, c_);
    try std.testing.expect(!c_.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), c_.grad.len);
    try std.testing.expect(c_.creator == null);
    try std.testing.expectEqualSlices(f32, X.data, c_.data);

    const XT = try X.transposeView(0, 1, allocator);
    defer free(allocator, XT);
    const ct = try XT.contiguous(allocator);
    defer free(allocator, ct);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 2, 4 }, ct.data);
}

/// 测试辅助：朴素多维广播二元运算参考实现
fn referenceBroadcast(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor, comptime op: fn (f32, f32) f32) !*Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape);
    const out = try zeros(allocator, out_shape.dims[0..out_shape.len]);
    var coord: [8]usize = @splat(0);
    for (out.data) |*o| {
        var ca: [8]usize = @splat(0);
        var cb: [8]usize = @splat(0);
        for (0..a.shape.len) |d| {
            const od = out_shape.len - a.shape.len + d;
            ca[d] = if (a.shape.dims[d] == 1) 0 else coord[od];
        }
        for (0..b.shape.len) |d| {
            const od = out_shape.len - b.shape.len + d;
            cb[d] = if (b.shape.dims[d] == 1) 0 else coord[od];
        }
        o.* = op(atCoord(a, ca[0..a.shape.len]), atCoord(b, cb[0..b.shape.len]));
        out_shape.incrementCoord(&coord);
    }
    return out;
}

fn refSub(x: f32, y: f32) f32 {
    return x - y;
}

test "audit: broadcast binary ops match a naive reference for many layouts" {
    const allocator = std.testing.allocator;
    const Case = struct { a: []const usize, b: []const usize, transpose_a: bool = false };
    const cases = [_]Case{
        .{ .a = &.{ 4, 5 }, .b = &.{5} },
        .{ .a = &.{ 4, 5 }, .b = &.{ 4, 1 } },
        .{ .a = &.{ 4, 1 }, .b = &.{ 1, 5 } },
        .{ .a = &.{ 2, 3, 4 }, .b = &.{ 3, 1 } },
        .{ .a = &.{ 2, 1, 4 }, .b = &.{ 2, 3, 1 } },
        .{ .a = &.{ 3, 4 }, .b = &.{ 4, 3 }, .transpose_a = true },
        .{ .a = &.{ 5, 4 }, .b = &.{ 1, 5 }, .transpose_a = true },
        .{ .a = &.{1}, .b = &.{ 3, 2 } },
        .{ .a = &.{ 2, 0, 3 }, .b = &.{3} },
    };
    for (cases, 0..) |cs, ci| {
        const a_base = try zeros(allocator, cs.a);
        defer free(allocator, a_base);
        fillDeterministic(a_base, 10 + ci);
        const a = if (cs.transpose_a) try a_base.transposeView(0, 1, allocator) else a_base;
        defer if (cs.transpose_a) free(allocator, a);
        const b = try zeros(allocator, cs.b);
        defer free(allocator, b);
        fillDeterministic(b, 100 + ci);

        const got = try a.sub(b, allocator);
        defer free(allocator, got);
        const want = try referenceBroadcast(allocator, a, b, refSub);
        defer free(allocator, want);
        try std.testing.expect(got.shape.eq(want.shape));
        try std.testing.expectEqualSlices(f32, want.data, got.data);
    }
}

test "audit: sum and arg/extremum reductions match a naive reference along every axis" {
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 3, 4, 5 });
    defer free(allocator, X);
    fillDeterministic(X, 42);
    X.data[7] = X.data[27]; // 制造相同数值，验证保留首个下标
    const XT = try X.transposeView(0, 2, allocator);
    defer free(allocator, XT);

    for ([_]*Tensor{ X, XT }) |t| {
        for (0..3) |ax| {
            const s = try t.sum(ax, true, allocator);
            defer free(allocator, s);
            const am = try t.argmax(ax, allocator);
            defer free(allocator, am);
            const an = try t.argmin(ax, allocator);
            defer free(allocator, an);
            const mx = try t.max(ax, allocator);
            defer free(allocator, mx);
            const mn = try t.min(ax, allocator);
            defer free(allocator, mn);

            var coord: [8]usize = @splat(0);
            for (0..s.data.len) |o| {
                var acc: f64 = 0.0;
                var best_max: f32 = -std.math.inf(f32);
                var best_min: f32 = std.math.inf(f32);
                var idx_max: usize = 0;
                var idx_min: usize = 0;
                var cc = coord;
                for (0..t.shape.dims[ax]) |k| {
                    cc[ax] = k;
                    const v = atCoord(t, cc[0..3]);
                    acc += v;
                    if (v > best_max) {
                        best_max = v;
                        idx_max = k;
                    }
                    if (v < best_min) {
                        best_min = v;
                        idx_min = k;
                    }
                }
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), s.data[o], 1e-5);
                try std.testing.expectEqual(@as(f32, @floatFromInt(idx_max)), am.data[o]);
                try std.testing.expectEqual(@as(f32, @floatFromInt(idx_min)), an.data[o]);
                try std.testing.expectEqual(best_max, mx.data[o]);
                try std.testing.expectEqual(best_min, mn.data[o]);
                s.shape.incrementCoord(&coord);
            }
        }
    }

    const v = try X.variance(1, false, 1, allocator);
    defer free(allocator, v);
    try std.testing.expectEqual(@as(usize, 15), v.data.len);
    for (0..3) |i| {
        for (0..5) |k| {
            var m: f64 = 0.0;
            for (0..4) |j| m += X.data[(i * 4 + j) * 5 + k];
            m /= 4.0;
            var ss: f64 = 0.0;
            for (0..4) |j| {
                const d = X.data[(i * 4 + j) * 5 + k] - m;
                ss += d * d;
            }
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(ss / 3.0)), v.data[i * 5 + k], 1e-5);
        }
    }
}

test "audit: batchMatMul handles transposed and strided views" {
    const allocator = std.testing.allocator;
    const A = try zeros(allocator, &.{ 2, 3, 4, 5 });
    defer free(allocator, A);
    fillDeterministic(A, 1);
    const Bs = try zeros(allocator, &.{ 2, 3, 6, 5 });
    defer free(allocator, Bs);
    fillDeterministic(Bs, 2);

    // B^T 视图：[2, 3, 5, 6]，最后两维列优先
    const BT = try Bs.transposeView(2, 3, allocator);
    defer free(allocator, BT);
    const BT_copy = try BT.contiguous(allocator);
    defer free(allocator, BT_copy);

    const want = try A.batchMatMul(BT_copy, allocator);
    defer free(allocator, want);
    const got = try A.batchMatMul(BT, allocator);
    defer free(allocator, got);
    for (want.data, got.data) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // 头维与序列维交换的视图：[2, 4, 3, 5] -> 最后两维为跨步行
    const AT = try A.transposeView(1, 2, allocator);
    defer free(allocator, AT);
    const AT_copy = try AT.contiguous(allocator);
    defer free(allocator, AT_copy);
    const W = try zeros(allocator, &.{ 2, 4, 5, 2 });
    defer free(allocator, W);
    fillDeterministic(W, 3);
    const want2 = try AT_copy.batchMatMul(W, allocator);
    defer free(allocator, want2);
    const got2 = try AT.batchMatMul(W, allocator);
    defer free(allocator, got2);
    for (want2.data, got2.data) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // 逐元素校验一个输出
    var acc: f32 = 0.0;
    for (0..5) |k| acc += A.data[k] * Bs.data[k];
    try std.testing.expectApproxEqAbs(acc, got.data[0], 1e-5);
}

test "audit: repeatKV materializes non-contiguous inputs" {
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 1, 3, 2, 4 });
    defer free(allocator, X);
    fillDeterministic(X, 5);
    const XT = try X.transposeView(1, 2, allocator); // [1, 2, 3, 4]
    defer free(allocator, XT);
    const XT_copy = try XT.contiguous(allocator);
    defer free(allocator, XT_copy);
    const want = try XT_copy.repeatKV(2, allocator);
    defer free(allocator, want);
    const got = try XT.repeatKV(2, allocator);
    defer free(allocator, got);
    try std.testing.expectEqualSlices(f32, want.data, got.data);
}

/// 测试辅助：朴素二维卷积参考实现（一维卷积视为 H = KH = 1）
fn referenceConv2d(x: *const Tensor, w: *const Tensor, bias: ?*const Tensor, stride: usize, padding: usize, out: *const Tensor) !void {
    const N = x.shape.dims[0];
    const C_in = x.shape.dims[1];
    const H = x.shape.dims[2];
    const W = x.shape.dims[3];
    const C_out = w.shape.dims[0];
    const KH = w.shape.dims[2];
    const KW = w.shape.dims[3];
    const H_out = out.shape.dims[2];
    const W_out = out.shape.dims[3];
    for (0..N) |n| for (0..C_out) |co| for (0..H_out) |oh| for (0..W_out) |ow| {
        var acc: f32 = if (bias) |b| b.data[co] else 0.0;
        for (0..C_in) |ci| for (0..KH) |kh| for (0..KW) |kw| {
            const ih = @as(isize, @intCast(oh * stride + kh)) - @as(isize, @intCast(padding));
            const iw = @as(isize, @intCast(ow * stride + kw)) - @as(isize, @intCast(padding));
            if (ih < 0 or iw < 0 or ih >= H or iw >= W) continue;
            acc += atCoord(x, &.{ n, ci, @intCast(ih), @intCast(iw) }) * atCoord(w, &.{ co, ci, kh, kw });
        };
        try std.testing.expectApproxEqAbs(acc, out.data[((n * C_out + co) * H_out + oh) * W_out + ow], 1e-4);
    };
}

/// 测试辅助：朴素二维转置卷积参考实现（散射累加）
fn referenceConvTranspose2d(allocator: std.mem.Allocator, x: *const Tensor, w: *const Tensor, bias: ?*const Tensor, stride: usize, padding: usize, out: *const Tensor) !void {
    const N = x.shape.dims[0];
    const C_in = x.shape.dims[1];
    const H = x.shape.dims[2];
    const W = x.shape.dims[3];
    const C_out = w.shape.dims[1];
    const KH = w.shape.dims[2];
    const KW = w.shape.dims[3];
    const H_out = out.shape.dims[2];
    const W_out = out.shape.dims[3];
    const ref = try allocator.alloc(f32, out.data.len);
    defer allocator.free(ref);
    for (0..N) |n| for (0..C_out) |co| {
        const plane = ref[(n * C_out + co) * H_out * W_out ..][0 .. H_out * W_out];
        @memset(plane, if (bias) |b| b.data[co] else 0.0);
    };
    for (0..N) |n| for (0..C_in) |ci| for (0..H) |h| for (0..W) |wi| for (0..C_out) |co| for (0..KH) |kh| for (0..KW) |kw| {
        const oh = @as(isize, @intCast(h * stride + kh)) - @as(isize, @intCast(padding));
        const ow = @as(isize, @intCast(wi * stride + kw)) - @as(isize, @intCast(padding));
        if (oh < 0 or ow < 0 or oh >= H_out or ow >= W_out) continue;
        const o_idx = ((n * C_out + co) * H_out + @as(usize, @intCast(oh))) * W_out + @as(usize, @intCast(ow));
        ref[o_idx] += atCoord(x, &.{ n, ci, h, wi }) * atCoord(w, &.{ ci, co, kh, kw });
    };
    for (ref, out.data) |r, o| try std.testing.expectApproxEqAbs(r, o, 1e-4);
}

test "audit: im2col conv and GEMM transposed conv match naive references" {
    const allocator = std.testing.allocator;
    const Case = struct { stride: usize, padding: usize, kh: usize, kw: usize, bias: bool, transpose_x: bool };
    const cases = [_]Case{
        .{ .stride = 1, .padding = 0, .kh = 3, .kw = 3, .bias = true, .transpose_x = false },
        .{ .stride = 1, .padding = 1, .kh = 3, .kw = 2, .bias = false, .transpose_x = true },
        .{ .stride = 2, .padding = 1, .kh = 3, .kw = 3, .bias = true, .transpose_x = false },
        .{ .stride = 3, .padding = 2, .kh = 2, .kw = 4, .bias = true, .transpose_x = true },
        .{ .stride = 2, .padding = 0, .kh = 1, .kw = 1, .bias = false, .transpose_x = false },
    };
    for (cases, 0..) |cs, ci| {
        // 输入 [2, 3, 6, 7]（可选：由 [2, 3, 7, 6] 的转置视图得到非连续输入）
        const x_base = if (cs.transpose_x) try zeros(allocator, &.{ 2, 3, 7, 6 }) else try zeros(allocator, &.{ 2, 3, 6, 7 });
        defer free(allocator, x_base);
        fillDeterministic(x_base, 200 + ci);
        const x = if (cs.transpose_x) try x_base.transposeView(2, 3, allocator) else x_base;
        defer if (cs.transpose_x) free(allocator, x);

        const w = try zeros(allocator, &.{ 4, 3, cs.kh, cs.kw });
        defer free(allocator, w);
        fillDeterministic(w, 300 + ci);
        const b = try zeros(allocator, &.{4});
        defer free(allocator, b);
        fillDeterministic(b, 400 + ci);
        const bias: ?*Tensor = if (cs.bias) b else null;

        const y = try x.conv2d(w, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
        defer free(allocator, y);
        try referenceConv2d(x, w, bias, cs.stride, cs.padding, y);

        // 转置卷积权重 [C_in = 3, C_out = 4, KH, KW]
        const wt = try zeros(allocator, &.{ 3, 4, cs.kh, cs.kw });
        defer free(allocator, wt);
        fillDeterministic(wt, 500 + ci);
        if ((6 - 1) * cs.stride + cs.kh > 2 * cs.padding and (7 - 1) * cs.stride + cs.kw > 2 * cs.padding) {
            const yt = try x.convTranspose2d(wt, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
            defer free(allocator, yt);
            try referenceConvTranspose2d(allocator, x, wt, bias, cs.stride, cs.padding, yt);
        }

        // 一维卷积 / 转置卷积：取第 0 行作为 [N, C, L] 输入
        const x1 = try zeros(allocator, &.{ 2, 3, 7 });
        defer free(allocator, x1);
        fillDeterministic(x1, 600 + ci);
        const w1 = try zeros(allocator, &.{ 4, 3, cs.kw });
        defer free(allocator, w1);
        fillDeterministic(w1, 700 + ci);
        const y1 = try x1.conv1d(w1, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
        defer free(allocator, y1);
        const x1_4d = try x1.reshape(&.{ 2, 3, 1, 7 }, allocator);
        defer free(allocator, x1_4d);
        const w1_4d = try w1.reshape(&.{ 4, 3, 1, cs.kw }, allocator);
        defer free(allocator, w1_4d);
        const y1_4d = try y1.reshape(&.{ 2, 4, 1, y1.shape.dims[2] }, allocator);
        defer free(allocator, y1_4d);
        // 一维卷积只在长度方向填充
        const N = 2;
        const L_out = y1.shape.dims[2];
        for (0..N) |n| for (0..4) |co| for (0..L_out) |ol| {
            var acc: f32 = if (bias) |bb| bb.data[co] else 0.0;
            for (0..3) |c_in| for (0..cs.kw) |k| {
                const il = @as(isize, @intCast(ol * cs.stride + k)) - @as(isize, @intCast(cs.padding));
                if (il < 0 or il >= 7) continue;
                acc += x1.data[(n * 3 + c_in) * 7 + @as(usize, @intCast(il))] * w1.data[(co * 3 + c_in) * cs.kw + k];
            };
            try std.testing.expectApproxEqAbs(acc, y1.data[(n * 4 + co) * L_out + ol], 1e-4);
        };

        const wt1 = try zeros(allocator, &.{ 3, 4, cs.kw });
        defer free(allocator, wt1);
        fillDeterministic(wt1, 800 + ci);
        if ((7 - 1) * cs.stride + cs.kw > 2 * cs.padding) {
            const yt1 = try x1.convTranspose1d(wt1, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
            defer free(allocator, yt1);
            const Lt = yt1.shape.dims[2];
            const ref = try allocator.alloc(f32, yt1.data.len);
            defer allocator.free(ref);
            for (0..N) |n| for (0..4) |co| @memset(ref[(n * 4 + co) * Lt ..][0..Lt], if (bias) |bb| bb.data[co] else 0.0);
            for (0..N) |n| for (0..3) |c_in| for (0..7) |l| for (0..4) |co| for (0..cs.kw) |k| {
                const o = @as(isize, @intCast(l * cs.stride + k)) - @as(isize, @intCast(cs.padding));
                if (o < 0 or o >= Lt) continue;
                ref[(n * 4 + co) * Lt + @as(usize, @intCast(o))] += x1.data[(n * 3 + c_in) * 7 + l] * wt1.data[(c_in * 4 + co) * cs.kw + k];
            };
            for (ref, yt1.data) |r, o| try std.testing.expectApproxEqAbs(r, o, 1e-4);
        }
    }
}

test "audit: rope matches the per-element reference formula for both pairing modes" {
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 2, 3, 5, 10 });
    defer free(allocator, X);
    fillDeterministic(X, 77);
    const start_pos: usize = 4;
    inline for (.{ tensor.RopeOptions.Mode.interleaved, tensor.RopeOptions.Mode.split_half }) |mode| {
        const opts = tensor.RopeOptions{ .mode = mode, .rotary_offset = 2, .partial_rotary_factor = 0.75 };
        const Y = try X.rope(start_pos, opts, allocator);
        defer free(allocator, Y);
        const D: usize = 10;
        const rot_dim: usize = D - opts.rotary_offset;
        const half = rot_dim / 2;
        const num_angles: usize = @intFromFloat(@as(f32, @floatFromInt(half)) * opts.partial_rotary_factor);
        for (0..6) |o| for (0..5) |t| {
            const row_in = X.data[(o * 5 + t) * D ..][0..D];
            const row_out = Y.data[(o * 5 + t) * D ..][0..D];
            var expected: [10]f32 = row_in[0..10].*;
            for (0..num_angles) |i| {
                const freq = 1.0 / std.math.pow(f32, opts.base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rot_dim)));
                const theta = @as(f32, @floatFromInt(start_pos + t)) * freq;
                const a_idx = if (mode == .interleaved) opts.rotary_offset + 2 * i else opts.rotary_offset + i;
                const b_idx = if (mode == .interleaved) a_idx + 1 else a_idx + half;
                expected[a_idx] = row_in[a_idx] * @cos(theta) - row_in[b_idx] * @sin(theta);
                expected[b_idx] = row_in[a_idx] * @sin(theta) + row_in[b_idx] * @cos(theta);
            }
            for (expected, row_out) |e, g| try std.testing.expectApproxEqAbs(e, g, 1e-5);
        };
    }
}

test "audit: matmul accepts transposed views without materializing and matches the copied result" {
    const allocator = std.testing.allocator;
    const A_base = try zeros(allocator, &.{ 5, 3 });
    defer free(allocator, A_base);
    fillDeterministic(A_base, 11);
    const B_base = try zeros(allocator, &.{ 4, 5 });
    defer free(allocator, B_base);
    fillDeterministic(B_base, 12);
    const AT = try A_base.transposeView(0, 1, allocator); // [3, 5]
    defer free(allocator, AT);
    const BT = try B_base.transposeView(0, 1, allocator); // [5, 4]
    defer free(allocator, BT);
    const AT_c = try AT.contiguous(allocator);
    defer free(allocator, AT_c);
    const BT_c = try BT.contiguous(allocator);
    defer free(allocator, BT_c);

    const want = try AT_c.matmul(BT_c, allocator);
    defer free(allocator, want);
    const got = try AT.matmul(BT, allocator);
    defer free(allocator, got);
    for (want.data, got.data) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // 行切片视图 (连续但带偏移)
    const rows = try A_base.slice(&.{.{ .start = 1, .end = 4 }}, allocator); // [3, 3]
    defer free(allocator, rows);
    const rows_c = try rows.contiguous(allocator);
    defer free(allocator, rows_c);
    const want2 = try rows_c.matmul(AT_c, allocator);
    defer free(allocator, want2);
    const got2 = try rows.matmul(AT, allocator);
    defer free(allocator, got2);
    for (want2.data, got2.data) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
}

test "audit: applyRoPETensor 4D layout matches the 3D layout" {
    const allocator = std.testing.allocator;
    const T: usize = 4;
    const H: usize = 3;
    const D: usize = 6;
    // 四维 [1, H, T, D] 与三维 [T, H, D] 存放同一组向量
    const x4 = try zeros(allocator, &.{ 1, H, T, D });
    defer free(allocator, x4);
    fillDeterministic(x4, 21);
    const x3 = try zeros(allocator, &.{ T, H, D });
    defer free(allocator, x3);
    for (0..H) |h| for (0..T) |t| {
        @memcpy(x3.data[(t * H + h) * D ..][0..D], x4.data[(h * T + t) * D ..][0..D]);
    };
    tensor.applyRoPETensor(x4, 10000.0);
    tensor.applyRoPETensor(x3, 10000.0);
    for (0..H) |h| for (0..T) |t| {
        try std.testing.expectEqualSlices(f32, x3.data[(t * H + h) * D ..][0..D], x4.data[(h * T + t) * D ..][0..D]);
    };
}

test "audit: solveRidgeAnalytical recovers multi-feature weights" {
    const allocator = std.testing.allocator;
    const N: usize = 40;
    const Dm: usize = 3;
    const true_w = [_]f32{ 1.5, -2.0, 0.5 };
    const true_b: f32 = 0.75;
    var x: [N * Dm]f32 = undefined;
    var y: [N]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(9);
    for (0..N) |i| {
        var acc: f32 = true_b;
        for (0..Dm) |j| {
            x[i * Dm + j] = prng.random().float(f32) * 4.0 - 2.0;
            acc += true_w[j] * x[i * Dm + j];
        }
        y[i] = acc;
    }
    var w: [Dm]f32 = undefined;
    var b: f32 = 0.0;
    try tensor.solveRidgeAnalytical(allocator, &x, &y, N, Dm, 0.0, &w, &b);
    for (true_w, w) |tw, ww| try std.testing.expectApproxEqAbs(tw, ww, 1e-3);
    try std.testing.expectApproxEqAbs(true_b, b, 1e-3);
    try std.testing.expectError(error.EmptyInputs, tensor.solveRidgeAnalytical(allocator, x[0..0], y[0..0], 0, Dm, 0.0, &w, &b));
}

test "audit: softmax rows sum to one, are shift invariant and give exact zeros for -inf logits" {
    const allocator = std.testing.allocator;
    const X = try zeros(allocator, &.{ 3, 13 });
    defer free(allocator, X);
    fillDeterministic(X, 31);
    X.data[13 + 5] = -std.math.inf(f32);
    X.data[26] = 1000.0;
    const P = try X.softmax(allocator);
    defer free(allocator, P);
    for (0..3) |r| {
        const row = P.data[r * 13 ..][0..13];
        var total: f32 = 0.0;
        for (row) |v| total += v;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
        for (row, X.data[r * 13 ..][0..13]) |p, xv| {
            var denom: f64 = 0.0;
            var mx: f32 = -std.math.inf(f32);
            for (X.data[r * 13 ..][0..13]) |v| mx = @max(mx, v);
            for (X.data[r * 13 ..][0..13]) |v| denom += @exp(@as(f64, v - mx));
            const want: f32 = @floatCast(@exp(@as(f64, xv - mx)) / denom);
            try std.testing.expectApproxEqAbs(want, p, 1e-6);
        }
    }
    try std.testing.expectEqual(@as(f32, 0.0), P.data[13 + 5]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), P.data[26], 1e-6);
}

test "audit: topKInto and gatherMatMulInto validate shapes and indices" {
    const allocator = std.testing.allocator;
    const X = try array(allocator, &.{ 2, 4 }, &.{ 0.1, 0.9, 0.5, 0.9, 3.0, 1.0, 2.0, 0.0 });
    defer free(allocator, X);
    const vals_bad = try zeros(allocator, &.{ 2, 1 });
    defer free(allocator, vals_bad);
    const idx_ok = try zeros(allocator, &.{ 2, 2 });
    defer free(allocator, idx_ok);
    try std.testing.expectError(error.ShapeMismatch, X.topKInto(2, vals_bad, idx_ok));

    const top = try X.topK(2, allocator);
    defer free(allocator, top.values);
    defer free(allocator, top.indices);
    try std.testing.expectEqualSlices(f32, &.{ 0.9, 0.9, 3.0, 2.0 }, top.values.data);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 0, 2 }, top.indices.data);

    // 两个专家：权重 [4, 3]
    var ws: [2]*Tensor = undefined;
    for (&ws, 0..) |*w, e| {
        w.* = try zeros(allocator, &.{ 4, 3 });
        fillDeterministic(w.*, 50 + e);
    }
    defer for (ws) |w| free(allocator, w);
    const sel = try array(allocator, &.{ 2, 2 }, &.{ 0, 1, 1, 7 }); // 7 截断为最后一个专家
    defer free(allocator, sel);
    const Y = try X.gatherMatMul(sel, &ws, null, allocator);
    defer free(allocator, Y);
    for (0..2) |n| for (0..2) |k| {
        const e: usize = @min(@as(usize, @intFromFloat(sel.data[n * 2 + k])), 1);
        for (0..3) |j| {
            var acc: f32 = 0.0;
            for (0..4) |i| acc += X.data[n * 4 + i] * ws[e].data[i * 3 + j];
            try std.testing.expectApproxEqAbs(acc, Y.data[(n * 2 + k) * 3 + j], 1e-5);
        }
    };

    const sel_neg = try array(allocator, &.{ 2, 2 }, &.{ 0, -1, 1, 0 });
    defer free(allocator, sel_neg);
    try std.testing.expectError(error.IndexOutOfBounds, X.gatherMatMul(sel_neg, &ws, null, allocator));
    const sel_nan = try array(allocator, &.{ 2, 2 }, &.{ 0, std.math.nan(f32), 1, 0 });
    defer free(allocator, sel_nan);
    try std.testing.expectError(error.IndexOutOfBounds, X.gatherMatMul(sel_nan, &ws, null, allocator));
    const w_bad = try zeros(allocator, &.{ 5, 3 });
    defer free(allocator, w_bad);
    try std.testing.expectError(error.ShapeMismatch, X.gatherMatMul(sel, &.{ ws[0], w_bad }, null, allocator));
}

test "audit: where broadcasts condition, x and y across strided layouts" {
    const allocator = std.testing.allocator;
    const cond = try array(allocator, &.{ 3, 1 }, &.{ 1, 0, 1 });
    defer free(allocator, cond);
    const x_base = try zeros(allocator, &.{ 4, 3 });
    defer free(allocator, x_base);
    fillDeterministic(x_base, 61);
    const x = try x_base.transposeView(0, 1, allocator); // [3, 4]
    defer free(allocator, x);
    const y = try array(allocator, &.{4}, &.{ -1, -2, -3, -4 });
    defer free(allocator, y);
    const out = try tensor.where(cond, x, y, allocator);
    defer free(allocator, out);
    for (0..3) |i| for (0..4) |j| {
        const want = if (cond.data[i] != 0) x_base.data[j * 3 + i] else y.data[j];
        try std.testing.expectEqual(want, out.data[i * 4 + j]);
    };
}

test "audit: arange computes start + i * step without accumulated drift" {
    const allocator = std.testing.allocator;
    const t = try arange(allocator, 0.0, 1000.0, 0.1);
    defer free(allocator, t);
    try std.testing.expectEqual(@as(usize, 10000), t.data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 999.9), t.data[9999], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 500.0), t.data[5000], 1e-3);
}

test "audit: embedding gathers rows from a transposed (non-contiguous) table" {
    const allocator = std.testing.allocator;
    const base = try array(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, base);
    const table = try base.transposeView(0, 1, allocator); // [3, 2]: 行为 [1,4], [2,5], [3,6]
    defer free(allocator, table);
    const idx = [_]usize{ 2, 0 };
    const out = try table.embedding(@as([]const usize, &idx), allocator);
    defer free(allocator, out);
    try std.testing.expectEqualSlices(f32, &.{ 3, 6, 1, 4 }, out.data);
}

// ============================================================================
// 后续修复回归测试 (Follow-up Fix Regression Tests)：专家分组收集矩阵乘法、跨步归约与填充、NaN 全序与视图读取
// ============================================================================

/// 测试辅助：朴素索引收集矩阵乘法参考实现，`x_row(n, k)` 为第 (n, k) 行输入向量
fn expectGatherMatMulMatchesReference(X: *const Tensor, indices: *const Tensor, ws: []const *Tensor, bs: ?[]const *Tensor, Y: *const Tensor) !void {
    const N = indices.shape.dims[0];
    const K = indices.shape.dims[1];
    const in_dim = ws[0].shape.dims[0];
    const out_dim = ws[0].shape.dims[1];
    for (0..N) |n| for (0..K) |k| {
        const e: usize = @min(@as(usize, @intFromFloat(indices.data[n * K + k])), ws.len - 1);
        for (0..out_dim) |j| {
            var acc: f32 = if (bs) |b| b[e].data[j] else 0.0;
            for (0..in_dim) |i| {
                const x_val = if (X.shape.len == 3) atCoord(X, &.{ n, k, i }) else atCoord(X, &.{ n, i });
                acc += x_val * ws[e].data[i * out_dim + j];
            }
            try std.testing.expectApproxEqAbs(acc, Y.data[(n * K + k) * out_dim + j], 1e-4);
        }
    };
}

test "fix: gatherMatMul groups rows by expert and matches the per-row reference" {
    const allocator = std.testing.allocator;
    const N: usize = 37;
    const K: usize = 3;
    const E: usize = 5;
    const in_dim: usize = 7;
    const out_dim: usize = 11;

    var ws: [E]*Tensor = undefined;
    var bs: [E]*Tensor = undefined;
    for (&ws, &bs, 0..) |*w, *b, e| {
        w.* = try zeros(allocator, &.{ in_dim, out_dim });
        fillDeterministic(w.*, 900 + e);
        b.* = try zeros(allocator, &.{out_dim});
        fillDeterministic(b.*, 950 + e);
    }
    defer for (ws, bs) |w, b| {
        free(allocator, w);
        free(allocator, b);
    };

    // 专家 2 不被任何词元选中；越界下标 9 截断为最后一个专家
    const sel = try zeros(allocator, &.{ N, K });
    defer free(allocator, sel);
    var prng = std.Random.DefaultPrng.init(17);
    for (sel.data, 0..) |*v, i| {
        const choices = [_]f32{ 0, 1, 3, 4, 9 };
        v.* = if (i % 7 == 0) 9 else choices[prng.random().uintLessThan(usize, choices.len)];
    }

    // 二维共享输入 [N, in_dim]
    const X2 = try zeros(allocator, &.{ N, in_dim });
    defer free(allocator, X2);
    fillDeterministic(X2, 901);
    const Y2 = try X2.gatherMatMul(sel, &ws, &bs, allocator);
    defer free(allocator, Y2);
    try expectGatherMatMulMatchesReference(X2, sel, &ws, &bs, Y2);

    // 三维逐槽位输入：由 [K, N, in_dim] 交换前两维得到的跨步视图 [N, K, in_dim]（行内连续）
    const X3_base = try zeros(allocator, &.{ K, N, in_dim });
    defer free(allocator, X3_base);
    fillDeterministic(X3_base, 902);
    const X3 = try X3_base.transposeView(0, 1, allocator);
    defer free(allocator, X3);
    const Y3 = try zeros(allocator, &.{ N, K, out_dim });
    defer free(allocator, Y3);
    try X3.gatherMatMulInto(sel, &ws, null, Y3, allocator);
    try expectGatherMatMulMatchesReference(X3, sel, &ws, null, Y3);

    // 分配失败时不泄漏
    try std.testing.checkAllAllocationFailures(allocator, struct {
        fn run(a: std.mem.Allocator, x: *Tensor, s: *Tensor, w: []const *Tensor, b: []const *Tensor) !void {
            const y = try x.gatherMatMul(s, w, b, a);
            free(a, y);
        }
    }.run, .{ X2, sel, @as([]const *Tensor, &ws), @as([]const *Tensor, &bs) });
}

