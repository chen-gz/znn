const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
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

test "tensor argmax edge cases negative values and unsupported dimension error" {
    const allocator = std.testing.allocator;

    // 2x3 matrix with all negative values
    var t = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, t);

    // Row 0: [-10.0, -2.0, -5.0] -> max index is 1 (-2.0)
    // Row 1: [-1.0, -8.0, -4.0]  -> max index is 0 (-1.0)
    t.set(&.{ 0, 0 }, -10.0);
    t.set(&.{ 0, 1 }, -2.0);
    t.set(&.{ 0, 2 }, -5.0);
    t.set(&.{ 1, 0 }, -1.0);
    t.set(&.{ 1, 1 }, -8.0);
    t.set(&.{ 1, 2 }, -4.0);

    // 1. argmax dim=1
    const idx_col = try t.argmax(1, allocator);
    defer free(allocator, idx_col);
    try std.testing.expectEqual(@as(f32, 1.0), idx_col.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_col.data[1]);

    // 2. argmax dim=0
    // Col 0: -10 vs -1 -> index 1 (-1)
    // Col 1: -2 vs -8  -> index 0 (-2)
    // Col 2: -5 vs -4  -> index 1 (-4)
    const idx_row = try t.argmax(0, allocator);
    defer free(allocator, idx_row);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_row.data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[2]);

    // 3. Unsupported dimension on 3D tensor: dim=2 passes assert(dim < shape.len) and reaches error.UnsupportedDimension
    var t_3d = try zeros(allocator, &.{ 2, 2, 2 });
    defer free(allocator, t_3d);
    try std.testing.expectError(error.UnsupportedDimension, t_3d.argmax(2, allocator));
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
        var graph = autodiff.Graph.init(allocator);
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
    var graph = autodiff.Graph.init(allocator);
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

