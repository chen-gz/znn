const std = @import("std");
const testing = std.testing;
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const QROptions = tensor.QROptions;
const QRMode = tensor.QRMode;
const zeros = tensor.zeros;
const free = tensor.free;

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

test "Tensor linear algebra: complete QR decomposition (mode = .complete)" {
    const allocator = std.testing.allocator;

    // Tall matrix 4x2
    const A = try zeros(allocator, &.{ 4, 2 });
    defer free(allocator, A);
    @memcpy(A.data, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    });

    const qr_res = try tensor.qrWithOptions(A.*, allocator, .{ .mode = .complete });
    defer qr_res.deinit(allocator);

    // Complete QR: Q is [4, 4], R is [4, 2]
    try std.testing.expectEqual(@as(usize, 2), qr_res.Q.shape.len);
    try std.testing.expectEqual(@as(usize, 4), qr_res.Q.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), qr_res.Q.shape.dims[1]);

    try std.testing.expectEqual(@as(usize, 2), qr_res.R.shape.len);
    try std.testing.expectEqual(@as(usize, 4), qr_res.R.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), qr_res.R.shape.dims[1]);

    // Check bottom rows of R (row 2 and 3) are zeros
    for (2..4) |i| {
        for (0..2) |j| {
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), qr_res.R.data[i * 2 + j], 1e-5);
        }
    }

    // Check Q^T * Q = I_4
    const QT = try qr_res.Q.transpose(0, 1, allocator);
    defer free(allocator, QT);
    const QTQ = try QT.matmul(qr_res.Q, allocator);
    defer free(allocator, QTQ);
    for (0..4) |i| {
        for (0..4) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, QTQ.data[i * 4 + j], 1e-4);
        }
    }

    // Check Q * Q^T = I_4 (complete orthogonal matrix is unitary)
    const QQT = try qr_res.Q.matmul(QT, allocator);
    defer free(allocator, QQT);
    for (0..4) |i| {
        for (0..4) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, QQT.data[i * 4 + j], 1e-4);
        }
    }

    // Check reconstruction: Q * R == A
    const A_rec = try qr_res.Q.matmul(qr_res.R, allocator);
    defer free(allocator, A_rec);
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(A.data[i], A_rec.data[i], 1e-4);
    }
}

test "Tensor linear algebra: solveTriangularUpper and qrSolve least squares" {
    const allocator = std.testing.allocator;

    // 1. Test solveTriangularUpper on 3x3 upper triangular system:
    // [2  1 -1] [x0]   [ 1]
    // [0  3  2] [x1] = [ 8]
    // [0  0  4] [x2]   [ 4]
    // Solution: x2 = 1, x1 = (8 - 2*1)/3 = 2, x0 = (1 - 1*2 - (-1)*1)/2 = 0
    const R = try zeros(allocator, &.{ 3, 3 });
    defer free(allocator, R);
    @memcpy(R.data, &[_]f32{
        2.0, 1.0, -1.0,
        0.0, 3.0,  2.0,
        0.0, 0.0,  4.0,
    });
    const b = try zeros(allocator, &.{3});
    defer free(allocator, b);
    @memcpy(b.data, &[_]f32{ 1.0, 8.0, 4.0 });

    const x = try tensor.solveTriangularUpper(R.*, b.*, allocator);
    defer free(allocator, x);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x.data[2], 1e-5);

    // 2. Test qrSolve on overdetermined least-squares system A * w = y
    // A: 4x2 matrix, w_true = [3.0, -1.5]
    const A_mat = try zeros(allocator, &.{ 4, 2 });
    defer free(allocator, A_mat);
    @memcpy(A_mat.data, &[_]f32{
        1.0, 2.0,
        2.0, 0.5,
        1.5, 3.0,
        3.0, 1.0,
    });
    const y_vec = try zeros(allocator, &.{4});
    defer free(allocator, y_vec);
    for (0..4) |i| {
        y_vec.data[i] = A_mat.data[i * 2 + 0] * 3.0 + A_mat.data[i * 2 + 1] * -1.5;
    }

    const w_sol = try tensor.qrSolve(A_mat.*, y_vec.*, allocator);
    defer free(allocator, w_sol);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), w_sol.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), w_sol.data[1], 1e-4);
}

test "Tensor linear algebra: Jacobi real symmetric eigenvalue decomposition (symeig)" {
    const allocator = std.testing.allocator;

    // Real symmetric matrix 3x3
    const A = try zeros(allocator, &.{ 3, 3 });
    defer free(allocator, A);
    @memcpy(A.data, &[_]f32{
        4.0,  1.0, -2.0,
        1.0,  2.0, 0.0,
        -2.0, 0.0, 3.0,
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

