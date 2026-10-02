const std = @import("std");
const builtin = @import("builtin");

// C 语言基础线性代数子程序 (C Basic Linear Algebra Subprograms, CBLAS) 布局与转置常量的 C 语言接口包装
pub const CblasRowMajor = 101;
pub const CblasColMajor = 102;
pub const CblasNoTrans = 111;
pub const CblasTrans = 112;

// 判断是否是 macOS / Apple 平台，来决定链接底层库还是使用纯 Zig 后备实现
pub const cblas_sgemm = if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .tvos or builtin.os.tag == .watchos)
    cblas_sgemm_accelerate
else
    cblas_sgemm_fallback;

// macOS Accelerate 框架动态导入的单精度通用矩阵乘法 (Single-Precision General Matrix Multiply, SGEMM) `cblas_sgemm` 接口原型
const cblas_sgemm_accelerate = @extern(*const fn (
    order: c_int,
    TransA: c_int,
    TransB: c_int,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) callconv(.c) void, .{
    .name = "cblas_sgemm",
});

// 纯 Zig 编写的通用矩阵乘法 (General Matrix Multiply, GEMM) 后备实现，确保在 Linux / Windows / WebAssembly 上均能编译且结果正确
fn cblas_sgemm_fallback(
    order: c_int,
    TransA: c_int,
    TransB: c_int,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) callconv(.c) void {
    std.debug.assert(order == CblasRowMajor);

    const m = @as(usize, @intCast(M));
    const n = @as(usize, @intCast(N));
    const k = @as(usize, @intCast(K));

    // 1. 初始化或缩放输出矩阵 C: C = beta * C
    if (beta == 0.0) {
        for (0..m) |i| {
            const c_row = C + i * @as(usize, @intCast(ldc));
            @memset(c_row[0..n], 0.0);
        }
    } else if (beta != 1.0) {
        for (0..m) |i| {
            const c_row = C + i * @as(usize, @intCast(ldc));
            for (0..n) |j| {
                c_row[j] *= beta;
            }
        }
    }

    if (alpha == 0.0) return;

    const ta = (TransA == CblasTrans);
    const tb = (TransB == CblasTrans);

    const lda_u = @as(usize, @intCast(lda));
    const ldb_u = @as(usize, @intCast(ldb));
    const ldc_u = @as(usize, @intCast(ldc));

    // 2. 单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量化矩阵乘法计算: C = alpha * A * B + C
    const Vec8 = @Vector(8, f32);
    const vec_len = 8;

    if (!ta and tb) {
        // NoTrans x Trans: A[i, p] 与 B[j, p] 均沿内积维度 p 连续排列，直接执行 8 路单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量化点积
        for (0..m) |i| {
            const a_row = A + i * lda_u;
            const c_row = C + i * ldc_u;
            for (0..n) |j| {
                const b_row = B + j * ldb_u;
                var acc_vec: Vec8 = @splat(0.0);
                var p: usize = 0;
                while (p + vec_len <= k) : (p += vec_len) {
                    const a_ptr: *const [vec_len]f32 = @ptrCast(a_row + p);
                    const b_ptr: *const [vec_len]f32 = @ptrCast(b_row + p);
                    const a_v: Vec8 = a_ptr.*;
                    const b_v: Vec8 = b_ptr.*;
                    acc_vec += a_v * b_v;
                }
                var dot: f32 = @reduce(.Add, acc_vec);
                while (p < k) : (p += 1) {
                    dot += a_row[p] * b_row[p];
                }
                c_row[j] += alpha * dot;
            }
        }
        return;
    }

    for (0..m) |i| {
        const c_row = C + i * ldc_u;
        for (0..k) |p| {
            const a_val = alpha * (if (ta) A[p * lda_u + i] else A[i * lda_u + p]);
            if (a_val == 0.0) continue;

            if (!tb) {
                const a_vec: Vec8 = @splat(a_val);
                var j: usize = 0;
                while (j + vec_len <= n) : (j += vec_len) {
                    const b_ptr: *const [vec_len]f32 = @ptrCast(B + p * ldb_u + j);
                    const b_vec: Vec8 = b_ptr.*;
                    const c_ptr: *[vec_len]f32 = @ptrCast(c_row + j);
                    var c_vec: Vec8 = c_ptr.*;
                    c_vec += a_vec * b_vec;
                    c_ptr.* = c_vec;
                }
                while (j < n) : (j += 1) {
                    c_row[j] += a_val * B[p * ldb_u + j];
                }
            } else {
                for (0..n) |j| {
                    const b_val = B[j * ldb_u + p];
                    c_row[j] += a_val * b_val;
                }
            }
        }
    }
}

test "cblas_sgemm_fallback matrix multiplication" {
    const A = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const B = [_]f32{
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
    };
    var C = [_]f32{ 0.0 } ** 4;

    cblas_sgemm_fallback(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        2, 2, 3,
        1.0,
        &A, 3,
        &B, 2,
        0.0,
        &C, 2,
    );

    // C = A * B
    try std.testing.expectApproxEqAbs(@as(f32, 2.2), C[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.8), C[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.9), C[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.4), C[3], 1e-5);

    // Test NoTrans x Trans with K = 10 (covers 8-wide SIMD + scalar remainder)
    var A10: [20]f32 = undefined;
    var B10: [20]f32 = undefined;
    for (0..20) |idx| {
        A10[idx] = @as(f32, @floatFromInt(idx + 1)) * 0.1;
        B10[idx] = @as(f32, @floatFromInt(idx + 1)) * 0.2;
    }
    var C_nt = [_]f32{0.0} ** 4;
    cblas_sgemm_fallback(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        2, 2, 10,
        1.0,
        &A10, 10,
        &B10, 10,
        0.0,
        &C_nt, 2,
    );
    var expected00: f32 = 0.0;
    var expected01: f32 = 0.0;
    for (0..10) |p| {
        expected00 += A10[p] * B10[p];
        expected01 += A10[p] * B10[10 + p];
    }
    try std.testing.expectApproxEqAbs(expected00, C_nt[0], 1e-5);
    try std.testing.expectApproxEqAbs(expected01, C_nt[1], 1e-5);
}


