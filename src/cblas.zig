const std = @import("std");

// 从 macOS Accelerate 框架中动态导入的 cblas_sgemm 接口原型
// 用于在 CPU 上实现极速的单精度通用矩阵乘法（GEMM）
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

// CBLAS 布局与转置常量的 C 语言接口包装
pub const CblasRowMajor = 101;
pub const CblasColMajor = 102;
pub const CblasNoTrans = 111;
pub const CblasTrans = 112;
pub const cblas_sgemm = cblas_sgemm_accelerate;


