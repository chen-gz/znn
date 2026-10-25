//! 统一计算设备架构 (Compute Unified Device Architecture, CUDA) 硬件加速内核模块
//!
//! 提供基于并行线程执行 (Parallel Thread Execution, PTX) 虚拟指令集的底层深度学习算子内核，
//! 包括通用矩阵乘法 (General Matrix Multiply, GEMM)、均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)、
//! 归一化指数函数 (Softmax)、旋转位置编码 (Rotary Position Embedding, RoPE) 与注意力计算。
//! 所有 PTX 汇编在运行时由 NVIDIA 显卡驱动内置的即时编译器 (Just-In-Time Compiler, JIT) 编译为原生机器码。

const std = @import("std");

/// 算子内核 PTX 汇编代码集合 (纯 ASCII 字符集以满足 NVIDIA PTX 编译器要求)
pub const KERNELS_PTX =
    \\.version 7.0
    \\.target sm_50
    \\.address_size 64
    \\
    \\// GEMM Kernel: C = A * B
    \\// A: [M, K], B: [K, N], C: [M, N]
    \\.visible .entry sgemm_kernel(
    \\    .param .u64 param_A,
    \\    .param .u64 param_B,
    \\    .param .u64 param_C,
    \\    .param .u32 param_M,
    \\    .param .u32 param_N,
    \\    .param .u32 param_K
    \\) {
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<12>;
    \\    .reg .b64 %rd<15>;
    \\    .reg .f32 %f<5>;
    \\
    \\    mov.u32 %r0, %ctaid.y;
    \\    mov.u32 %r1, %ntid.y;
    \\    mov.u32 %r2, %tid.y;
    \\    mad.lo.s32 %r3, %r0, %r1, %r2;
    \\
    \\    mov.u32 %r4, %ctaid.x;
    \\    mov.u32 %r5, %ntid.x;
    \\    mov.u32 %r6, %tid.x;
    \\    mad.lo.s32 %r7, %r4, %r5, %r6;
    \\
    \\    ld.param.u32 %r8, [param_M];
    \\    ld.param.u32 %r9, [param_N];
    \\    ld.param.u32 %r10, [param_K];
    \\
    \\    setp.ge.u32 %p0, %r3, %r8;
    \\    @%p0 bra EXIT_SGEMM;
    \\    setp.ge.u32 %p1, %r7, %r9;
    \\    @%p1 bra EXIT_SGEMM;
    \\
    \\    ld.param.u64 %rd0, [param_A];
    \\    ld.param.u64 %rd1, [param_B];
    \\    ld.param.u64 %rd2, [param_C];
    \\
    \\    mov.f32 %f0, 0f00000000;
    \\    mov.u32 %r11, 0;
    \\
    \\LOOP_SGEMM:
    \\    setp.ge.u32 %p2, %r11, %r10;
    \\    @%p2 bra STORE_SGEMM;
    \\
    \\    mad.lo.s32 %r0, %r3, %r10, %r11;
    \\    mul.wide.u32 %rd3, %r0, 4;
    \\    add.s64 %rd4, %rd0, %rd3;
    \\    ld.global.f32 %f1, [%rd4];
    \\
    \\    mad.lo.s32 %r1, %r11, %r9, %r7;
    \\    mul.wide.u32 %rd5, %r1, 4;
    \\    add.s64 %rd6, %rd1, %rd5;
    \\    ld.global.f32 %f2, [%rd6];
    \\
    \\    fma.rn.f32 %f0, %f1, %f2, %f0;
    \\
    \\    add.u32 %r11, %r11, 1;
    \\    bra LOOP_SGEMM;
    \\
    \\STORE_SGEMM:
    \\    mad.lo.s32 %r0, %r3, %r9, %r7;
    \\    mul.wide.u32 %rd7, %r0, 4;
    \\    add.s64 %rd8, %rd2, %rd7;
    \\    st.global.f32 [%rd8], %f0;
    \\
    \\EXIT_SGEMM:
    \\    ret;
    \\}
    \\
    \\// RMSNorm Kernel: out = (x / sqrt(mean(x^2) + eps)) * weight
    \\.visible .entry rms_norm_kernel(
    \\    .param .u64 param_x,
    \\    .param .u64 param_w,
    \\    .param .u64 param_out,
    \\    .param .f32 param_eps,
    \\    .param .u32 param_rows,
    \\    .param .u32 param_cols
    \\) {
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<10>;
    \\    .reg .b64 %rd<10>;
    \\    .reg .f32 %f<10>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    ld.param.u32 %r1, [param_rows];
    \\    setp.ge.u32 %p0, %r0, %r1;
    \\    @%p0 bra EXIT_RMS;
    \\
    \\    ld.param.u32 %r2, [param_cols];
    \\    ld.param.u64 %rd0, [param_x];
    \\    ld.param.u64 %rd1, [param_w];
    \\    ld.param.u64 %rd2, [param_out];
    \\    ld.param.f32 %f0, [param_eps];
    \\
    \\    mul.lo.u32 %r3, %r0, %r2;
    \\    mul.wide.u32 %rd5, %r3, 4;
    \\    add.s64 %rd6, %rd0, %rd5;
    \\    add.s64 %rd7, %rd2, %rd5;
    \\
    \\    mov.f32 %f1, 0f00000000;
    \\    mov.u32 %r4, 0;
    \\LOOP_RMS_SUM:
    \\    setp.ge.u32 %p1, %r4, %r2;
    \\    @%p1 bra CALC_RMS_SCALE;
    \\    mul.wide.u32 %rd8, %r4, 4;
    \\    add.s64 %rd9, %rd6, %rd8;
    \\    ld.global.f32 %f2, [%rd9];
    \\    fma.rn.f32 %f1, %f2, %f2, %f1;
    \\    add.u32 %r4, %r4, 1;
    \\    bra LOOP_RMS_SUM;
    \\
    \\CALC_RMS_SCALE:
    \\    cvt.rn.f32.u32 %f3, %r2;
    \\    div.approx.f32 %f4, %f1, %f3;
    \\    add.f32 %f5, %f4, %f0;
    \\    rsqrt.approx.f32 %f6, %f5;
    \\
    \\    mov.u32 %r4, 0;
    \\LOOP_RMS_APPLY:
    \\    setp.ge.u32 %p2, %r4, %r2;
    \\    @%p2 bra EXIT_RMS;
    \\    mul.wide.u32 %rd8, %r4, 4;
    \\    add.s64 %rd9, %rd6, %rd8;
    \\    ld.global.f32 %f2, [%rd9];
    \\    mul.f32 %f7, %f2, %f6;
    \\
    \\    setp.eq.u64 %p3, %rd1, 0;
    \\    @%p3 bra STORE_RMS_VAL;
    \\    add.s64 %rd3, %rd1, %rd8;
    \\    ld.global.f32 %f8, [%rd3];
    \\    mul.f32 %f7, %f7, %f8;
    \\
    \\STORE_RMS_VAL:
    \\    add.s64 %rd4, %rd7, %rd8;
    \\    st.global.f32 [%rd4], %f7;
    \\    add.u32 %r4, %r4, 1;
    \\    bra LOOP_RMS_APPLY;
    \\
    \\EXIT_RMS:
    \\    ret;
    \\}
    \\
    \\// Elementwise Add Kernel: C = A + B
    \\.visible .entry add_kernel(
    \\    .param .u64 param_a,
    \\    .param .u64 param_b,
    \\    .param .u64 param_c,
    \\    .param .u32 param_n
    \\) {
    \\    .reg .pred %p;
    \\    .reg .b32 %r<5>;
    \\    .reg .b64 %rd<8>;
    \\    .reg .f32 %f<3>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    mov.u32 %r1, %ntid.x;
    \\    mov.u32 %r2, %tid.x;
    \\    mad.lo.s32 %r3, %r0, %r1, %r2;
    \\    ld.param.u32 %r4, [param_n];
    \\    setp.ge.u32 %p, %r3, %r4;
    \\    @%p bra EXIT_ADD;
    \\
    \\    ld.param.u64 %rd0, [param_a];
    \\    ld.param.u64 %rd1, [param_b];
    \\    ld.param.u64 %rd2, [param_c];
    \\    mul.wide.u32 %rd3, %r3, 4;
    \\    add.s64 %rd4, %rd0, %rd3;
    \\    add.s64 %rd5, %rd1, %rd3;
    \\    add.s64 %rd6, %rd2, %rd3;
    \\    ld.global.f32 %f0, [%rd4];
    \\    ld.global.f32 %f1, [%rd5];
    \\    add.f32 %f2, %f0, %f1;
    \\    st.global.f32 [%rd6], %f2;
    \\
    \\EXIT_ADD:
    \\    ret;
    \\}
    \\
    \\// Elementwise Multiply Kernel: C = A * B
    \\.visible .entry multiply_kernel(
    \\    .param .u64 param_a,
    \\    .param .u64 param_b,
    \\    .param .u64 param_c,
    \\    .param .u32 param_n
    \\) {
    \\    .reg .pred %p;
    \\    .reg .b32 %r<5>;
    \\    .reg .b64 %rd<8>;
    \\    .reg .f32 %f<3>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    mov.u32 %r1, %ntid.x;
    \\    mov.u32 %r2, %tid.x;
    \\    mad.lo.s32 %r3, %r0, %r1, %r2;
    \\    ld.param.u32 %r4, [param_n];
    \\    setp.ge.u32 %p, %r3, %r4;
    \\    @%p bra EXIT_MUL;
    \\
    \\    ld.param.u64 %rd0, [param_a];
    \\    ld.param.u64 %rd1, [param_b];
    \\    ld.param.u64 %rd2, [param_c];
    \\    mul.wide.u32 %rd3, %r3, 4;
    \\    add.s64 %rd4, %rd0, %rd3;
    \\    add.s64 %rd5, %rd1, %rd3;
    \\    add.s64 %rd6, %rd2, %rd3;
    \\    ld.global.f32 %f0, [%rd4];
    \\    ld.global.f32 %f1, [%rd5];
    \\    mul.f32 %f2, %f0, %f1;
    \\    st.global.f32 [%rd6], %f2;
    \\
    \\EXIT_MUL:
    \\    ret;
    \\}
    \\
    \\// Scalar Multiply Kernel: C = A * scalar
    \\.visible .entry mul_scalar_kernel(
    \\    .param .u64 param_a,
    \\    .param .f32 param_scalar,
    \\    .param .u64 param_c,
    \\    .param .u32 param_n
    \\) {
    \\    .reg .pred %p;
    \\    .reg .b32 %r<5>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<3>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    mov.u32 %r1, %ntid.x;
    \\    mov.u32 %r2, %tid.x;
    \\    mad.lo.s32 %r3, %r0, %r1, %r2;
    \\    ld.param.u32 %r4, [param_n];
    \\    setp.ge.u32 %p, %r3, %r4;
    \\    @%p bra EXIT_MUL_SCALAR;
    \\
    \\    ld.param.u64 %rd0, [param_a];
    \\    ld.param.f32 %f0, [param_scalar];
    \\    ld.param.u64 %rd1, [param_c];
    \\    mul.wide.u32 %rd2, %r3, 4;
    \\    add.s64 %rd3, %rd0, %rd2;
    \\    add.s64 %rd4, %rd1, %rd2;
    \\    ld.global.f32 %f1, [%rd3];
    \\    mul.f32 %f2, %f1, %f0;
    \\    st.global.f32 [%rd4], %f2;
    \\
    \\EXIT_MUL_SCALAR:
    \\    ret;
    \\}
    \\
    \\// GELU Kernel: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    \\.visible .entry gelu_kernel(
    \\    .param .u64 param_a,
    \\    .param .u64 param_out,
    \\    .param .u32 param_n
    \\) {
    \\    .reg .pred %p;
    \\    .reg .b32 %r<5>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<15>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    mov.u32 %r1, %ntid.x;
    \\    mov.u32 %r2, %tid.x;
    \\    mad.lo.s32 %r3, %r0, %r1, %r2;
    \\    ld.param.u32 %r4, [param_n];
    \\    setp.ge.u32 %p, %r3, %r4;
    \\    @%p bra EXIT_GELU;
    \\
    \\    ld.param.u64 %rd0, [param_a];
    \\    ld.param.u64 %rd1, [param_out];
    \\    mul.wide.u32 %rd2, %r3, 4;
    \\    add.s64 %rd3, %rd0, %rd2;
    \\    add.s64 %rd4, %rd1, %rd2;
    \\    ld.global.f32 %f0, [%rd3];
    \\
    \\    mul.f32 %f1, %f0, %f0;
    \\    mul.f32 %f2, %f1, %f0;
    \\    mov.f32 %f3, 0f3d372713;
    \\    fma.rn.f32 %f4, %f2, %f3, %f0;
    \\    mov.f32 %f5, 0f3f4c422a;
    \\    mul.f32 %f6, %f4, %f5;
    \\
    \\    add.f32 %f7, %f6, %f6;
    \\    max.f32 %f8, %f7, 0fc1700000;
    \\    min.f32 %f9, %f8, 0f41700000;
    \\    mov.f32 %f10, 0f3fb8aa3b;
    \\    mul.f32 %f11, %f9, %f10;
    \\    ex2.approx.f32 %f12, %f11;
    \\
    \\    add.f32 %f13, %f12, 0f3f800000;
    \\    sub.f32 %f14, %f12, 0f3f800000;
    \\    div.approx.f32 %f8, %f14, %f13;
    \\
    \\    add.f32 %f9, %f8, 0f3f800000;
    \\    mul.f32 %f10, %f0, 0f3f000000;
    \\    mul.f32 %f11, %f10, %f9;
    \\    st.global.f32 [%rd4], %f11;
    \\
    \\EXIT_GELU:
    \\    ret;
    \\}
    \\
    \\// Softmax Kernel: out = exp(x - max) / sum(exp(x - max))
    \\.visible .entry softmax_kernel(
    \\    .param .u64 param_x,
    \\    .param .u64 param_out,
    \\    .param .u32 param_rows,
    \\    .param .u32 param_cols
    \\) {
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<10>;
    \\    .reg .b64 %rd<10>;
    \\    .reg .f32 %f<12>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    ld.param.u32 %r1, [param_rows];
    \\    setp.ge.u32 %p0, %r0, %r1;
    \\    @%p0 bra EXIT_SOFTMAX;
    \\
    \\    ld.param.u32 %r2, [param_cols];
    \\    ld.param.u64 %rd0, [param_x];
    \\    ld.param.u64 %rd1, [param_out];
    \\
    \\    mul.lo.u32 %r3, %r0, %r2;
    \\    mul.wide.u32 %rd2, %r3, 4;
    \\    add.s64 %rd3, %rd0, %rd2;
    \\    add.s64 %rd4, %rd1, %rd2;
    \\
    \\    ld.global.f32 %f0, [%rd3];
    \\    mov.u32 %r4, 1;
    \\LOOP_SOFTMAX_MAX:
    \\    setp.ge.u32 %p1, %r4, %r2;
    \\    @%p1 bra INIT_SOFTMAX_SUM;
    \\    mul.wide.u32 %rd5, %r4, 4;
    \\    add.s64 %rd6, %rd3, %rd5;
    \\    ld.global.f32 %f1, [%rd6];
    \\    max.f32 %f0, %f0, %f1;
    \\    add.u32 %r4, %r4, 1;
    \\    bra LOOP_SOFTMAX_MAX;
    \\
    \\INIT_SOFTMAX_SUM:
    \\    mov.f32 %f2, 0f00000000;
    \\    mov.f32 %f3, 0f3fb8aa3b;
    \\    mov.u32 %r4, 0;
    \\LOOP_SOFTMAX_SUM:
    \\    setp.ge.u32 %p2, %r4, %r2;
    \\    @%p2 bra CALC_SOFTMAX_RCP;
    \\    mul.wide.u32 %rd5, %r4, 4;
    \\    add.s64 %rd6, %rd3, %rd5;
    \\    ld.global.f32 %f1, [%rd6];
    \\    sub.f32 %f4, %f1, %f0;
    \\    mul.f32 %f5, %f4, %f3;
    \\    ex2.approx.f32 %f6, %f5;
    \\    add.f32 %f2, %f2, %f6;
    \\    add.u32 %r4, %r4, 1;
    \\    bra LOOP_SOFTMAX_SUM;
    \\
    \\CALC_SOFTMAX_RCP:
    \\    rcp.approx.f32 %f7, %f2;
    \\    mov.u32 %r4, 0;
    \\LOOP_SOFTMAX_NORM:
    \\    setp.ge.u32 %p3, %r4, %r2;
    \\    @%p3 bra EXIT_SOFTMAX;
    \\    mul.wide.u32 %rd5, %r4, 4;
    \\    add.s64 %rd6, %rd3, %rd5;
    \\    ld.global.f32 %f1, [%rd6];
    \\    sub.f32 %f4, %f1, %f0;
    \\    mul.f32 %f5, %f4, %f3;
    \\    ex2.approx.f32 %f6, %f5;
    \\    mul.f32 %f8, %f6, %f7;
    \\    add.s64 %rd7, %rd4, %rd5;
    \\    st.global.f32 [%rd7], %f8;
    \\    add.u32 %r4, %r4, 1;
    \\    bra LOOP_SOFTMAX_NORM;
    \\
    \\EXIT_SOFTMAX:
    \\    ret;
    \\}
    \\
    \\// RoPE Kernel
    \\.visible .entry rope_kernel(
    \\    .param .u64 param_x,
    \\    .param .u64 param_out,
    \\    .param .u32 param_total_tokens,
    \\    .param .u32 param_head_dim,
    \\    .param .u32 param_start_pos,
    \\    .param .f32 param_base,
    \\    .param .u32 param_traditional
    \\) {
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<12>;
    \\    .reg .b64 %rd<12>;
    \\    .reg .f32 %f<18>;
    \\
    \\    mov.u32 %r0, %ctaid.x;
    \\    ld.param.u32 %r1, [param_total_tokens];
    \\    setp.ge.u32 %p0, %r0, %r1;
    \\    @%p0 bra EXIT_ROPE;
    \\
    \\    ld.param.u32 %r2, [param_head_dim];
    \\    ld.param.u32 %r3, [param_start_pos];
    \\    ld.param.f32 %f0, [param_base];
    \\    ld.param.u32 %r4, [param_traditional];
    \\    ld.param.u64 %rd0, [param_x];
    \\    ld.param.u64 %rd1, [param_out];
    \\
    \\    add.u32 %r5, %r3, %r0;
    \\    cvt.rn.f32.u32 %f1, %r5;
    \\
    \\    mul.lo.u32 %r6, %r0, %r2;
    \\    mul.wide.u32 %rd2, %r6, 4;
    \\    add.s64 %rd3, %rd0, %rd2;
    \\    add.s64 %rd4, %rd1, %rd2;
    \\
    \\    shr.u32 %r7, %r2, 1;
    \\    cvt.rn.f32.u32 %f2, %r2;
    \\    lg2.approx.f32 %f3, %f0;
    \\
    \\    mov.u32 %r8, 0;
    \\LOOP_ROPE_PAIR:
    \\    setp.ge.u32 %p1, %r8, %r7;
    \\    @%p1 bra EXIT_ROPE;
    \\
    \\    shl.b32 %r9, %r8, 1;
    \\    cvt.rn.f32.u32 %f4, %r9;
    \\    neg.f32 %f5, %f4;
    \\    div.approx.f32 %f6, %f5, %f2;
    \\    mul.f32 %f7, %f6, %f3;
    \\    ex2.approx.f32 %f8, %f7;
    \\
    \\    mul.f32 %f9, %f1, %f8;
    \\    cos.approx.f32 %f10, %f9;
    \\    sin.approx.f32 %f11, %f9;
    \\
    \\    setp.eq.u32 %p2, %r4, 1;
    \\    @%p2 bra ROPE_INTERLEAVED;
    \\
    \\    mul.wide.u32 %rd5, %r8, 4;
    \\    add.s64 %rd6, %rd3, %rd5;
    \\    ld.global.f32 %f12, [%rd6];
    \\    add.u32 %r10, %r8, %r7;
    \\    mul.wide.u32 %rd7, %r10, 4;
    \\    add.s64 %rd8, %rd3, %rd7;
    \\    ld.global.f32 %f13, [%rd8];
    \\
    \\    mul.f32 %f14, %f12, %f10;
    \\    neg.f32 %f16, %f11;
    \\    fma.rn.f32 %f14, %f13, %f16, %f14;
    \\    mul.f32 %f15, %f12, %f11;
    \\    fma.rn.f32 %f15, %f13, %f10, %f15;
    \\
    \\    add.s64 %rd9, %rd4, %rd5;
    \\    st.global.f32 [%rd9], %f14;
    \\    add.s64 %rd10, %rd4, %rd7;
    \\    st.global.f32 [%rd10], %f15;
    \\    bra NEXT_ROPE_PAIR;
    \\
    \\ROPE_INTERLEAVED:
    \\    mul.wide.u32 %rd5, %r9, 4;
    \\    add.s64 %rd6, %rd3, %rd5;
    \\    ld.global.f32 %f12, [%rd6];
    \\    add.u32 %r10, %r9, 1;
    \\    mul.wide.u32 %rd7, %r10, 4;
    \\    add.s64 %rd8, %rd3, %rd7;
    \\    ld.global.f32 %f13, [%rd8];
    \\
    \\    mul.f32 %f14, %f12, %f10;
    \\    neg.f32 %f16, %f11;
    \\    fma.rn.f32 %f14, %f13, %f16, %f14;
    \\    mul.f32 %f15, %f12, %f11;
    \\    fma.rn.f32 %f15, %f13, %f10, %f15;
    \\
    \\    add.s64 %rd9, %rd4, %rd5;
    \\    st.global.f32 [%rd9], %f14;
    \\    add.s64 %rd10, %rd4, %rd7;
    \\    st.global.f32 [%rd10], %f15;
    \\
    \\NEXT_ROPE_PAIR:
    \\    add.u32 %r8, %r8, 1;
    \\    bra LOOP_ROPE_PAIR;
    \\
    \\EXIT_ROPE:
    \\    ret;
    \\}
    \\
;
