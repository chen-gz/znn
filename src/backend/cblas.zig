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

// 纯 Zig 编写的通用矩阵乘法 (General Matrix Multiply, GEMM) 后备实现，确保在 Linux / Windows / WebAssembly 上均能编译且结果正确。
// 语义与标准 CBLAS 一致：支持行主序与列主序；beta == 0 时不读取 C 的原值；alpha == 0 时不读取 A 与 B；
// 其余情况下完整计算全部乘积（不跳过零元素），因此 0 * Inf、NaN 等非有限值与 Accelerate 一样正常传播。
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
    std.debug.assert(order == CblasRowMajor or order == CblasColMajor);

    const m: usize = @intCast(M);
    const n: usize = @intCast(N);
    const k: usize = @intCast(K);
    const ta = TransA != CblasNoTrans;
    const tb = TransB != CblasNoTrans;
    const lda_u: usize = @intCast(lda);
    const ldb_u: usize = @intCast(ldb);
    const ldc_u: usize = @intCast(ldc);

    if (order == CblasColMajor) {
        // 列主序 C (M×N) 的内存按行主序解读即为 Cᵀ (N×M) = op(B)ᵀ·op(A)ᵀ：交换 A/B 与 M/N 后按行主序计算
        sgemmRowMajor(tb, ta, n, m, k, alpha, B, ldb_u, A, lda_u, beta, C, ldc_u);
    } else {
        sgemmRowMajor(ta, tb, m, n, k, alpha, A, lda_u, B, ldb_u, beta, C, ldc_u);
    }
}

/// 行主序通用矩阵乘法：C = alpha * op(A) * op(B) + beta * C，op(A) 为 m×k，op(B) 为 k×n
fn sgemmRowMajor(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    A: [*]const f32,
    lda: usize,
    B: [*]const f32,
    ldb: usize,
    beta: f32,
    C: [*]f32,
    ldc: usize,
) void {
    // 1. 初始化或缩放输出矩阵 C: C = beta * C（beta == 0 时直接置零，不读取 C 中可能残留的 NaN）
    if (beta == 0.0) {
        for (0..m) |i| @memset(C[i * ldc ..][0..n], 0.0);
    } else if (beta != 1.0) {
        for (0..m) |i| {
            for (C[i * ldc ..][0..n]) |*v| v.* *= beta;
        }
    }

    if (m == 0 or n == 0 or k == 0 or alpha == 0.0) return;

    // 2. 累加 alpha * op(A) * op(B)：输出行数足以填满寄存器分块时走打包分块路径（列数不足 NR 时补零列仍远快于逐元素累加）；
    //    NoTrans x Trans 且列数不足 NR 时连续点积更快；输出行数不足 MR（例如单行向量-矩阵乘法）时打包开销无法摊薄，走流式路径
    const nt_dot = !ta and tb and n < NR;
    if (m >= MR and !nt_dot) {
        sgemmBlocked(ta, tb, m, n, k, alpha, A, lda, B, ldb, C, ldc);
    } else {
        sgemmSmall(ta, tb, m, n, k, alpha, A, lda, B, ldb, C, ldc);
    }
}

/// 单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量宽度与类型
const vec_len = 8;
const Vec = @Vector(vec_len, f32);

/// 目标平台是否具备硬件融合乘加 (Fused Multiply-Add, FMA) 指令；不具备时 `@mulAdd` 会退化为逐元素软件调用，故改用普通乘加
const has_fma = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
    else => false,
};

inline fn mulAdd(a: Vec, b: Vec, c: Vec) Vec {
    return if (has_fma) @mulAdd(Vec, a, b, c) else a * b + c;
}

/// 寄存器分块微内核尺寸：每次计算 MR×NR 的输出块（6×16 = 12 个 8 路累加向量，恰好放入 16 个向量寄存器）
const MR = 6;
const NR = 2 * vec_len;
/// 内积维度分块长度：一个 KC×NR 的 B 面板 (16 KB) 常驻一级缓存
const KC = 256;
/// 行分块长度 (MR 的整数倍)：一个 MC×KC 的 A 块 (72 KB) 常驻二级缓存，并在所有 B 面板间复用
const MC = 72;

comptime {
    std.debug.assert(MC % MR == 0);
}

/// 打包分块矩阵乘法 (Packed Blocked GEMM)：按 KC 切分内积维度、按 MC 切分行，
/// A 块按 MR 行条带转置打包（同时预乘 alpha），B 按 NR 列面板打包，打包缓冲区位于栈上，不进行堆分配。
fn sgemmBlocked(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    A: [*]const f32,
    lda: usize,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
) void {
    var a_pack: [MC * KC]f32 align(64) = undefined;
    var b_pack: [KC * NR]f32 align(64) = undefined;

    var p0: usize = 0;
    while (p0 < k) : (p0 += KC) {
        const kc = @min(KC, k - p0);
        var row0: usize = 0;
        while (row0 < m) : (row0 += MC) {
            const mc = @min(MC, m - row0);
            packA(ta, A, lda, row0, mc, p0, kc, alpha, &a_pack);
            var j0: usize = 0;
            while (j0 < n) : (j0 += NR) {
                const nr = @min(NR, n - j0);
                packB(tb, B, ldb, p0, kc, j0, nr, &b_pack);
                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    microKernel(kc, a_pack[ir * kc ..].ptr, &b_pack, C + (row0 + ir) * ldc + j0, ldc, @min(MR, mc - ir), nr);
                }
            }
        }
    }
}

/// 将 op(A) 的 [row0, row0 + mc) 行、[p0, p0 + kc) 列打包为若干 MR 行条带，条带内按 [p][r] 排列并预乘 alpha，不足 MR 行补零
fn packA(ta: bool, A: [*]const f32, lda: usize, row0: usize, mc: usize, p0: usize, kc: usize, alpha: f32, dst: *[MC * KC]f32) void {
    var ir: usize = 0;
    while (ir < mc) : (ir += MR) {
        const mr = @min(MR, mc - ir);
        const sliver = dst[ir * kc ..][0 .. MR * kc];
        if (ta) {
            // op(A) = Aᵀ：A 的第 p 行上 MR 个相邻元素即为条带的一列
            for (0..kc) |p| {
                const src = A + (p0 + p) * lda + row0 + ir;
                for (0..MR) |r| sliver[p * MR + r] = if (r < mr) alpha * src[r] else 0.0;
            }
        } else {
            for (0..MR) |r| {
                if (r < mr) {
                    const src = A + (row0 + ir + r) * lda + p0;
                    for (0..kc) |p| sliver[p * MR + r] = alpha * src[p];
                } else {
                    for (0..kc) |p| sliver[p * MR + r] = 0.0;
                }
            }
        }
    }
}

/// 将 op(B) 的 [p0, p0 + kc) 行、[j0, j0 + nr) 列打包为按 [p][j] 排列的 NR 列面板，不足 NR 列补零
fn packB(tb: bool, B: [*]const f32, ldb: usize, p0: usize, kc: usize, j0: usize, nr: usize, dst: *[KC * NR]f32) void {
    if (tb) {
        // op(B) = Bᵀ：B 的第 j0 + j 行沿内积维度连续，逐行读取后分散写入面板第 j 列
        for (0..NR) |j| {
            if (j < nr) {
                const src = B + (j0 + j) * ldb + p0;
                for (0..kc) |p| dst[p * NR + j] = src[p];
            } else {
                for (0..kc) |p| dst[p * NR + j] = 0.0;
            }
        }
    } else {
        for (0..kc) |p| {
            const src = B + (p0 + p) * ldb + j0;
            const row = dst[p * NR ..][0..NR];
            if (nr == NR) {
                row.* = src[0..NR].*;
            } else {
                @memcpy(row[0..nr], src[0..nr]);
                @memset(row[nr..], 0.0);
            }
        }
    }
}

/// MR×NR 寄存器分块微内核：C[0..mr, 0..nr] += a_slivers · b_panel
fn microKernel(kc: usize, a: [*]const f32, b: [*]const f32, c: [*]f32, ldc: usize, mr: usize, nr: usize) void {
    var lo: [MR]Vec = @splat(@splat(0.0));
    var hi: [MR]Vec = @splat(@splat(0.0));
    for (0..kc) |p| {
        const b_lo: Vec = b[p * NR ..][0..vec_len].*;
        const b_hi: Vec = b[p * NR + vec_len ..][0..vec_len].*;
        inline for (0..MR) |r| {
            const av: Vec = @splat(a[p * MR + r]);
            lo[r] = mulAdd(av, b_lo, lo[r]);
            hi[r] = mulAdd(av, b_hi, hi[r]);
        }
    }

    if (mr == MR and nr == NR) {
        inline for (0..MR) |r| {
            const row = c + r * ldc;
            row[0..vec_len].* = @as(Vec, row[0..vec_len].*) + lo[r];
            row[vec_len..NR].* = @as(Vec, row[vec_len..NR].*) + hi[r];
        }
    } else {
        var tile: [MR][NR]f32 = undefined;
        inline for (0..MR) |r| {
            tile[r][0..vec_len].* = lo[r];
            tile[r][vec_len..NR].* = hi[r];
        }
        for (0..mr) |r| {
            for (0..nr) |j| c[r * ldc + j] += tile[r][j];
        }
    }
}

/// 流式路径（输出行数小于 MR，例如单行向量-矩阵乘法；或列数小于 NR 的 NoTrans x Trans）：打包开销无法摊薄，直接流式访问原矩阵
fn sgemmSmall(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    A: [*]const f32,
    lda: usize,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
) void {
    if (!ta and tb) {
        // NoTrans x Trans: A[i, p] 与 B[j, p] 均沿内积维度 p 连续排列，直接执行向量化点积
        for (0..m) |i| {
            const a_row = A + i * lda;
            const c_row = C + i * ldc;
            for (0..n) |j| {
                c_row[j] += alpha * dotProduct(a_row, B + j * ldb, k);
            }
        }
        return;
    }

    for (0..m) |i| {
        const c_row = C + i * ldc;
        for (0..k) |p| {
            const a_val = alpha * (if (ta) A[p * lda + i] else A[i * lda + p]);
            if (!tb) {
                const b_row = B + p * ldb;
                const a_vec: Vec = @splat(a_val);
                var j: usize = 0;
                while (j + vec_len <= n) : (j += vec_len) {
                    const b_vec: Vec = b_row[j..][0..vec_len].*;
                    // 访存受限的轴向累加 (AXPY) 循环无循环携带依赖，分离的乘法与加法比融合乘加更利于双发射
                    c_row[j..][0..vec_len].* = @as(Vec, c_row[j..][0..vec_len].*) + a_vec * b_vec;
                }
                while (j < n) : (j += 1) {
                    c_row[j] += a_val * b_row[j];
                }
            } else {
                for (0..n) |j| {
                    c_row[j] += a_val * B[j * ldb + p];
                }
            }
        }
    }
}

/// 向量化点积：4 组独立累加器隐藏融合乘加 (Fused Multiply-Add, FMA) 延迟，尾部逐元素累加
fn dotProduct(a: [*]const f32, b: [*]const f32, k: usize) f32 {
    var acc: [4]Vec = @splat(@splat(0.0));
    var p: usize = 0;
    while (p + 4 * vec_len <= k) : (p += 4 * vec_len) {
        inline for (0..4) |u| {
            const off = p + u * vec_len;
            acc[u] = mulAdd(a[off..][0..vec_len].*, b[off..][0..vec_len].*, acc[u]);
        }
    }
    while (p + vec_len <= k) : (p += vec_len) {
        acc[0] = mulAdd(a[p..][0..vec_len].*, b[p..][0..vec_len].*, acc[0]);
    }
    var dot: f32 = @reduce(.Add, (acc[0] + acc[1]) + (acc[2] + acc[3]));
    while (p < k) : (p += 1) {
        dot += a[p] * b[p];
    }
    return dot;
}

/// 朴素三重循环参考实现：仅用于测试中逐元素校验各分块路径
fn referenceGemm(
    col_major: bool,
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    beta: f32,
    C: []f32,
    ldc: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0.0;
            for (0..k) |p| {
                // 行主序下 op(X)[r, s] 的偏移；列主序时行列步长互换
                const a_idx = if (ta != col_major) p * lda + i else i * lda + p;
                const b_idx = if (tb != col_major) j * ldb + p else p * ldb + j;
                acc += @as(f64, A[a_idx]) * @as(f64, B[b_idx]);
            }
            const c_idx = if (col_major) j * ldc + i else i * ldc + j;
            const prev: f64 = if (beta == 0.0) 0.0 else @as(f64, beta) * @as(f64, C[c_idx]);
            C[c_idx] = @floatCast(@as(f64, alpha) * acc + prev);
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
    var C: [4]f32 = @splat(0.0);

    cblas_sgemm_fallback(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        2,
        2,
        3,
        1.0,
        &A,
        3,
        &B,
        2,
        0.0,
        &C,
        2,
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
    var C_nt: [4]f32 = @splat(0.0);
    cblas_sgemm_fallback(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        2,
        2,
        10,
        1.0,
        &A10,
        10,
        &B10,
        10,
        0.0,
        &C_nt,
        2,
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

test "cblas_sgemm_fallback matches reference for all layouts, transposes, edge tiles and padded strides" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();

    // 覆盖小矩阵路径、恰好整块、行列尾块以及跨越多个 KC / MC 分块的尺寸
    const shapes = [_][3]usize{
        .{ 1, 1, 1 },     .{ 3, 5, 7 },      .{ 1, 40, 33 },   .{ 33, 3, 17 },
        .{ 6, 16, 8 },    .{ 7, 17, 9 },     .{ 13, 35, 300 }, .{ 80, 50, 600 },
        .{ 145, 33, 70 }, .{ 12, 200, 257 },
    };
    const scalars = [_][2]f32{ .{ 1.0, 0.0 }, .{ -0.5, 1.0 }, .{ 2.0, 0.25 } };

    for (shapes) |s| {
        const m = s[0];
        const n = s[1];
        const k = s[2];
        for ([_]bool{ false, true }) |col_major| {
            for ([_]bool{ false, true }) |ta| {
                for ([_]bool{ false, true }) |tb| {
                    // 存储形状：行主序下 op(A) = A 时 A 为 m×k，否则为 k×m；列主序时行列含义互换；步长额外填充 3 列以覆盖 ld > 宽度
                    const a_rows = if (ta != col_major) k else m;
                    const a_cols = if (ta != col_major) m else k;
                    const b_rows = if (tb != col_major) n else k;
                    const b_cols = if (tb != col_major) k else n;
                    const c_rows = if (col_major) n else m;
                    const c_cols = if (col_major) m else n;
                    const lda = a_cols + 3;
                    const ldb = b_cols + 3;
                    const ldc = c_cols + 3;

                    const A = try allocator.alloc(f32, a_rows * lda);
                    defer allocator.free(A);
                    const B = try allocator.alloc(f32, b_rows * ldb);
                    defer allocator.free(B);
                    const C = try allocator.alloc(f32, c_rows * ldc);
                    defer allocator.free(C);
                    const C_ref = try allocator.alloc(f32, c_rows * ldc);
                    defer allocator.free(C_ref);

                    for (A) |*v| v.* = rnd.floatNorm(f32);
                    for (B) |*v| v.* = rnd.floatNorm(f32);

                    for (scalars) |ab| {
                        for (C, C_ref) |*x, *y| {
                            x.* = rnd.floatNorm(f32);
                            y.* = x.*;
                        }
                        cblas_sgemm_fallback(
                            if (col_major) CblasColMajor else CblasRowMajor,
                            if (ta) CblasTrans else CblasNoTrans,
                            if (tb) CblasTrans else CblasNoTrans,
                            @intCast(m),
                            @intCast(n),
                            @intCast(k),
                            ab[0],
                            A.ptr,
                            @intCast(lda),
                            B.ptr,
                            @intCast(ldb),
                            ab[1],
                            C.ptr,
                            @intCast(ldc),
                        );
                        referenceGemm(col_major, ta, tb, m, n, k, ab[0], A, lda, B, ldb, ab[1], C_ref, ldc);

                        const tol = 1e-4 * @sqrt(@as(f32, @floatFromInt(k))) * 4.0;
                        for (0..c_rows) |r| {
                            for (0..ldc) |cc| {
                                const idx = r * ldc + cc;
                                if (cc < c_cols) {
                                    try std.testing.expectApproxEqAbs(C_ref[idx], C[idx], tol);
                                } else {
                                    // 步长填充区不得被写入
                                    try std.testing.expectEqual(C_ref[idx], C[idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

test "cblas_sgemm_fallback propagates NaN from 0 * Inf consistently across transpose paths and tile sizes" {
    const allocator = std.testing.allocator;
    // 第 0 行 A 全为 0，B 中含 Inf：IEEE 754 下 0 * Inf = NaN，必须传播到该行所有受影响的输出（与 Accelerate 一致）
    for ([_][3]usize{ .{ 1, 1, 2 }, .{ 8, 20, 9 } }) |s| {
        const m = s[0];
        const n = s[1];
        const k = s[2];
        for ([_]bool{ false, true }) |ta| {
            for ([_]bool{ false, true }) |tb| {
                const A = try allocator.alloc(f32, m * k);
                defer allocator.free(A);
                const B = try allocator.alloc(f32, k * n);
                defer allocator.free(B);
                const C = try allocator.alloc(f32, m * n);
                defer allocator.free(C);
                @memset(A, 1.0);
                @memset(B, 1.0);
                for (0..k) |p| A[if (ta) p * m else p] = 0.0; // op(A)[0, p] = 0
                B[0] = std.math.inf(f32); // op(B)[0, 0] = Inf（B 与 Bᵀ 的首元素相同）
                cblas_sgemm_fallback(
                    CblasRowMajor,
                    if (ta) CblasTrans else CblasNoTrans,
                    if (tb) CblasTrans else CblasNoTrans,
                    @intCast(m),
                    @intCast(n),
                    @intCast(k),
                    1.0,
                    A.ptr,
                    @intCast(if (ta) m else k),
                    B.ptr,
                    @intCast(if (tb) k else n),
                    0.0,
                    C.ptr,
                    @intCast(n),
                );
                try std.testing.expect(std.math.isNan(C[0]));
                // 其余行的 op(A)[i, 0] = 1，与 Inf 相乘得 +Inf；第 0 行其余列只含有限值，结果为 0
                if (m > 1) try std.testing.expect(std.math.isPositiveInf(C[n]));
                if (n > 1) try std.testing.expectEqual(@as(f32, 0.0), C[1]);
            }
        }
    }
}
