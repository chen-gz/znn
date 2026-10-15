const std = @import("std");
const core = @import("core.zig");
const Tensor = core.Tensor;
const ops = @import("ops.zig");
const zeros = ops.zeros;
const free = ops.free;

/// 正交三角分解 (QR Decomposition, QR) 输出结果
pub const QRResult = struct {
    Q: *Tensor,
    R: *Tensor,

    pub fn deinit(self: QRResult, allocator: std.mem.Allocator) void {
        self.Q.deinit(allocator);
        self.R.deinit(allocator);
    }
};

/// 实对称特征值分解 (Symmetric Eigenvalue Decomposition, SymEig) 输出结果
pub const SymEigResult = struct {
    eigenvalues: *Tensor,
    eigenvectors: *Tensor,

    pub fn deinit(self: SymEigResult, allocator: std.mem.Allocator) void {
        self.eigenvalues.deinit(allocator);
        self.eigenvectors.deinit(allocator);
    }
};

/// 奇异值分解 (Singular Value Decomposition, SVD) 输出结果
pub const SVDResult = struct {
    U: *Tensor,
    S: *Tensor,
    Vt: *Tensor,

    pub fn deinit(self: SVDResult, allocator: std.mem.Allocator) void {
        self.U.deinit(allocator);
        self.S.deinit(allocator);
        self.Vt.deinit(allocator);
    }
};

/// 基于豪斯霍尔德变换 (Householder Transformation) 的精简正交三角分解 (Reduced QR Decomposition)
/// 对于形状为 [M, N] 的矩阵 A (K = min(M, N)):
/// Q 的形状为 [M, K], 具有标准正交列 (Q^T * Q = I_K)
/// R 的形状为 [K, N], 为上三角 (Upper Triangular) 矩阵
/// 满足 A = Q * R 且 R 的对角元素非负
pub fn qr(self: Tensor, allocator: std.mem.Allocator) !QRResult {
    if (self.shape.len != 2) return error.IncompatibleDimensions;
    const M = self.shape.dims[0];
    const N = self.shape.dims[1];
    if (M == 0 or N == 0) return error.EmptyInputs;

    const contig_ptr: ?*Tensor = if (!self.isContiguous() or self.is_view) try self.contiguous(allocator) else null;
    defer if (contig_ptr) |cp| free(allocator, cp);
    const a_data = if (contig_ptr) |cp| cp.data else self.data;

    const K = @min(M, N);

    const R_work = try allocator.alloc(f32, M * N);
    defer allocator.free(R_work);
    @memcpy(R_work, a_data);

    const Q_work = try allocator.alloc(f32, M * M);
    defer allocator.free(Q_work);
    for (0..M) |i| {
        for (0..M) |j| {
            Q_work[i * M + j] = if (i == j) 1.0 else 0.0;
        }
    }

    const v = try allocator.alloc(f32, M);
    defer allocator.free(v);
    const w = try allocator.alloc(f32, @max(M, N));
    defer allocator.free(w);

    for (0..K) |k| {
        const p = M - k;
        if (p <= 1) break;

        var sub_norm_sq: f32 = 0.0;
        for (k + 1..M) |i| {
            const val = R_work[i * N + k];
            sub_norm_sq += val * val;
        }
        if (sub_norm_sq < 1e-24) continue;

        const r_kk = R_work[k * N + k];
        const norm = @sqrt(r_kk * r_kk + sub_norm_sq);
        const alpha: f32 = if (r_kk >= 0.0) -norm else norm;
        const diff_k = r_kk - alpha;

        v[0] = diff_k;
        for (1..p) |i| {
            v[i] = R_work[(k + i) * N + k];
        }

        var v_norm_sq: f32 = 0.0;
        for (0..p) |i| {
            v_norm_sq += v[i] * v[i];
        }
        const v_norm = @sqrt(v_norm_sq);
        if (v_norm < 1e-12) continue;

        for (0..p) |i| {
            v[i] /= v_norm;
        }

        // 应用 Householder 变换于 R_work: R = H * R
        for (k..N) |j| {
            var dot: f32 = 0.0;
            for (0..p) |i| {
                dot += v[i] * R_work[(k + i) * N + j];
            }
            w[j] = dot;
        }
        for (k..N) |j| {
            const factor = 2.0 * w[j];
            for (0..p) |i| {
                R_work[(k + i) * N + j] -= factor * v[i];
            }
        }
        R_work[k * N + k] = alpha;
        for (k + 1..M) |i| {
            R_work[i * N + k] = 0.0;
        }

        // 累计 Householder 变换于 Q_work: Q = Q * H
        for (0..M) |row| {
            var dot: f32 = 0.0;
            for (0..p) |j| {
                dot += Q_work[row * M + (k + j)] * v[j];
            }
            w[row] = dot;
        }
        for (0..M) |row| {
            const factor = 2.0 * w[row];
            for (0..p) |j| {
                Q_work[row * M + (k + j)] -= factor * v[j];
            }
        }
    }

    // 规范化 R 的对角元素为非负，保证 QR 分解的标准唯一性
    for (0..K) |i| {
        if (R_work[i * N + i] < 0.0) {
            for (i..N) |j| {
                R_work[i * N + j] = -R_work[i * N + j];
            }
            for (0..M) |row| {
                Q_work[row * M + i] = -Q_work[row * M + i];
            }
        }
    }

    const Q = try zeros(allocator, &.{ M, K });
    for (0..M) |i| {
        for (0..K) |j| {
            Q.data[i * K + j] = Q_work[i * M + j];
        }
    }

    const R = try zeros(allocator, &.{ K, N });
    for (0..K) |i| {
        for (0..N) |j| {
            R.data[i * N + j] = if (i <= j) R_work[i * N + j] else 0.0;
        }
    }

    return QRResult{
        .Q = Q,
        .R = R,
    };
}

/// 基于雅可比旋转法 (Jacobi Rotation Method) 的实对称特征值分解 (Symmetric Eigenvalue Decomposition, SymEig)
/// 输入 A 为形状 [N, N] 的实对称矩阵
/// 返回 eigenvalues: 形状 [N], 升序排列的特征值
/// 返回 eigenvectors: 形状 [N, N], 每一列对应相应特征值的标准正交特征向量
/// 满足 A * V = V * diag(eigenvalues) 且 V^T * V = I_N
pub fn symeig(self: Tensor, allocator: std.mem.Allocator) !SymEigResult {
    if (self.shape.len != 2) return error.IncompatibleDimensions;
    const N = self.shape.dims[0];
    if (N != self.shape.dims[1]) return error.ShapeMismatch;
    if (N == 0) return error.EmptyInputs;

    const contig_ptr: ?*Tensor = if (!self.isContiguous() or self.is_view) try self.contiguous(allocator) else null;
    defer if (contig_ptr) |cp| free(allocator, cp);
    const a_data = if (contig_ptr) |cp| cp.data else self.data;

    if (N == 1) {
        const eigenvalues = try zeros(allocator, &.{1});
        eigenvalues.data[0] = a_data[0];
        const eigenvectors = try zeros(allocator, &.{ 1, 1 });
        eigenvectors.data[0] = 1.0;
        return SymEigResult{
            .eigenvalues = eigenvalues,
            .eigenvectors = eigenvectors,
        };
    }

    const D = try allocator.alloc(f32, N * N);
    defer allocator.free(D);
    @memcpy(D, a_data);

    // 对称化输入以消除微小浮点非对称误差
    for (0..N) |i| {
        for (i + 1..N) |j| {
            const avg = 0.5 * (D[i * N + j] + D[j * N + i]);
            D[i * N + j] = avg;
            D[j * N + i] = avg;
        }
    }

    const V = try allocator.alloc(f32, N * N);
    defer allocator.free(V);
    for (0..N) |i| {
        for (0..N) |j| {
            V[i * N + j] = if (i == j) 1.0 else 0.0;
        }
    }

    const max_sweeps: usize = 100;
    var sweep: usize = 0;
    while (sweep < max_sweeps) : (sweep += 1) {
        var off_diag_sum: f32 = 0.0;
        for (0..N) |i| {
            for (i + 1..N) |j| {
                off_diag_sum += @abs(D[i * N + j]);
            }
        }
        if (off_diag_sum < 1e-7) break;

        const thresh: f32 = if (sweep < 3) 0.2 * off_diag_sum / @as(f32, @floatFromInt(N * N)) else 0.0;

        for (0..N) |p| {
            for (p + 1..N) |q| {
                const d_pq = D[p * N + q];
                if (@abs(d_pq) <= thresh) continue;

                const d_pp = D[p * N + p];
                const d_qq = D[q * N + q];
                const diff = d_qq - d_pp;

                var t: f32 = undefined;
                if (@abs(d_pq) < 1e-12) {
                    t = 0.0;
                } else {
                    const tau = diff / (2.0 * d_pq);
                    if (tau >= 0.0) {
                        t = 1.0 / (tau + @sqrt(1.0 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + @sqrt(1.0 + tau * tau));
                    }
                }

                const c = 1.0 / @sqrt(1.0 + t * t);
                const s = t * c;

                D[p * N + p] -= t * d_pq;
                D[q * N + q] += t * d_pq;
                D[p * N + q] = 0.0;
                D[q * N + p] = 0.0;

                for (0..N) |r| {
                    if (r != p and r != q) {
                        const d_rp = D[r * N + p];
                        const d_rq = D[r * N + q];
                        const new_rp = c * d_rp - s * d_rq;
                        const new_rq = s * d_rp + c * d_rq;
                        D[r * N + p] = new_rp;
                        D[p * N + r] = new_rp;
                        D[r * N + q] = new_rq;
                        D[q * N + r] = new_rq;
                    }
                }

                for (0..N) |k| {
                    const v_kp = V[k * N + p];
                    const v_kq = V[k * N + q];
                    V[k * N + p] = c * v_kp - s * v_kq;
                    V[k * N + q] = s * v_kp + c * v_kq;
                }
            }
        }
    }

    const eigenvalues = try zeros(allocator, &.{N});
    for (0..N) |i| {
        eigenvalues.data[i] = D[i * N + i];
    }

    // 按特征值升序排序，并同步置换特征向量矩阵 V 的对应列
    for (0..N) |i| {
        var min_idx = i;
        for (i + 1..N) |j| {
            if (eigenvalues.data[j] < eigenvalues.data[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            const tmp_e = eigenvalues.data[i];
            eigenvalues.data[i] = eigenvalues.data[min_idx];
            eigenvalues.data[min_idx] = tmp_e;

            for (0..N) |row| {
                const tmp_v = V[row * N + i];
                V[row * N + i] = V[row * N + min_idx];
                V[row * N + min_idx] = tmp_v;
            }
        }
    }

    const eigenvectors = try zeros(allocator, &.{ N, N });
    @memcpy(eigenvectors.data, V);

    return SymEigResult{
        .eigenvalues = eigenvalues,
        .eigenvectors = eigenvectors,
    };
}

/// 针对 M >= N 矩阵的单边雅可比奇异值分解 (One-Sided Jacobi Singular Value Decomposition, One-Sided Jacobi SVD)
fn svdTall(self: Tensor, allocator: std.mem.Allocator) !SVDResult {
    const M = self.shape.dims[0];
    const N = self.shape.dims[1];
    std.debug.assert(M >= N);

    const contig_ptr: ?*Tensor = if (!self.isContiguous() or self.is_view) try self.contiguous(allocator) else null;
    defer if (contig_ptr) |cp| free(allocator, cp);
    const a_data = if (contig_ptr) |cp| cp.data else self.data;

    const B = try allocator.alloc(f32, M * N);
    defer allocator.free(B);
    @memcpy(B, a_data);

    const V = try allocator.alloc(f32, N * N);
    defer allocator.free(V);
    for (0..N) |i| {
        for (0..N) |j| {
            V[i * N + j] = if (i == j) 1.0 else 0.0;
        }
    }

    const max_sweeps: usize = 100;
    var sweep: usize = 0;
    while (sweep < max_sweeps) : (sweep += 1) {
        var max_corr: f32 = 0.0;

        for (0..N) |i| {
            for (i + 1..N) |j| {
                var p: f32 = 0.0;
                var q: f32 = 0.0;
                var r: f32 = 0.0;

                for (0..M) |k| {
                    const b_ki = B[k * N + i];
                    const b_kj = B[k * N + j];
                    p += b_ki * b_kj;
                    q += b_ki * b_ki;
                    r += b_kj * b_kj;
                }

                if (q * r < 1e-24) continue;
                const corr = @abs(p) / @sqrt(q * r);
                if (corr > max_corr) max_corr = corr;
                if (corr < 1e-6) continue;

                const tau = (r - q) / (2.0 * p);
                var t: f32 = undefined;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + @sqrt(1.0 + tau * tau));
                } else {
                    t = -1.0 / (-tau + @sqrt(1.0 + tau * tau));
                }

                const c = 1.0 / @sqrt(1.0 + t * t);
                const s = t * c;

                for (0..M) |k| {
                    const b_ki = B[k * N + i];
                    const b_kj = B[k * N + j];
                    B[k * N + i] = c * b_ki - s * b_kj;
                    B[k * N + j] = s * b_ki + c * b_kj;
                }

                for (0..N) |k| {
                    const v_ki = V[k * N + i];
                    const v_kj = V[k * N + j];
                    V[k * N + i] = c * v_ki - s * v_kj;
                    V[k * N + j] = s * v_ki + c * v_kj;
                }
            }
        }

        if (max_corr < 1e-6) break;
    }

    const S_vals = try allocator.alloc(f32, N);
    defer allocator.free(S_vals);

    for (0..N) |i| {
        var norm_sq: f32 = 0.0;
        for (0..M) |k| {
            const val = B[k * N + i];
            norm_sq += val * val;
        }
        const norm = @sqrt(norm_sq);
        S_vals[i] = norm;

        if (norm > 1e-12) {
            for (0..M) |k| {
                B[k * N + i] /= norm;
            }
        } else {
            // 格拉姆-施密特正交化 (Gram-Schmidt Orthonormalization) 填补零奇异值对应的标准正交基
            for (0..M) |k| {
                B[k * N + i] = 0.0;
            }
            for (0..M) |dim_idx| {
                for (0..M) |k| {
                    B[k * N + i] = if (k == dim_idx) 1.0 else 0.0;
                }
                for (0..i) |prev| {
                    var dot: f32 = 0.0;
                    for (0..M) |k| {
                        dot += B[k * N + i] * B[k * N + prev];
                    }
                    for (0..M) |k| {
                        B[k * N + i] -= dot * B[k * N + prev];
                    }
                }
                var rem_norm_sq: f32 = 0.0;
                for (0..M) |k| {
                    rem_norm_sq += B[k * N + i] * B[k * N + i];
                }
                const rem_norm = @sqrt(rem_norm_sq);
                if (rem_norm > 1e-4) {
                    for (0..M) |k| {
                        B[k * N + i] /= rem_norm;
                    }
                    break;
                }
            }
        }
    }

    // 按奇异值降序排列 (Singular Values Descending Order)
    for (0..N) |i| {
        var max_idx = i;
        for (i + 1..N) |j| {
            if (S_vals[j] > S_vals[max_idx]) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            const tmp_s = S_vals[i];
            S_vals[i] = S_vals[max_idx];
            S_vals[max_idx] = tmp_s;

            for (0..M) |row| {
                const tmp_u = B[row * N + i];
                B[row * N + i] = B[row * N + max_idx];
                B[row * N + max_idx] = tmp_u;
            }

            for (0..N) |row| {
                const tmp_v = V[row * N + i];
                V[row * N + i] = V[row * N + max_idx];
                V[row * N + max_idx] = tmp_v;
            }
        }
    }

    const U = try zeros(allocator, &.{ M, N });
    @memcpy(U.data, B);

    const S = try zeros(allocator, &.{N});
    @memcpy(S.data, S_vals);

    const Vt = try zeros(allocator, &.{ N, N });
    for (0..N) |i| {
        for (0..N) |j| {
            Vt.data[i * N + j] = V[j * N + i];
        }
    }

    return SVDResult{
        .U = U,
        .S = S,
        .Vt = Vt,
    };
}

/// 奇异值分解 (Singular Value Decomposition, SVD)
/// 对于形状为 [M, N] 的任意矩阵 A (K = min(M, N)):
/// U: 形状 [M, K], 具有标准正交列 (U^T * U = I_K)
/// S: 形状 [K], 降序排列的非负奇异值向量 (sigma_0 >= sigma_1 >= ... >= 0)
/// Vt: 形状 [K, N], 具有标准正交行 (Vt * Vt^T = I_K)
/// 满足 A = U * diag(S) * Vt
pub fn svd(self: Tensor, allocator: std.mem.Allocator) !SVDResult {
    if (self.shape.len != 2) return error.IncompatibleDimensions;
    const M = self.shape.dims[0];
    const N = self.shape.dims[1];
    if (M == 0 or N == 0) return error.EmptyInputs;

    if (M >= N) {
        return svdTall(self, allocator);
    }

    // 当 M < N 时，通过转置矩阵 A^T 进行求解：
    // 若 A^T = U_t * diag(S) * Vt_t，则 A = (Vt_t)^T * diag(S) * (U_t)^T
    var self_copy = self;
    const a_trans = try self_copy.transpose(0, 1, allocator);
    defer free(allocator, a_trans);

    const tall_res = try svdTall(a_trans.*, allocator);
    defer tall_res.deinit(allocator);

    const U = try zeros(allocator, &.{ M, M });
    for (0..M) |i| {
        for (0..M) |j| {
            U.data[i * M + j] = tall_res.Vt.data[j * M + i];
        }
    }

    const S = try zeros(allocator, &.{M});
    @memcpy(S.data, tall_res.S.data);

    const Vt = try zeros(allocator, &.{ M, N });
    for (0..M) |i| {
        for (0..N) |j| {
            Vt.data[i * N + j] = tall_res.U.data[j * M + i];
        }
    }

    return SVDResult{
        .U = U,
        .S = S,
        .Vt = Vt,
    };
}
