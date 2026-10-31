//! Allocation-free arithmetic into caller-owned contiguous output storage.
const std = @import("std");
const backend = @import("../backend.zig");
const core = @import("core.zig");
const shape = @import("shape.zig");
const Tensor = core.Tensor;
const TensorError = @import("types.zig").TensorError;

const Layout = struct { count: usize, span: usize };

fn validate(t: *const Tensor) TensorError!Layout {
    if (!t.storage.device.eql(backend.Device.cpu) or t.storage.typed_owner != null) return error.DeviceMismatch;
    if (t.shape.len > t.shape.dims.len) return error.DimensionOutOfBounds;
    if (t.strides.len != t.shape.len) return error.StrideMismatch;
    for (t.shape.dims[0..t.shape.len]) |dim| {
        if (dim == 0) return .{ .count = 0, .span = 0 };
    }
    var count: usize = 1;
    var last: usize = 0;
    for (t.shape.dims[0..t.shape.len], t.strides.dims[0..t.strides.len]) |dim, stride| {
        count = std.math.mul(usize, count, dim) catch return error.InvalidStorage;
        // The row iterator also computes stride * dimension when advancing.
        _ = std.math.mul(usize, stride, dim) catch return error.StrideMismatch;
        const extent = std.math.mul(usize, dim - 1, stride) catch return error.StrideMismatch;
        last = std.math.add(usize, last, extent) catch return error.StrideMismatch;
    }
    const span = std.math.add(usize, last, 1) catch return error.InvalidStorage;
    if (span > t.data.len) return error.InvalidStorage;
    return .{ .count = count, .span = span };
}

fn overlaps(a: []const f32, a_span: usize, b: []const f32, b_span: usize) TensorError!bool {
    if (a_span == 0 or b_span == 0) return false;
    const a_start = @intFromPtr(a.ptr);
    const b_start = @intFromPtr(b.ptr);
    const a_bytes = std.math.mul(usize, a_span, @sizeOf(f32)) catch return error.InvalidStorage;
    const b_bytes = std.math.mul(usize, b_span, @sizeOf(f32)) catch return error.InvalidStorage;
    const a_end = std.math.add(usize, a_start, a_bytes) catch return error.InvalidStorage;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return error.InvalidStorage;
    return a_start < b_end and b_start < a_end;
}

fn isGraphTensor(t: *const Tensor) bool {
    return t.requires_grad or t.creator != null or t.is_graph_view;
}

// Validate every source before calling this helper. No output is touched here.
fn validateOutput(dest: *const Tensor, layout: Layout, sources: []const *const Tensor, layouts: []const Layout) TensorError!void {
    if (dest.is_graph_view) return error.InPlaceOpOnGraphTensor;
    if (layout.count == 0) return;
    if (!shape.isContiguousStrides(dest.shape, dest.strides)) return error.NonContiguousLayout;
    if (layout.count > dest.data.len) return error.InvalidStorage;
    for (sources, layouts) |source, source_layout| {
        if (try overlaps(dest.data, layout.count, source.data, source_layout.span)) {
            if (isGraphTensor(dest) or isGraphTensor(source)) return error.InPlaceOpOnGraphTensor;
            return error.AliasedOutput;
        }
    }
}

fn binaryInto(dest: *Tensor, a: *const Tensor, b: *const Tensor, comptime op: fn (f32, f32) f32) TensorError!void {
    const dest_layout = try validate(dest);
    const a_layout = try validate(a);
    const b_layout = try validate(b);
    const output_shape = shape.broadcastShapes(a.shape, b.shape) catch return error.ShapeMismatch;
    if (!dest.shape.eq(output_shape)) return error.ShapeMismatch;
    try validateOutput(dest, dest_layout, &.{ a, b }, &.{ a_layout, b_layout });
    if (dest_layout.count == 0) return;
    shape.broadcastBinaryOpRaw(dest.data[0..dest_layout.count], dest.shape, a.data, a.shape, a.strides, b.data, b.shape, b.strides, op);
}

/// Add into an independent contiguous destination without allocation or device
/// transfer. Inputs may be strided and broadcast. Overlapping storage spans,
/// including the gaps in strided views, are rejected before any output write.
/// Independent graph-owned outputs are supported for subsequent graph replay.
pub fn addInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    return binaryInto(dest, a, b, struct {
        fn apply(x: f32, y: f32) f32 {
            return x + y;
        }
    }.apply);
}

/// Multiply with the same allocation, layout, and aliasing contract as addInto.
pub fn mulInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    return binaryInto(dest, a, b, struct {
        fn apply(x: f32, y: f32) f32 {
            return x * y;
        }
    }.apply);
}

/// Two-dimensional matrix multiplication into independent contiguous CPU storage.
/// Both inputs may be strided. A zero contraction dimension writes zeros.
pub fn matmulInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    const dl = try validate(dest);
    const al = try validate(a);
    const bl = try validate(b);
    if (a.shape.len != 2 or b.shape.len != 2) return error.IncompatibleDimensions;
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];
    if (k != b.shape.dims[0] or !dest.shape.eq(shape.Shape.init(&.{ m, n }))) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ a, b }, &.{ al, bl });
    if (dl.count == 0) return;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |l| sum += a.data[i * a.strides.dims[0] + l * a.strides.dims[1]] * b.data[l * b.strides.dims[0] + j * b.strides.dims[1]];
            dest.data[i * n + j] = sum;
        }
    }
}

/// Root mean square normalization (RMSNorm) along the last axis, without
/// allocation. Optional weights have shape [dim] and may themselves be strided.
pub fn rmsNormInto(
    dest: *Tensor,
    x: *const Tensor,
    weight: ?*const Tensor,
    options: @import("types.zig").RmsNormOptions,
    save_inv_rms: ?[]f32,
) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    const wl = if (weight) |w| try validate(w) else Layout{ .count = 0, .span = 0 };
    if (x.shape.len == 0) return error.IncompatibleDimensions;
    const dim = x.shape.dims[x.shape.len - 1];
    if (dim == 0) return error.DivisionByZero;
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    if (weight) |w| {
        if (w.shape.len != 1 or w.shape.dims[0] != dim) return error.ShapeMismatch;
    }
    const M = dl.count / dim;
    if (save_inv_rms) |s| if (s.len != M) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (weight) |w| try validateOutput(dest, dl, &.{w}, &.{wl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    const w_stride = if (weight) |w| (if (w.shape.len > 0) w.strides.dims[w.shape.len - 1] else 1) else 0;
    for (0..M) |i| {
        const row = out[i * dim ..][0..dim];
        const inv = 1.0 / @sqrt(backend.vec_math.sumSquares(row) / @as(f32, @floatFromInt(dim)) + options.eps);
        if (save_inv_rms) |s| s[i] = inv;
        for (row, 0..) |*v, j| {
            v.* *= inv;
            if (weight) |w| v.* *= w.data[j * w_stride] + options.weight_offset;
        }
    }
}

/// Stable softmax along an explicit axis, or the last axis when null.
/// Empty outer dimensions are valid; an empty normalization axis is rejected.
pub fn softmaxInto(dest: *Tensor, x: *const Tensor, axis: ?usize) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (x.shape.len == 0) return error.IncompatibleDimensions;
    const ax = axis orelse x.shape.len - 1;
    if (ax >= x.shape.len) return error.DimensionOutOfBounds;
    const dim = x.shape.dims[ax];
    if (dim == 0) return error.DivisionByZero;
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    var inner: usize = 1;
    for (x.shape.dims[ax + 1 .. x.shape.len]) |d| inner *= d;
    const outer = dl.count / dim / inner;
    if (inner == 1) {
        for (0..outer) |i| backend.vec_math.softmaxRowInPlace(out[i * dim ..][0..dim]);
        return;
    }
    for (0..outer) |i| {
        for (0..inner) |j| {
            const base = i * dim * inner + j;
            var max: f32 = -std.math.inf(f32);
            for (0..dim) |d| max = @max(max, out[base + d * inner]);
            var sum: f32 = 0;
            for (0..dim) |d| {
                const idx = base + d * inner;
                out[idx] = @exp(out[idx] - max);
                sum += out[idx];
            }
            const inv = 1.0 / sum;
            for (0..dim) |d| out[base + d * inner] *= inv;
        }
    }
}

/// Gaussian error linear unit (GELU), exact or tanh approximation, into an
/// independent contiguous destination. Scalar and empty inputs are supported.
pub fn geluInto(dest: *Tensor, x: *const Tensor, approximation: backend.GeluApproximation) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    switch (approximation) {
        inline else => |approx| for (out) |*v| {
            v.* = backend.activation.gelu(approx, v.*);
        },
    }
}

/// Subtract with the same allocation, layout, and aliasing contract as addInto.
pub fn subInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    return binaryInto(dest, a, b, struct {
        fn apply(x: f32, y: f32) f32 {
            return x - y;
        }
    }.apply);
}

/// Divide with the same allocation, layout, and aliasing contract as addInto.
pub fn divInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    return binaryInto(dest, a, b, struct {
        fn apply(x: f32, y: f32) f32 {
            return x / y;
        }
    }.apply);
}

fn unaryInto(dest: *Tensor, x: *const Tensor, comptime op: fn (f32) f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    for (out) |*v| {
        v.* = op(v.*);
    }
}

pub fn reluInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return if (v > 0) v else 0;
        }
    }.apply);
}

pub fn sigmoidInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return 1.0 / (1.0 + @exp(-v));
        }
    }.apply);
}

pub fn tanhInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            const exp2x = @exp(2.0 * v);
            return (exp2x - 1.0) / (exp2x + 1.0);
        }
    }.apply);
}

pub fn siluInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    }.apply);
}

pub fn sqrtInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return @sqrt(v);
        }
    }.apply);
}

pub fn expInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return @exp(v);
        }
    }.apply);
}

pub fn logInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return @log(v);
        }
    }.apply);
}

pub fn absInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return @abs(v);
        }
    }.apply);
}

pub fn negInto(dest: *Tensor, x: *const Tensor) TensorError!void {
    return unaryInto(dest, x, struct {
        fn apply(v: f32) f32 {
            return -v;
        }
    }.apply);
}

pub fn softcapInto(dest: *Tensor, x: *const Tensor, cap: f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    const inv_cap = 1.0 / cap;
    for (out) |*v| {
        const scaled = v.* * inv_cap;
        const exp2x = @exp(2.0 * scaled);
        const th = (exp2x - 1.0) / (exp2x + 1.0);
        v.* = cap * th;
    }
}

pub fn leakyReluInto(dest: *Tensor, x: *const Tensor, alpha: f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    for (out) |*v| {
        v.* = if (v.* > 0) v.* else alpha * v.*;
    }
}

fn scalarInto(dest: *Tensor, x: *const Tensor, scalar: f32, comptime op: fn (f32, f32) f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    for (out) |*v| {
        v.* = op(v.*, scalar);
    }
}

pub fn mulScalarInto(dest: *Tensor, x: *const Tensor, val: f32) TensorError!void {
    return scalarInto(dest, x, val, struct {
        fn apply(a: f32, b: f32) f32 {
            return a * b;
        }
    }.apply);
}

pub fn divScalarInto(dest: *Tensor, x: *const Tensor, val: f32) TensorError!void {
    return scalarInto(dest, x, val, struct {
        fn apply(a: f32, b: f32) f32 {
            return a / b;
        }
    }.apply);
}

pub fn addScalarInto(dest: *Tensor, x: *const Tensor, val: f32) TensorError!void {
    return scalarInto(dest, x, val, struct {
        fn apply(a: f32, b: f32) f32 {
            return a + b;
        }
    }.apply);
}

pub fn subScalarInto(dest: *Tensor, x: *const Tensor, val: f32) TensorError!void {
    return scalarInto(dest, x, val, struct {
        fn apply(a: f32, b: f32) f32 {
            return a - b;
        }
    }.apply);
}

pub fn copyInto(dest: *Tensor, src: *const Tensor) TensorError!void {
    const dl = try validate(dest);
    const sl = try validate(src);
    if (dl.count != sl.count) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{src}, &.{sl});
    if (dl.count == 0) return;
    shape.gatherStrided(dest.data[0..dl.count], src.data, src.shape, src.strides);
}

pub fn transposeInto(dest: *Tensor, src: *const Tensor, dim0: usize, dim1: usize) TensorError!void {
    if (dim0 >= src.shape.len or dim1 >= src.shape.len) return error.DimensionOutOfBounds;
    var t_shape = src.shape;
    var t_strides = src.strides;
    std.mem.swap(usize, &t_shape.dims[dim0], &t_shape.dims[dim1]);
    std.mem.swap(usize, &t_strides.dims[dim0], &t_strides.dims[dim1]);
    const dl = try validate(dest);
    const sl = try validate(src);
    if (!dest.shape.eq(t_shape)) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{src}, &.{sl});
    if (dl.count == 0) return;
    shape.gatherStrided(dest.data[0..dl.count], src.data, t_shape, t_strides);
}

pub fn layerNormInto(
    dest: *Tensor,
    x: *const Tensor,
    gamma: *const Tensor,
    beta: *const Tensor,
    eps: f32,
    save_mean: ?[]f32,
    save_inv_std: ?[]f32,
) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    const gl = try validate(gamma);
    const bl = try validate(beta);
    if (x.shape.len == 0) return error.IncompatibleDimensions;
    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    if (!dest.shape.eq(x.shape)) return error.ShapeMismatch;
    if (gamma.shape.numel() != D or beta.shape.numel() != D) return error.ShapeMismatch;
    const M = dl.count / D;
    if (save_mean) |s| if (s.len != M) return error.ShapeMismatch;
    if (save_inv_std) |s| if (s.len != M) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ x, gamma, beta }, &.{ xl, gl, bl });
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    const d_f = @as(f32, @floatFromInt(D));
    const g_stride = if (gamma.shape.len > 0) gamma.strides.dims[gamma.shape.len - 1] else 1;
    const b_stride = if (beta.shape.len > 0) beta.strides.dims[beta.shape.len - 1] else 1;
    for (0..M) |i| {
        const row = out[i * D .. (i + 1) * D];
        const mean_val = backend.vec_math.sum(row) / d_f;
        const var_sum = backend.vec_math.sumSquaredDeviations(row, mean_val);
        const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);
        if (save_mean) |s| s[i] = mean_val;
        if (save_inv_std) |s| s[i] = inv_std;
        for (row, 0..) |*val, j| {
            const g = gamma.data[j * g_stride];
            const b = beta.data[j * b_stride];
            val.* = (val.* - mean_val) * inv_std * g + b;
        }
    }
}

pub fn mseLossInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) TensorError!void {
    const dl = try validate(dest);
    const al = try validate(a);
    const bl = try validate(b);
    if (!a.shape.eq(b.shape)) return error.ShapeMismatch;
    if (dest.shape.numel() != 1) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ a, b }, &.{ al, bl });
    if (al.count == 0) {
        dest.data[0] = 0;
        return;
    }
    var sum_val: f32 = 0;
    for (0..al.count) |i| {
        const diff = a.data[i] - b.data[i];
        sum_val += diff * diff;
    }
    dest.data[0] = sum_val / @as(f32, @floatFromInt(al.count));
}

pub fn bceLossInto(dest: *Tensor, p_in: *const Tensor, targets: *const Tensor, eps: f32) TensorError!void {
    const dl = try validate(dest);
    const pl = try validate(p_in);
    const tl = try validate(targets);
    if (!p_in.shape.eq(targets.shape)) return error.ShapeMismatch;
    if (dest.shape.numel() != 1) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ p_in, targets }, &.{ pl, tl });
    const N = pl.count;
    if (N == 0) {
        dest.data[0] = 0;
        return;
    }
    var total_loss: f32 = 0.0;
    for (0..N) |i| {
        const p = p_in.data[i];
        const y = targets.data[i];
        const p_clip = @max(p, eps);
        const one_minus_p_clip = @max(1.0 - p, eps);
        total_loss += -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
    }
    dest.data[0] = total_loss / @as(f32, @floatFromInt(N));
}

pub fn bceWithLogitsLossInto(dest: *Tensor, logits: *const Tensor, targets: *const Tensor) TensorError!void {
    const dl = try validate(dest);
    const ll = try validate(logits);
    const tl = try validate(targets);
    if (!logits.shape.eq(targets.shape)) return error.ShapeMismatch;
    if (dest.shape.numel() != 1) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ logits, targets }, &.{ ll, tl });
    const N = ll.count;
    if (N == 0) {
        dest.data[0] = 0;
        return;
    }
    var total_loss: f32 = 0.0;
    for (0..N) |i| {
        const x = logits.data[i];
        const y = targets.data[i];
        const max_x = @max(x, 0.0);
        const abs_x = @abs(x);
        total_loss += max_x - x * y + @log(1.0 + @exp(-abs_x));
    }
    dest.data[0] = total_loss / @as(f32, @floatFromInt(N));
}

pub fn l1LossInto(dest: *Tensor, x: *const Tensor, lambda: f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (dest.shape.numel() != 1) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    var sum_val: f32 = 0;
    for (x.data[0..xl.count]) |v| sum_val += @abs(v);
    dest.data[0] = lambda * sum_val;
}

pub fn l2LossInto(dest: *Tensor, x: *const Tensor, lambda: f32) TensorError!void {
    const dl = try validate(dest);
    const xl = try validate(x);
    if (dest.shape.numel() != 1) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{x}, &.{xl});
    var sum_val: f32 = 0;
    for (x.data[0..xl.count]) |v| sum_val += v * v;
    dest.data[0] = 0.5 * lambda * sum_val;
}

pub fn embeddingInto(dest: *Tensor, weights: *const Tensor, indices: *const Tensor) TensorError!void {
    const dl = try validate(dest);
    const wl = try validate(weights);
    const il = try validate(indices);
    if (weights.shape.len != 2) return error.IncompatibleDimensions;
    const D = weights.shape.dims[1];
    if (dl.count != il.count * D) return error.ShapeMismatch;
    try validateOutput(dest, dl, &.{ weights, indices }, &.{ wl, il });
    if (dl.count == 0) return;
    const nn_kernels = @import("nn_kernels.zig");
    try nn_kernels.embeddingLookupInto(f32, weights.data[0..wl.count], indices.data[0..il.count], D, 1.0, dest.data[0..dl.count]);
}
