//! Allocation-free arithmetic into caller-owned contiguous output storage.
const std = @import("std");
const backend = @import("../backend.zig");
const core = @import("core.zig");
const shape = @import("shape.zig");
const Tensor = core.Tensor;
const TensorError = @import("types.zig").TensorError;

const Layout = struct { count: usize, span: usize };

fn validate(t: *const Tensor) TensorError!Layout {
    if (!t.device.eql(backend.Device.cpu) or t.device_array != null) return error.DeviceMismatch;
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
pub fn rmsNormInto(dest: *Tensor, x: *const Tensor, weight: ?*const Tensor, options: @import("types.zig").RmsNormOptions) TensorError!void {
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
    try validateOutput(dest, dl, &.{x}, &.{xl});
    if (weight) |w| try validateOutput(dest, dl, &.{w}, &.{wl});
    if (dl.count == 0) return;
    const out = dest.data[0..dl.count];
    shape.gatherStrided(out, x.data, x.shape, x.strides);
    for (0..dl.count / dim) |i| {
        const row = out[i * dim ..][0..dim];
        const inv = 1.0 / @sqrt(backend.vec_math.sumSquares(row) / @as(f32, @floatFromInt(dim)) + options.eps);
        for (row, 0..) |*v, j| {
            v.* *= inv;
            if (weight) |w| v.* *= w.data[j * w.strides.dims[0]] + options.weight_offset;
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
