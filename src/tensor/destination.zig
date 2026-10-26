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

fn binaryInto(dest: *Tensor, a: *const Tensor, b: *const Tensor, comptime op: fn (f32, f32) f32) TensorError!void {
    const dest_layout = try validate(dest);
    const a_layout = try validate(a);
    const b_layout = try validate(b);
    const output_shape = shape.broadcastShapes(a.shape, b.shape) catch return error.ShapeMismatch;
    if (!dest.shape.eq(output_shape)) return error.ShapeMismatch;
    if (dest.is_graph_view) return error.InPlaceOpOnGraphTensor;
    if (dest_layout.count == 0) return;
    if (!shape.isContiguousStrides(dest.shape, dest.strides)) return error.NonContiguousLayout;
    if (dest_layout.count > dest.data.len) return error.InvalidStorage;
    for ([_]*const Tensor{ a, b }, [_]usize{ a_layout.span, b_layout.span }) |source, span| {
        if (try overlaps(dest.data, dest_layout.count, source.data, span)) {
            if (isGraphTensor(dest) or isGraphTensor(source)) return error.InPlaceOpOnGraphTensor;
            return error.AliasedOutput;
        }
    }
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
