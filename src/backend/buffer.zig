//! 物理设备内存缓冲区抽象 (Physical Device Memory Buffer Abstraction)
//!
//! 拥有具体设备上的连续无类型物理字节内存，宿主分配基址保持 64 字节对齐，视图按元素对齐。
//! 严格剥离任何张量语义（无形状 shape、无步长 strides、无数据类型 dtype、无自动微分拓扑）。

const std = @import("std");
const device_mod = @import("device.zig");
const Device = device_mod.Device;
const cuda = @import("cuda.zig");
const mlx = @import("mlx.zig");

pub const BufferError = error{
    OutOfMemory,
    DeviceNotAvailable,
    HostAllocatorRequired,
    InvalidBufferSize,
    BufferOverflow,
    DeviceTransferFailed,
    DeviceMismatch,
    UnsupportedDevice,
    MisalignedBuffer,
    ReferenceCountOverflow,
};

/// 底层驱动显存/物理内存句柄联合体
pub const BufferHandle = union(enum) {
    cpu: [*]u8,
    cuda: cuda.CUdeviceptr,
    metal: mlx.Array,
};

const CpuOwner = struct {
    bytes: []align(64) u8,
    allocator: std.mem.Allocator,
    references: std.atomic.Value(usize) = std.atomic.Value(usize).init(1),
};

var empty_cpu_storage: [0]u8 align(64) = .{};

/// Physical device storage. CPU buffers share an allocation only through retain()
/// or slice(); plain descriptor assignment does not acquire ownership. Each owned
/// descriptor must be freed once and must not be accessed concurrently with free().
/// Atomic reference counts protect lifetime, not concurrent access to buffer data.
pub const DeviceBuffer = struct {
    device: Device,
    byte_size: usize,
    handle: BufferHandle,
    cpu_owner: ?*CpuOwner = null,

    fn emptyCpu(device: Device) DeviceBuffer {
        return .{ .device = device, .byte_size = 0, .handle = .{ .cpu = empty_cpu_storage[0..].ptr } };
    }

    /// 物理设备缓冲区内存分配 (Physical Device Memory Allocation)
    /// - CPU: 要求传入 host_allocator，分配 64 字节对齐内存。
    /// - CUDA: 经由 CUDA Driver API 分配显存。
    /// - Metal: 经由 MLX Metal 驱动分配设备内存。
    pub fn alloc(
        device: Device,
        byte_size: usize,
        host_allocator: ?std.mem.Allocator,
    ) BufferError!DeviceBuffer {
        if (byte_size == 0) {
            if (device.type != .cpu) return error.InvalidBufferSize;
            return emptyCpu(device);
        }
        if (!device_mod.isAvailable(device)) return error.DeviceNotAvailable;

        switch (device.type) {
            .cpu => {
                const allocator = host_allocator orelse return error.HostAllocatorRequired;
                const bytes = allocator.alignedAlloc(u8, .@"64", byte_size) catch return error.OutOfMemory;
                errdefer allocator.free(bytes);
                const owner = allocator.create(CpuOwner) catch return error.OutOfMemory;
                owner.* = .{ .bytes = bytes, .allocator = allocator };
                return .{
                    .device = device,
                    .byte_size = byte_size,
                    .handle = .{ .cpu = bytes.ptr },
                    .cpu_owner = owner,
                };
            },
            .cuda => {
                const dptr = cuda.alloc(byte_size) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return error.DeviceNotAvailable,
                };
                return .{
                    .device = device,
                    .byte_size = byte_size,
                    .handle = .{ .cuda = dptr },
                };
            },
            .metal => {
                const arr = mlx.Array.zeros(&.{byte_size}, .uint8, null) catch return error.OutOfMemory;
                return .{
                    .device = device,
                    .byte_size = byte_size,
                    .handle = .{ .metal = arr },
                };
            },
        }
    }

    /// Acquire an independent reference to the same CPU storage. The allocator
    /// used for allocation must remain alive until the final reference is freed.
    pub fn retain(self: *const DeviceBuffer) BufferError!DeviceBuffer {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        if (self.byte_size == 0) return emptyCpu(self.device);
        const owner = self.cpu_owner orelse return error.InvalidBufferSize;
        var refs = owner.references.load(.monotonic);
        while (true) {
            if (refs == std.math.maxInt(usize)) return error.ReferenceCountOverflow;
            if (owner.references.cmpxchgWeak(refs, refs + 1, .monotonic, .monotonic)) |actual| {
                refs = actual;
            } else break;
        }
        return self.*;
    }

    /// Acquire a byte-range CPU view. Offset bounds are checked before addition;
    /// empty views own no allocation and typed alignment is checked on access.
    pub fn slice(self: *const DeviceBuffer, offset: usize, byte_len: usize) BufferError!DeviceBuffer {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        if (offset > self.byte_size or byte_len > self.byte_size - offset) return error.BufferOverflow;
        if (byte_len == 0) return emptyCpu(self.device);
        var view = try self.retain();
        view.handle.cpu += offset;
        view.byte_size = byte_len;
        return view;
    }

    /// Release this descriptor using its allocation's stored allocator. CPU
    /// views can outlive their parent; repeated release of this descriptor is safe.
    pub fn free(self: *DeviceBuffer) void {
        switch (self.handle) {
            .cpu => {
                if (self.cpu_owner) |owner| {
                    if (owner.references.fetchSub(1, .acq_rel) == 1) {
                        const allocator = owner.allocator;
                        allocator.free(owner.bytes);
                        allocator.destroy(owner);
                    }
                }
            },
            .cuda => |dptr| cuda.free(dptr),
            .metal => |*arr| arr.deinit(),
        }
        self.* = emptyCpu(Device.cpu);
    }

    fn copyCpu(dst: []u8, src: []const u8) void {
        if (@intFromPtr(dst.ptr) <= @intFromPtr(src.ptr)) {
            std.mem.copyForwards(u8, dst, src);
        } else {
            std.mem.copyBackwards(u8, dst, src);
        }
    }

    /// 将主机内存数据拷贝入设备缓冲区 (Host -> Device Copy)
    pub fn copyHostToDevice(self: *DeviceBuffer, src: []const u8) BufferError!void {
        if (src.len > self.byte_size) return error.BufferOverflow;
        switch (self.handle) {
            .cpu => |ptr| {
                copyCpu(ptr[0..src.len], src);
            },
            .cuda => |dptr| {
                cuda.copyHostToDevice(dptr, src.ptr, src.len) catch return error.DeviceTransferFailed;
            },
            .metal => |*arr| {
                const new_arr = mlx.Array.fromRawData(src.ptr, &.{src.len}, .uint8) catch return error.DeviceTransferFailed;
                arr.deinit();
                arr.* = new_arr;
            },
        }
    }

    /// 将设备缓冲区数据拷贝回主机内存 (Device -> Host Copy)
    pub fn copyDeviceToHost(self: *const DeviceBuffer, dst: []u8) BufferError!void {
        if (dst.len > self.byte_size) return error.BufferOverflow;
        switch (self.handle) {
            .cpu => |ptr| {
                copyCpu(dst, ptr[0..dst.len]);
            },
            .cuda => |dptr| {
                cuda.copyDeviceToHost(dst.ptr, dptr, dst.len) catch return error.DeviceTransferFailed;
            },
            .metal => |arr| {
                if (!mlx.has_mlx) return error.UnsupportedDevice;
                arr.eval();
                const src_ptr = mlx.c.mlx_array_data_uint8(arr.handle) orelse return error.DeviceTransferFailed;
                @memcpy(dst, src_ptr[0..dst.len]);
            },
        }
    }

    /// 设备内或同构设备间数据拷贝 (Device -> Device Copy)
    pub fn copyDeviceToDevice(dst: *DeviceBuffer, src: *const DeviceBuffer) BufferError!void {
        if (!dst.device.eql(src.device)) return error.DeviceMismatch;
        if (dst.byte_size < src.byte_size) return error.BufferOverflow;

        switch (dst.handle) {
            .cpu => |dst_ptr| {
                copyCpu(dst_ptr[0..src.byte_size], src.handle.cpu[0..src.byte_size]);
            },
            .cuda => |dst_dptr| {
                cuda.copyDeviceToDevice(dst_dptr, src.handle.cuda, src.byte_size) catch return error.DeviceTransferFailed;
            },
            .metal => |*dst_arr| {
                if (!mlx.has_mlx) return error.UnsupportedDevice;
                var stream = mlx.Stream.defaultGpu() catch return error.DeviceTransferFailed;
                defer stream.deinit();
                var cloned = mlx.Array.init();
                errdefer cloned.deinit();
                if (mlx.c.mlx_copy(&cloned.handle, src.handle.metal.handle, stream.handle) != 0 or cloned.handle.ctx == null)
                    return error.DeviceTransferFailed;
                dst_arr.deinit();
                dst_arr.* = cloned;
            },
        }
    }

    /// 缓冲区清零 (Zero Memory Buffer)
    pub fn setZero(self: *DeviceBuffer) BufferError!void {
        switch (self.handle) {
            .cpu => |ptr| {
                @memset(ptr[0..self.byte_size], 0);
            },
            .cuda => |dptr| {
                cuda.setZero(dptr, self.byte_size) catch return error.DeviceTransferFailed;
            },
            .metal => |*arr| {
                const new_arr = mlx.Array.zeros(&.{self.byte_size}, .uint8, null) catch return error.DeviceTransferFailed;
                arr.deinit();
                arr.* = new_arr;
            },
        }
    }

    /// 获取 CPU 强类型对齐只读切片 (仅限 CPU 设备)
    pub fn asCpuSlice(self: *const DeviceBuffer, comptime T: type) BufferError![]const T {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        const elem_size = @sizeOf(T);
        if (elem_size == 0) return error.InvalidBufferSize;
        if (self.byte_size % elem_size != 0) return error.InvalidBufferSize;
        if (@intFromPtr(self.handle.cpu) % @alignOf(T) != 0) return error.MisalignedBuffer;
        const num_elems = self.byte_size / elem_size;
        const typed_ptr: [*]const T = @ptrCast(@alignCast(self.handle.cpu));
        return typed_ptr[0..num_elems];
    }

    /// 获取 CPU 强类型对齐可变切片 (仅限 CPU 设备)
    pub fn asCpuSliceMut(self: *DeviceBuffer, comptime T: type) BufferError![]T {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        const elem_size = @sizeOf(T);
        if (elem_size == 0) return error.InvalidBufferSize;
        if (self.byte_size % elem_size != 0) return error.InvalidBufferSize;
        if (@intFromPtr(self.handle.cpu) % @alignOf(T) != 0) return error.MisalignedBuffer;
        const num_elems = self.byte_size / elem_size;
        const typed_ptr: [*]T = @ptrCast(@alignCast(self.handle.cpu));
        return typed_ptr[0..num_elems];
    }
};

test "buffer: allocation, copy and lifecycle" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buf = try DeviceBuffer.alloc(Device.cpu, 128 * @sizeOf(f32), allocator);
    defer buf.free();

    try testing.expectEqual(@as(usize, 128 * @sizeOf(f32)), buf.byte_size);

    const slice_mut = try buf.asCpuSliceMut(f32);
    for (slice_mut, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    const slice = try buf.asCpuSlice(f32);
    try testing.expectEqual(@as(f32, 0.0), slice[0]);
    try testing.expectEqual(@as(f32, 42.0), slice[42]);

    try buf.setZero();
    try testing.expectEqual(@as(f32, 0.0), slice[42]);
}

test "buffer: disabled MLX rejects Metal access" {
    if (mlx.has_mlx) return error.SkipZigTest;
    const testing = std.testing;
    try testing.expect(!device_mod.isAvailable(Device.metal));
    try testing.expectError(error.DeviceNotAvailable, DeviceBuffer.alloc(Device.metal, 1, testing.allocator));

    // A disabled backend must reject a stale handle without touching its payload.
    const buffer = DeviceBuffer{
        .device = Device.metal,
        .byte_size = 1,
        .handle = .{ .metal = .{ .handle = null } },
    };
    var output = [_]u8{42};
    try testing.expectError(error.UnsupportedDevice, buffer.copyDeviceToHost(&output));
    try testing.expectEqual(@as(u8, 42), output[0]);
}

test "buffer: retained CPU views outlive parents in either release order" {
    const testing = std.testing;
    var root = try DeviceBuffer.alloc(Device.cpu, 16, testing.allocator);
    defer root.free();
    try root.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
    var retained = try root.retain();
    defer retained.free();
    var view = try root.slice(4, 8);
    defer view.free();
    var nested = try view.slice(2, 4);
    defer nested.free();

    root.free();
    root.free();
    view.free();
    try testing.expectEqualSlices(u8, &.{ 6, 7, 8, 9 }, try nested.asCpuSlice(u8));
    try nested.copyHostToDevice(&.{ 30, 31, 32, 33 });
    try testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 10, 11, 12, 13, 14, 15 }, try retained.asCpuSlice(u8));
    nested.free();
    try testing.expectEqual(@as(u8, 30), (try retained.asCpuSlice(u8))[6]);
    var short_lived = try retained.slice(0, 4);
    short_lived.free();
    try retained.setZero();
    for (try retained.asCpuSlice(u8)) |byte| try testing.expectEqual(@as(u8, 0), byte);
}

test "buffer: empty CPU storage needs no allocator and remains safe after release" {
    const testing = std.testing;
    var empty = try DeviceBuffer.alloc(Device.cpu, 0, null);
    defer empty.free();
    try testing.expectEqual(@as(usize, 0), (try empty.asCpuSlice(f32)).len);
    try testing.expectEqual(@as(usize, 0), (try empty.asCpuSliceMut(u64)).len);
    var retained = try empty.retain();
    defer retained.free();
    var view = try empty.slice(0, 0);
    defer view.free();
    try empty.copyHostToDevice(&.{});
    var output: [0]u8 = .{};
    try empty.copyDeviceToHost(&output);
    try DeviceBuffer.copyDeviceToDevice(&view, &empty);
    try empty.setZero();
    empty.free();
    empty.free();
    try testing.expectEqual(@as(usize, 0), (try empty.asCpuSlice(u8)).len);
    try testing.expectError(error.BufferOverflow, empty.copyHostToDevice(&.{1}));
    try testing.expectError(error.BufferOverflow, empty.slice(1, 0));
    try testing.expectError(error.HostAllocatorRequired, DeviceBuffer.alloc(Device.cpu, 1, null));

    var parent = try DeviceBuffer.alloc(Device.cpu, 8, testing.allocator);
    var end = try parent.slice(8, 0);
    parent.free();
    defer end.free();
    try testing.expectEqual(@as(usize, 0), (try end.asCpuSlice(u8)).len);
}

test "buffer: CPU offsets enforce element alignment and checked bounds" {
    const testing = std.testing;
    var owner = try DeviceBuffer.alloc(Device.cpu, 16, testing.allocator);
    defer owner.free();
    try owner.setZero();
    try testing.expectEqual(@as(usize, 0), @intFromPtr(owner.handle.cpu) % 64);
    var aligned = try owner.slice(4, 8);
    defer aligned.free();
    const floats = try aligned.asCpuSliceMut(f32);
    floats[0] = 3.5;
    floats[1] = 8.5;
    try testing.expectEqualSlices(f32, &.{ 0, 3.5, 8.5, 0 }, try owner.asCpuSlice(f32));
    var misaligned = try owner.slice(1, 4);
    defer misaligned.free();
    try testing.expectError(error.MisalignedBuffer, misaligned.asCpuSlice(f32));
    try testing.expectError(error.MisalignedBuffer, misaligned.asCpuSliceMut(f32));
    try testing.expectEqual(@as(usize, 4), (try misaligned.asCpuSlice(u8)).len);
    var short = try owner.slice(0, 3);
    defer short.free();
    try testing.expectError(error.InvalidBufferSize, short.asCpuSlice(f32));
    try testing.expectError(error.InvalidBufferSize, owner.asCpuSlice(void));
    try testing.expectError(error.InvalidBufferSize, owner.asCpuSliceMut(void));
    try testing.expectError(error.BufferOverflow, owner.slice(17, 0));
    try testing.expectError(error.BufferOverflow, owner.slice(15, 2));
    try testing.expectError(error.BufferOverflow, owner.slice(std.math.maxInt(usize), 1));
    try testing.expectError(error.BufferOverflow, owner.slice(1, std.math.maxInt(usize)));
}

test "buffer: CPU view copies handle aliases and both overlap directions" {
    const testing = std.testing;
    var owner = try DeviceBuffer.alloc(Device.cpu, 8, testing.allocator);
    defer owner.free();
    var left = try owner.slice(0, 6);
    defer left.free();
    var right = try owner.slice(2, 6);
    defer right.free();
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try DeviceBuffer.copyDeviceToDevice(&right, &left);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 2, 3, 4, 5 }, try owner.asCpuSlice(u8));
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try DeviceBuffer.copyDeviceToDevice(&left, &right);
    try testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5, 6, 7, 6, 7 }, try owner.asCpuSlice(u8));
    try DeviceBuffer.copyDeviceToDevice(&owner, &owner);
    try testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5, 6, 7, 6, 7 }, try owner.asCpuSlice(u8));
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try right.copyHostToDevice(try left.asCpuSlice(u8));
    try testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 2, 3, 4, 5 }, try owner.asCpuSlice(u8));
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try left.copyHostToDevice(try right.asCpuSlice(u8));
    try owner.copyHostToDevice(try owner.asCpuSlice(u8));
    try testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5, 6, 7, 6, 7 }, try owner.asCpuSlice(u8));
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try left.copyDeviceToHost(try right.asCpuSliceMut(u8));
    try testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 2, 3, 4, 5 }, try owner.asCpuSlice(u8));
    try owner.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    try right.copyDeviceToHost(try left.asCpuSliceMut(u8));
    try owner.copyDeviceToHost(try owner.asCpuSliceMut(u8));
    try testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5, 6, 7, 6, 7 }, try owner.asCpuSlice(u8));
    try testing.expectError(error.BufferOverflow, right.copyHostToDevice(&.{ 0, 1, 2, 3, 4, 5, 6 }));
    try testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5, 6, 7, 6, 7 }, try owner.asCpuSlice(u8));
}

fn allocationFailureLifecycle(allocator: std.mem.Allocator) !void {
    var root = try DeviceBuffer.alloc(Device.cpu, 16, allocator);
    defer root.free();
    var view = try root.slice(4, 8);
    defer view.free();
    var retained = try view.retain();
    defer retained.free();
    root.free();
    view.free();
    try retained.setZero();
    try std.testing.expectEqual(@as(usize, 8), (try retained.asCpuSlice(u8)).len);
}

test "buffer: CPU allocation failures release bytes and owner metadata" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureLifecycle, .{});
}

test "buffer: GPU retention and slicing are explicitly unsupported" {
    // Metadata-only probe: neither operation may dereference this device handle.
    const gpu = DeviceBuffer{ .device = Device.cuda, .byte_size = 16, .handle = .{ .cuda = 0 } };
    try std.testing.expectError(error.UnsupportedDevice, gpu.retain());
    try std.testing.expectError(error.UnsupportedDevice, gpu.slice(0, 4));
}
