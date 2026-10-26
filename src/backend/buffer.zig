//! 物理设备内存缓冲区抽象 (Physical Device Memory Buffer Abstraction)
//!
//! 拥有具体设备上的连续无类型物理字节内存，具备 64 字节严格 SIMD 对齐，
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
};

/// 底层驱动显存/物理内存句柄联合体
pub const BufferHandle = union(enum) {
    cpu: [*]align(64) u8,
    cuda: cuda.CUdeviceptr,
    metal: mlx.Array,
};

/// 物理设备缓冲区 (Physical Device Buffer)
pub const DeviceBuffer = struct {
    device: Device,
    byte_size: usize,
    handle: BufferHandle,

    /// 物理设备缓冲区内存分配 (Physical Device Memory Allocation)
    /// - CPU: 要求传入 host_allocator，分配 64 字节对齐内存。
    /// - CUDA: 经由 CUDA Driver API 分配显存。
    /// - Metal: 经由 MLX Metal 驱动分配设备内存。
    pub fn alloc(
        device: Device,
        byte_size: usize,
        host_allocator: ?std.mem.Allocator,
    ) BufferError!DeviceBuffer {
        if (byte_size == 0) return error.InvalidBufferSize;
        if (!device_mod.isAvailable(device)) return error.DeviceNotAvailable;

        switch (device.type) {
            .cpu => {
                const allocator = host_allocator orelse return error.HostAllocatorRequired;
                const slice = allocator.alignedAlloc(u8, .@"64", byte_size) catch return error.OutOfMemory;
                return .{
                    .device = device,
                    .byte_size = byte_size,
                    .handle = .{ .cpu = slice.ptr },
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

    /// 释放物理设备显存/内存 (Free Device Memory)
    pub fn free(self: *DeviceBuffer, host_allocator: ?std.mem.Allocator) void {
        if (self.byte_size == 0) return;
        switch (self.handle) {
            .cpu => |ptr| {
                if (host_allocator) |allocator| {
                    const slice = ptr[0..self.byte_size];
                    allocator.free(slice);
                }
            },
            .cuda => |dptr| {
                cuda.free(dptr);
            },
            .metal => |*arr| {
                arr.deinit();
            },
        }
        self.byte_size = 0;
        self.handle = undefined;
    }

    /// 将主机内存数据拷贝入设备缓冲区 (Host -> Device Copy)
    pub fn copyHostToDevice(self: *DeviceBuffer, src: []const u8) BufferError!void {
        if (src.len > self.byte_size) return error.BufferOverflow;
        switch (self.handle) {
            .cpu => |ptr| {
                @memcpy(ptr[0..src.len], src);
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
                @memcpy(dst, ptr[0..dst.len]);
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
                @memcpy(dst_ptr[0..src.byte_size], src.handle.cpu[0..src.byte_size]);
            },
            .cuda => |dst_dptr| {
                cuda.copyDeviceToDevice(dst_dptr, src.handle.cuda, src.byte_size) catch return error.DeviceTransferFailed;
            },
            .metal => |*dst_arr| {
                const cloned = src.handle.metal.clone() catch return error.DeviceTransferFailed;
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
    pub fn asCpuSlice(self: *const DeviceBuffer, comptime T: type) BufferError![]align(64) const T {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        const elem_size = @sizeOf(T);
        if (self.byte_size % elem_size != 0) return error.InvalidBufferSize;
        const num_elems = self.byte_size / elem_size;
        const typed_ptr: [*]align(64) const T = @ptrCast(@alignCast(self.handle.cpu));
        return typed_ptr[0..num_elems];
    }

    /// 获取 CPU 强类型对齐可变切片 (仅限 CPU 设备)
    pub fn asCpuSliceMut(self: *DeviceBuffer, comptime T: type) BufferError![]align(64) T {
        if (self.device.type != .cpu) return error.UnsupportedDevice;
        const elem_size = @sizeOf(T);
        if (self.byte_size % elem_size != 0) return error.InvalidBufferSize;
        const num_elems = self.byte_size / elem_size;
        const typed_ptr: [*]align(64) T = @ptrCast(@alignCast(self.handle.cpu));
        return typed_ptr[0..num_elems];
    }
};

test "buffer: allocation, copy and lifecycle" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buf = try DeviceBuffer.alloc(Device.cpu, 128 * @sizeOf(f32), allocator);
    defer buf.free(allocator);

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
