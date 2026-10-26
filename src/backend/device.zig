//! 计算设备抽象与硬件探测 (Device Abstraction & Hardware Detection)
//!
//! 定义具体目标计算设备 (CPU, CUDA, Metal)，支持多卡索引 (.index: u32 = 0)，
//! 完全剔除含糊不清的 .gpu 枚举，与具体驱动实现解耦。

const std = @import("std");
const cuda = @import("cuda.zig");
const mlx = @import("mlx.zig");
const offload = @import("offload.zig");

/// 目标计算设备硬件类型枚举 (Hardware Device Type)
pub const DeviceType = enum {
    cpu,
    cuda,
    metal,

    pub fn isCpu(self: DeviceType) bool {
        return self == .cpu;
    }

    pub fn isCuda(self: DeviceType) bool {
        return self == .cuda;
    }

    pub fn isMetal(self: DeviceType) bool {
        return self == .metal;
    }

    pub fn isGpu(self: DeviceType) bool {
        return self == .cuda or self == .metal;
    }
};

/// 计算设备结构体 (Compute Device with Ordinal)
/// 遵循 AGENTS.md §4 规范：默认字段值提供开箱即用，暴露 defaultOptions()
pub const Device = struct {
    type: DeviceType = .cpu,
    index: u32 = 0,

    pub const cpu: Device = .{ .type = .cpu, .index = 0 };
    pub const cuda: Device = .{ .type = .cuda, .index = 0 };
    pub const metal: Device = .{ .type = .metal, .index = 0 };

    pub fn defaultOptions() Device {
        return defaultDevice();
    }

    pub fn isCpu(self: Device) bool {
        return self.type.isCpu();
    }

    pub fn isCuda(self: Device) bool {
        return self.type.isCuda();
    }

    pub fn isMetal(self: Device) bool {
        return self.type.isMetal();
    }

    pub fn isGpu(self: Device) bool {
        return self.type.isGpu();
    }

    pub fn eql(self: Device, other: Device) bool {
        return self.type == other.type and self.index == other.index;
    }

    pub fn format(
        self: Device,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{d}", .{ @tagName(self.type), self.index });
    }
};

/// 跨硬件统一设备显存数组联合体 (Unified Device Memory Array Union)
pub const DeviceArray = union(enum) {
    mlx: mlx.Array,
    cuda: cuda.Array,

    pub fn deinit(self: *DeviceArray) void {
        switch (self.*) {
            .mlx => |*arr| arr.deinit(),
            .cuda => |*arr| arr.deinit(),
        }
    }

    pub fn clone(self: DeviceArray) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = arr },
            .cuda => |arr| .{ .cuda = try arr.clone() },
        };
    }

    pub fn copyToSlice(self: DeviceArray, dest: []f32) !void {
        switch (self) {
            .mlx => |arr| try arr.copyToSlice(dest),
            .cuda => |arr| try arr.toHostSlice(dest),
        }
    }

    pub fn eval(self: DeviceArray) void {
        switch (self) {
            .mlx => |arr| arr.eval(),
            .cuda => {},
        }
    }

    pub fn reshape(self: DeviceArray, new_dims: []const usize, stream_opt: ?mlx.Stream) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = try arr.reshape(new_dims, stream_opt) },
            .cuda => |arr| .{ .cuda = try arr.reshape(new_dims) },
        };
    }

    pub fn transposeAxes(self: DeviceArray, axes: []const usize, stream_opt: ?mlx.Stream) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = try arr.transposeAxes(axes, stream_opt) },
            .cuda => error.OperationNotSupported,
        };
    }

    pub fn sliceAxis(self: DeviceArray, axis: usize, start: usize, end: usize, stream_opt: ?mlx.Stream) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = try arr.sliceAxis(axis, start, end, stream_opt) },
            .cuda => error.OperationNotSupported,
        };
    }

    pub fn isMlx(self: DeviceArray) bool {
        return self == .mlx;
    }

    pub fn isCuda(self: DeviceArray) bool {
        return self == .cuda;
    }

    pub fn asMlx(self: DeviceArray) ?mlx.Array {
        return switch (self) {
            .mlx => |arr| arr,
            .cuda => null,
        };
    }

    pub fn asCuda(self: DeviceArray) ?cuda.Array {
        return switch (self) {
            .cuda => |arr| arr,
            .mlx => null,
        };
    }
};

/// 跨硬件统一设备键值缓存联合体 (Unified Device Key-Value Cache Union)
pub const DeviceKvCache = union(enum) {
    mlx: offload.GpuKvCache,
    cuda: cuda.CudaKvCache,

    pub fn deinit(self: *DeviceKvCache) void {
        switch (self.*) {
            .mlx => |*c| c.deinit(),
            .cuda => |*c| c.deinit(),
        }
    }

    pub fn reset(self: *DeviceKvCache) void {
        switch (self.*) {
            .mlx => |*c| c.reset(),
            .cuda => |*c| c.reset(),
        }
    }

    pub fn truncate(self: *DeviceKvCache, new_len: usize) void {
        switch (self.*) {
            .mlx => |*c| c.truncate(new_len),
            .cuda => |*c| c.truncate(new_len),
        }
    }

    pub fn isMlx(self: DeviceKvCache) bool {
        return self == .mlx;
    }

    pub fn isCuda(self: DeviceKvCache) bool {
        return self == .cuda;
    }

    pub fn asMlx(self: *DeviceKvCache) ?*offload.GpuKvCache {
        return switch (self.*) {
            .mlx => |*c| c,
            .cuda => null,
        };
    }

    pub fn asCuda(self: *DeviceKvCache) ?*cuda.CudaKvCache {
        return switch (self.*) {
            .cuda => |*c| c,
            .mlx => null,
        };
    }
};

/// 通用键值缓存别名
pub const KvCache = DeviceKvCache;

/// 底层实际生效的硬件加速后端枚举
pub const BackendType = enum {
    cpu,
    cuda,
    mlx,
};

/// 检查指定设备在当前平台是否可用
pub fn isAvailable(device: Device) bool {
    return switch (device.type) {
        .cpu => true,
        .cuda => cuda.isAvailable() and device.index < cuda.deviceCount(),
        .metal => mlx.has_mlx and device.index == 0,
    };
}

/// 检查指定设备类型在当前平台是否可用
pub fn isTypeAvailable(dev_type: DeviceType) bool {
    return switch (dev_type) {
        .cpu => true,
        .cuda => cuda.isAvailable(),
        .metal => mlx.has_mlx,
    };
}

/// 获取系统默认首选计算设备 (CUDA > Metal > CPU)
pub fn defaultDevice() Device {
    if (cuda.isAvailable()) return Device.cuda;
    if (mlx.has_mlx) return Device.metal;
    return Device.cpu;
}

/// 获取首选 GPU 设备 (CUDA 或 Metal，若无 GPU 可用返回 null)
pub fn defaultGpu() ?Device {
    if (cuda.isAvailable()) return Device.cuda;
    if (mlx.has_mlx) return Device.metal;
    return null;
}

/// 获取指定设备实际分派的底层硬件后端类型
pub fn activeBackend(dev: Device) BackendType {
    return switch (dev.type) {
        .cpu => .cpu,
        .cuda => if (cuda.isAvailable()) .cuda else .cpu,
        .metal => if (mlx.has_mlx) .mlx else .cpu,
    };
}

test "device: detection and conversions" {
    const dev = defaultDevice();
    if (cuda.isAvailable()) {
        try std.testing.expectEqual(Device.cuda, dev);
        try std.testing.expect(isAvailable(Device.cuda));
        try std.testing.expect(isTypeAvailable(.cuda));
        try std.testing.expect(dev.isGpu());
        try std.testing.expectEqual(BackendType.cuda, activeBackend(Device.cuda));
    } else if (mlx.has_mlx) {
        try std.testing.expectEqual(Device.metal, dev);
        try std.testing.expect(isAvailable(Device.metal));
        try std.testing.expect(dev.isGpu());
        try std.testing.expectEqual(BackendType.mlx, activeBackend(Device.metal));
    } else {
        try std.testing.expectEqual(Device.cpu, dev);
        try std.testing.expect(isAvailable(Device.cpu));
        try std.testing.expect(dev.isCpu());
        try std.testing.expectEqual(BackendType.cpu, activeBackend(Device.cpu));
    }
}
