//! 计算设备抽象与硬件探测 (Device Abstraction & Hardware Detection)
//!
//! 定义多后端统一计算设备（中央处理器 CPU、图形处理器 GPU / 硬件加速器、统一计算设备架构 CUDA）与可用性探测，
//! 与底层具体厂商驱动（Apple MLX、NVIDIA CUDA、Vulkan 等）解耦。

const std = @import("std");
const mlx = @import("mlx.zig");
const cuda = @import("cuda.zig");
const offload = @import("offload.zig");

/// 目标计算设备类型枚举 (中央处理器 CPU 与图形处理器 GPU / CUDA)
pub const DeviceType = enum {
    cpu,
    gpu,
    cuda,

    pub fn isGpu(self: DeviceType) bool {
        return self == .gpu or self == .cuda;
    }

    pub fn isCuda(self: DeviceType) bool {
        return self == .cuda;
    }

    pub fn isCpu(self: DeviceType) bool {
        return self == .cpu;
    }

    /// 转换为底层 MLX 的 DeviceType (用于 Apple Silicon Metal 驱动互操作)
    pub fn toMlx(self: DeviceType) mlx.DeviceType {
        return switch (self) {
            .cpu => .cpu,
            .gpu, .cuda => .gpu,
        };
    }

    /// 从 MLX DeviceType 转换
    pub fn fromMlx(dev: mlx.DeviceType) DeviceType {
        return switch (dev) {
            .cpu => .cpu,
            .gpu => .gpu,
        };
    }
};

/// 计算设备别名
pub const Device = DeviceType;

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

    pub fn copyToTensor(self: DeviceArray, t: anytype) !void {
        switch (self) {
            .mlx => |arr| try arr.copyToTensor(t),
            .cuda => |arr| try arr.copyToTensor(t),
        }
    }

    pub fn eval(self: DeviceArray) void {
        switch (self) {
            .mlx => |arr| arr.eval(),
            .cuda => {},
        }
    }

    pub fn reshape(self: DeviceArray, new_dims: []const usize, stream_opt: anytype) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = try arr.reshape(new_dims, stream_opt) },
            .cuda => |arr| .{ .cuda = try arr.reshape(new_dims) },
        };
    }

    pub fn transposeAxes(self: DeviceArray, axes: []const usize, stream_opt: anytype) !DeviceArray {
        return switch (self) {
            .mlx => |arr| .{ .mlx = try arr.transposeAxes(axes, stream_opt) },
            .cuda => error.OperationNotSupported,
        };
    }

    pub fn sliceAxis(self: DeviceArray, axis: usize, start: usize, end: usize, stream_opt: anytype) !DeviceArray {
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

/// 探测指定计算设备在当前平台是否可用
pub fn isAvailable(dev: DeviceType) bool {
    return switch (dev) {
        .cpu => true,
        .cuda => cuda.isAvailable(),
        .gpu => cuda.isAvailable() or mlx.has_mlx,
    };
}

/// 获取指定设备类型实际分派的底层硬件后端类型
pub fn activeBackend(dev: DeviceType) BackendType {
    return switch (dev) {
        .cpu => .cpu,
        .cuda => if (cuda.isAvailable()) .cuda else .cpu,
        .gpu => if (cuda.isAvailable()) .cuda else if (mlx.has_mlx) .mlx else .cpu,
    };
}

/// 获取当前系统的默认首选计算设备 (若图形处理器 GPU 可用优先使用 GPU，否则回退到中央处理器 CPU)
pub fn defaultDevice() DeviceType {
    if (isAvailable(.gpu)) return .gpu;
    return .cpu;
}

test "device: detection and conversions" {
    const dev = defaultDevice();
    if (cuda.isAvailable()) {
        try std.testing.expectEqual(Device.gpu, dev);
        try std.testing.expect(isAvailable(.gpu));
        try std.testing.expect(isAvailable(.cuda));
        try std.testing.expect(dev.isGpu());
        try std.testing.expectEqual(BackendType.cuda, activeBackend(.gpu));
        try std.testing.expectEqual(BackendType.cuda, activeBackend(.cuda));
    } else if (mlx.has_mlx) {
        try std.testing.expectEqual(Device.gpu, dev);
        try std.testing.expect(isAvailable(.gpu));
        try std.testing.expect(dev.isGpu());
        try std.testing.expectEqual(mlx.DeviceType.gpu, dev.toMlx());
        try std.testing.expectEqual(BackendType.mlx, activeBackend(.gpu));
    } else {
        try std.testing.expectEqual(Device.cpu, dev);
        try std.testing.expect(isAvailable(.cpu));
        try std.testing.expect(dev.isCpu());
        try std.testing.expectEqual(BackendType.cpu, activeBackend(.gpu));
    }
}
