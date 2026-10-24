//! Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速后端
//!
//! 基于 Apple 官方 `mlx-c` C API，将矩阵乘法、注意力机制、归一化与旋转位置编码卸载至
//! Apple Silicon 统一内存的图形处理器 (Graphics Processing Unit, GPU) 执行。
//! 依赖工程化 vendored 的 `deps/mlx-c/` 子项目，构建期零外部 Homebrew 依赖。

const std = @import("std");
const builtin = @import("builtin");
const activation = @import("activation.zig");
const GeluApproximation = activation.GeluApproximation;

/// 当前构建目标是否支持 Apple 机器学习拓展 (Machine Learning Explore, MLX)
pub const has_mlx: bool = (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64);

pub const c = if (has_mlx) @import("mlx_c.zig") else struct {
    pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_closure = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
};

pub const MlxError = error{
    MlxNotSupported,
    MlxInitializationFailed,
    MlxOperationFailed,
    InvalidShape,
    OutOfMemory,
    ShapeMismatch,
    NonContiguousTensor,
    DTypeMismatch,
};

/// 硬件计算设备类型：图形处理器 (Graphics Processing Unit, GPU) 与中央处理器 (Central Processing Unit, CPU)
pub const DeviceType = enum(c_uint) {
    cpu = 0,
    gpu = 1,
};

/// 数据类型枚举
pub const DType = enum(c_uint) {
    bool = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    bfloat16 = 12,
    complex64 = 13,
};

/// 命令执行流 (Stream)
pub const Stream = struct {
    handle: if (has_mlx) c.mlx_stream else ?*anyopaque,

    /// 获取默认图形处理器 (Graphics Processing Unit, GPU) 命令流
    pub fn defaultGpu() !Stream {
        if (!has_mlx) return error.MlxNotSupported;
        const s = c.mlx_default_gpu_stream_new();
        if (s.ctx == null) return error.MlxInitializationFailed;
        return .{ .handle = s };
    }

    /// 获取默认中央处理器 (Central Processing Unit, CPU) 命令流
    pub fn defaultCpu() !Stream {
        if (!has_mlx) return error.MlxNotSupported;
        const s = c.mlx_default_cpu_stream_new();
        if (s.ctx == null) return error.MlxInitializationFailed;
        return .{ .handle = s };
    }

    /// 释放命令流资源
    pub fn deinit(self: *Stream) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_stream_free(self.handle);
            self.handle.ctx = null;
        }
    }

    /// 同步等待流中的 GPU 任务执行完毕
    pub fn synchronize(self: Stream) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_synchronize(self.handle);
        }
    }
};

/// 未显式传入命令流时全部 MLX 算子共用的默认 GPU 命令流句柄 (`mlx_stream.ctx`)：
/// 首次使用时惰性创建并缓存，之后所有算子与线程复用同一句柄，不再为每次算子调用各自创建并释放一个默认流句柄
var default_gpu_stream_ctx: std.atomic.Value(?*anyopaque) = .init(null);

/// 获取缓存的默认 GPU 命令流句柄（首次调用时创建；多线程并发首次调用时只保留一个句柄，其余立即释放）
fn defaultGpuStreamHandle() MlxError!c.mlx_stream {
    if (!has_mlx) return error.MlxNotSupported;
    if (default_gpu_stream_ctx.load(.acquire)) |ctx| return .{ .ctx = ctx };
    const created = c.mlx_default_gpu_stream_new();
    const created_ctx = created.ctx orelse return error.MlxInitializationFailed;
    if (default_gpu_stream_ctx.cmpxchgStrong(null, created_ctx, .acq_rel, .acquire)) |existing| {
        _ = c.mlx_stream_free(created);
        return .{ .ctx = existing };
    }
    return created;
}

/// 解析算子使用的命令流：显式传入时直接使用（所有权归调用方），否则使用缓存的默认 GPU 命令流
fn resolveStream(stream: ?Stream) MlxError!c.mlx_stream {
    if (!has_mlx) return error.MlxNotSupported;
    if (stream) |st| return st.handle;
    return defaultGpuStreamHandle();
}

/// 释放缓存的默认 GPU 命令流句柄，之后未指定命令流的算子会重新惰性创建。
/// 只能在没有任何 MLX 算子仍在使用默认流时调用（例如进程退出前或测试结束时）。
pub fn releaseDefaultStream() void {
    if (!has_mlx) return;
    if (default_gpu_stream_ctx.swap(null, .acq_rel)) |ctx| {
        _ = c.mlx_stream_free(.{ .ctx = ctx });
    }
}

/// 机器学习拓展计算图编译模式 (Compilation Mode)
pub const CompileMode = enum(c_uint) {
    disabled = 0,
    no_simplify = 1,
    no_fuse = 2,
    enabled = 3,
};

/// 启用机器学习拓展 (Machine Learning Explore, MLX) 计算图静态编译与算子融合
pub fn enableCompile() void {
    if (has_mlx) {
        _ = c.mlx_enable_compile();
    }
}

/// 禁用机器学习拓展 (Machine Learning Explore, MLX) 计算图静态编译
pub fn disableCompile() void {
    if (has_mlx) {
        _ = c.mlx_disable_compile();
    }
}

/// 设置机器学习拓展 (Machine Learning Explore, MLX) 计算图编译模式
pub fn setCompileMode(mode: CompileMode) void {
    if (has_mlx) {
        _ = c.mlx_set_compile_mode(@backingInt(mode));
    }
}

/// 清理 Apple 机器学习拓展 (Machine Learning Explore, MLX) 缓存池中的未激活显存
pub fn clearCache() void {
    if (has_mlx) {
        _ = c.mlx_clear_cache();
    }
}

/// 设置 Apple 机器学习拓展 (Machine Learning Explore, MLX) 固定物理内存 (Wired Memory) 锁定上限（字节）
pub fn setWiredLimit(limit: usize) usize {
    if (!has_mlx) return 0;
    var prev: usize = 0;
    _ = c.mlx_set_wired_limit(&prev, limit);
    return prev;
}

/// 自动查询当前默认图形处理器 (Graphics Processing Unit, GPU) 的推荐最大工作集大小 (`max_recommended_working_set_size`)，
/// 并将其设置为统一内存固定物理页锁定上限 (Wired Memory Limit)，防止推理过程中页面被操作系统换出
pub fn configureOptimalWiredMemory() usize {
    if (!has_mlx) return 0;
    var dev = c.mlx_device_new();
    defer _ = c.mlx_device_free(dev);
    if (c.mlx_get_default_device(&dev) != 0) return 0;

    var info = c.mlx_device_info_new();
    defer _ = c.mlx_device_info_free(info);
    if (c.mlx_device_info_get(&info, dev) != 0) return 0;

    var max_working_set: usize = 0;
    if (c.mlx_device_info_get_size(&max_working_set, info, "max_recommended_working_set_size") == 0 and max_working_set > 0) {
        return setWiredLimit(max_working_set);
    }
    return 0;
}

/// MLX 数组向量容器 (VectorArray)
pub const VectorArray = struct {
    handle: if (has_mlx) c.mlx_vector_array else ?*anyopaque,

    pub fn init() VectorArray {
        if (!has_mlx) return .{ .handle = null };
        return .{ .handle = c.mlx_vector_array_new() };
    }

    /// 由任意数量的数组构建向量容器（逐个追加句柄，不截断、不在 Zig 侧分配内存）；构建失败时返回空句柄，后续调用将报错
    pub fn fromArrays(arrays: []const Array) VectorArray {
        if (!has_mlx) return .{ .handle = null };
        return .{ .handle = newVectorFromArrays(arrays) catch .{ .ctx = null } };
    }

    pub fn deinit(self: *VectorArray) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_vector_array_free(self.handle);
            self.handle.ctx = null;
        }
    }

    pub fn size(self: VectorArray) usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        return c.mlx_vector_array_size(self.handle);
    }

    pub fn get(self: VectorArray, idx: usize) ?Array {
        if (!has_mlx or self.handle.ctx == null) return null;
        var res = Array.init();
        if (c.mlx_vector_array_get(&res.handle, self.handle, idx) != 0 or res.handle.ctx == null) {
            return null;
        }
        return res;
    }
};

/// 逐个追加数组句柄构建原生 `mlx_vector_array`（调用方负责以 `mlx_vector_array_free` 释放）
fn newVectorFromArrays(arrays: []const Array) MlxError!c.mlx_vector_array {
    const vec = c.mlx_vector_array_new();
    if (vec.ctx == null) return error.MlxOperationFailed;
    errdefer _ = c.mlx_vector_array_free(vec);
    for (arrays) |arr| {
        if (c.mlx_vector_array_append_value(vec, arr.handle) != 0) return error.MlxOperationFailed;
    }
    return vec;
}

/// MLX 计算图静态编译闭包 (Closure)
pub const Closure = struct {
    handle: if (has_mlx) c.mlx_closure else ?*anyopaque,

    pub fn deinit(self: *Closure) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_closure_free(self.handle);
            self.handle.ctx = null;
        }
    }

    /// 应用静态编译闭包并执行硬件加速前向推理 (Execute Compiled Closure)
    pub fn apply(self: Closure, inputs: VectorArray) !VectorArray {
        if (!has_mlx or self.handle.ctx == null) return error.MlxNotSupported;
        var res = VectorArray.init();
        errdefer res.deinit();
        if (c.mlx_closure_apply(&res.handle, self.handle, inputs.handle) != 0 or res.handle.ctx == null) {
            return error.MlxOperationFailed;
        }
        return res;
    }

    /// 零分配快速执行单输入双输出编译闭包：`{ out0, out1 } = cls(a)`
    pub fn apply1To2(self: Closure, a: Array) !struct { Array, Array } {
        if (!has_mlx or self.handle.ctx == null) return error.MlxNotSupported;
        const raw_in = [_]c.mlx_array{a.handle};
        const in_vec = c.mlx_vector_array_new_data(&raw_in, 1);
        defer _ = c.mlx_vector_array_free(in_vec);

        var out_vec = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(out_vec);
        if (c.mlx_closure_apply(&out_vec, self.handle, in_vec) != 0 or out_vec.ctx == null) {
            return error.MlxOperationFailed;
        }
        var out0 = Array.init();
        var out1 = Array.init();
        if (c.mlx_vector_array_get(&out0.handle, out_vec, 0) != 0 or out0.handle.ctx == null) {
            return error.MlxOperationFailed;
        }
        if (c.mlx_vector_array_get(&out1.handle, out_vec, 1) != 0 or out1.handle.ctx == null) {
            out0.deinit();
            return error.MlxOperationFailed;
        }
        return .{ out0, out1 };
    }

    /// 零分配快速执行三输入单输出编译闭包：`res = cls(a, b, d)`
    pub fn apply3(self: Closure, a: Array, b: Array, d: Array) !Array {
        if (!has_mlx or self.handle.ctx == null) return error.MlxNotSupported;
        const raw_in = [_]c.mlx_array{ a.handle, b.handle, d.handle };
        const in_vec = c.mlx_vector_array_new_data(&raw_in, 3);
        defer _ = c.mlx_vector_array_free(in_vec);

        var out_vec = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(out_vec);
        if (c.mlx_closure_apply(&out_vec, self.handle, in_vec) != 0 or out_vec.ctx == null) {
            return error.MlxOperationFailed;
        }
        var res = Array.init();
        if (c.mlx_vector_array_get(&res.handle, out_vec, 0) != 0 or res.handle.ctx == null) {
            return error.MlxOperationFailed;
        }
        return res;
    }
};

/// 静态编译函数闭包 (Compile Closure)，通过 Metal 即时编译 (Just-In-Time Compilation, JIT) 融合计算图内核
pub fn compile(
    fun: *const fn ([*c]c.mlx_vector_array, c.mlx_vector_array, ?*anyopaque) callconv(.c) c_int,
    payload: ?*anyopaque,
    shapeless: bool,
) !Closure {
    if (!has_mlx) return error.MlxNotSupported;
    const raw_cls = c.mlx_closure_new_func_payload(fun, payload, null);
    defer _ = c.mlx_closure_free(raw_cls);

    var compiled_cls = c.mlx_closure_new();
    const rc = c.mlx_compile(&compiled_cls, raw_cls, shapeless);
    if (rc != 0 or compiled_cls.ctx == null) {
        if (compiled_cls.ctx != null) _ = c.mlx_closure_free(compiled_cls);
        return error.MlxOperationFailed;
    }
    return .{ .handle = compiled_cls };
}

/// MLX 多维数组封装 (Array)
pub const Array = struct {
    handle: if (has_mlx) c.mlx_array else ?*anyopaque,

    /// 构造一个空数组
    pub fn init() Array {
        if (has_mlx) {
            return .{ .handle = c.mlx_array_new() };
        } else {
            return .{ .handle = null };
        }
    }

    /// 从切片数据复制创建单精度浮点张量
    pub fn fromSlice(slice: []const f32, shape_dims: []const usize) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len == 0 or shape_dims.len > 8) return error.InvalidShape;

        var c_shape: [8]c_int = undefined;
        var total_numel: usize = 1;
        for (shape_dims, 0..) |dim, i| {
            c_shape[i] = @intCast(dim);
            total_numel *= dim;
        }
        if (total_numel != slice.len) return error.InvalidShape;

        const arr = c.mlx_array_new_data(
            slice.ptr,
            &c_shape,
            @intCast(shape_dims.len),
            c.MLX_FLOAT32,
        );
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 从切片数据复制创建脑浮点 16 (Brain Floating Point 16, BF16) MLX 数组
    pub fn fromSliceBf16(slice: []const u16, shape_dims: []const usize) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len == 0 or shape_dims.len > 8) return error.InvalidShape;

        var c_shape: [8]c_int = undefined;
        var total_numel: usize = 1;
        for (shape_dims, 0..) |dim, i| {
            c_shape[i] = @intCast(dim);
            total_numel *= dim;
        }
        if (total_numel != slice.len) return error.InvalidShape;

        const arr = c.mlx_array_new_data(
            slice.ptr,
            &c_shape,
            @intCast(shape_dims.len),
            c.MLX_BFLOAT16,
        );
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 从 32 位有符号整型切片数据创建 MLX 数组 (Int32 Array Creation)
    pub fn fromSliceI32(slice: []const i32, shape_dims: []const usize) !Array {
        return fromRawData(slice.ptr, shape_dims, .int32);
    }

    /// 从 32 位无符号整型切片数据创建 MLX 数组 (UInt32 Array Creation)
    pub fn fromSliceU32(slice: []const u32, shape_dims: []const usize) !Array {
        return fromRawData(slice.ptr, shape_dims, .uint32);
    }

    /// 从原始数据指针与形状维度创建 MLX 数组 (Raw Pointer Array Creation)
    pub fn fromRawData(
        data: *const anyopaque,
        shape_dims: []const usize,
        dt: DType,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len == 0 or shape_dims.len > 8) return error.InvalidShape;

        var c_shape: [8]c_int = undefined;
        for (shape_dims, 0..) |dim, i| {
            c_shape[i] = @intCast(dim);
        }

        const arr = c.mlx_array_new_data(
            data,
            &c_shape,
            @intCast(shape_dims.len),
            @backingInt(dt),
        );
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 创建全零数组 (Zeros)
    pub fn zeros(shape_dims: []const usize, dt: DType, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len == 0 or shape_dims.len > 8) return error.InvalidShape;

        var c_shape: [8]c_int = undefined;
        for (shape_dims, 0..) |dim, i| {
            c_shape[i] = @intCast(dim);
        }

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_zeros(
            &res.handle,
            &c_shape,
            shape_dims.len,
            @backingInt(dt),
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 从单精度浮点标量创建 MLX 标量数组
    pub fn fromScalarF32(val: f32) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const arr = c.mlx_array_new_float32(val);
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 将单精度浮点数按 IEEE 754 最近偶数舍入转换为脑浮点 16 (Brain Floating Point 16, BF16) 的 16 位位宽表示
    pub fn f32ToBf16Bits(val: f32) u16 {
        const bits: u32 = @bitCast(val);
        if ((bits & 0x7f800000) == 0x7f800000 and (bits & 0x007fffff) != 0) {
            return @truncate((bits >> 16) | 0x0040);
        }
        const lsb = (bits >> 16) & 1;
        const rounding_bias = @as(u32, 0x00007fff) + lsb;
        return @truncate((bits + rounding_bias) >> 16);
    }

    /// 从单精度浮点标量创建脑浮点 16 (Brain Floating Point 16, BF16) MLX 0 维标量数组
    pub fn fromScalarBf16(val: f32) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const bf16_bits = f32ToBf16Bits(val);
        var dummy_shape: [1]c_int = .{0};
        const arr = c.mlx_array_new_data(&bf16_bits, &dummy_shape, 0, c.MLX_BFLOAT16);
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 从单精度浮点标量创建半精度浮点 16 (Float16, FP16) MLX 0 维标量数组
    pub fn fromScalarF16(val: f32) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const f16_val: f16 = @floatCast(val);
        var dummy_shape: [1]c_int = .{0};
        const arr = c.mlx_array_new_data(&f16_val, &dummy_shape, 0, c.MLX_FLOAT16);
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 从单精度浮点标量创建与指定数据类型对齐的 MLX 0 维标量数组（避免 bfloat16/float16 张量被隐式提升为 float32）
    pub fn fromScalarTyped(val: f32, target_dtype: DType) !Array {
        return switch (target_dtype) {
            .bfloat16 => fromScalarBf16(val),
            .float16 => fromScalarF16(val),
            else => fromScalarF32(val),
        };
    }

    /// 从 32 位整型标量创建 MLX 标量数组
    pub fn fromScalarI32(val: i32) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const arr = c.mlx_array_new_int(val);
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    fn noopDtor(_: ?*anyopaque) callconv(.c) void {}

    /// 从外部已分配内存零拷贝 (Zero-Copy) 创建单精度浮点 MLX 数组 (统一内存架构 Unified Memory Architecture, UMA)
    pub fn fromSliceZeroCopy(slice: []f32, shape_dims: []const usize) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len == 0 or shape_dims.len > 8) return error.InvalidShape;

        var c_shape: [8]c_int = undefined;
        var total_numel: usize = 1;
        for (shape_dims, 0..) |dim, i| {
            c_shape[i] = @intCast(dim);
            total_numel *= dim;
        }
        if (total_numel != slice.len) return error.InvalidShape;

        const arr = c.mlx_array_new_data_managed(
            slice.ptr,
            &c_shape,
            @intCast(shape_dims.len),
            c.MLX_FLOAT32,
            &noopDtor,
        );
        if (arr.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = arr };
    }

    /// 获取数组元素数据类型 (Element Data Type)
    pub fn dtype(self: Array) DType {
        if (!has_mlx or self.handle.ctx == null) return .float32;
        return @fromBackingInt(@intCast(c.mlx_array_dtype(self.handle)));
    }

    /// 获取数组元素总数
    pub fn size(self: Array) usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        return c.mlx_array_size(self.handle);
    }

    /// 获取数组形状切片到输出缓冲区中，返回实际维度数量
    pub fn shape(self: Array, out_dims: []usize) !usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        const n = self.ndim();
        if (out_dims.len < n) return error.InvalidShape;
        const c_shape = c.mlx_array_shape(self.handle);
        for (0..n) |i| {
            out_dims[i] = @intCast(c_shape[i]);
        }
        return n;
    }

    /// 检查张量在显存中是否为标准严格行主序连续存储 (C-contiguous / Row-Major)
    pub fn isContiguousRowMajor(self: Array) bool {
        if (!has_mlx or self.handle.ctx == null) return false;
        const n = self.ndim();
        if (n == 0) return true;
        const c_shape = c.mlx_array_shape(self.handle);
        const c_strides = c.mlx_array_strides(self.handle);
        if (c_strides[n - 1] != 1) return false;
        var expected_stride: usize = 1;
        var i: usize = n - 1;
        while (i > 0) {
            expected_stride *= @intCast(c_shape[i]);
            i -= 1;
            if (c_strides[i] != expected_stride) return false;
        }
        return true;
    }

    /// 将数组重整为标准行主序连续存储 (Row-Major Contiguous)
    pub fn contiguous(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);
        const rc = c.mlx_contiguous(&res.handle, self.handle, false, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 将 GPU 计算完毕后的浮点数据安全复制到目标切片中（自动感知并处理非连续、转置与广播布局；非单精度数组先转换为单精度）
    pub fn copyToSlice(self: Array, dest: []f32) !void {
        if (!has_mlx) return error.MlxNotSupported;
        if (self.dtype() != .float32) {
            var converted = try self.astype(.float32, null);
            defer converted.deinit();
            return converted.copyToSlice(dest);
        }
        const numel = self.size();
        if (dest.len != numel) return error.ShapeMismatch;
        if (numel == 0) return;

        self.eval();
        // 非连续视图的元素按步长分布在底层缓冲区中，跨度可能大于 numel（切片视图）也可能小于 numel（广播视图），
        // 因此以多元素指针按步长寻址，而不是截取长度为 numel 的切片
        const src: [*]const f32 = c.mlx_array_data_float32(self.handle) orelse return error.MlxOperationFailed;

        if (self.isContiguousRowMajor()) {
            @memcpy(dest, src[0..numel]);
            return;
        }

        const n = self.ndim();
        if (n > 8) return error.InvalidShape;
        const c_shape = c.mlx_array_shape(self.handle);
        const c_strides = c.mlx_array_strides(self.handle);

        // 最内层维度连续时按行整块复制（覆盖注意力算子常见的 (Batch, Head, SeqLen, HeadDim) 转置视图）
        const inner: usize = @intCast(c_shape[n - 1]);
        const inner_contiguous = c_strides[n - 1] == 1;
        const rows = numel / inner;
        var coord: [8]usize = @splat(0);
        for (0..rows) |row| {
            var in_off: usize = 0;
            for (0..n - 1) |d| in_off += coord[d] * c_strides[d];
            const out_row = dest[row * inner ..][0..inner];
            if (inner_contiguous) {
                @memcpy(out_row, src[in_off..][0..inner]);
            } else {
                for (out_row, 0..) |*o, e| o.* = src[in_off + e * c_strides[n - 1]];
            }

            // 按行主序递增除最内层外的多维坐标
            var dim_idx = n - 1;
            while (dim_idx > 0) {
                dim_idx -= 1;
                coord[dim_idx] += 1;
                if (coord[dim_idx] < @as(usize, @intCast(c_shape[dim_idx]))) break;
                coord[dim_idx] = 0;
            }
        }
    }

    /// 从任意张量对象复制创建 MLX 数组 (鸭子类型，保持 backend 为最底层包而不产生反向导入)
    pub fn fromTensor(t: anytype) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (!t.isContiguous()) return error.NonContiguousTensor;
        return fromSlice(t.data[0..t.shape.numel()], t.shape.dims[0..t.shape.len]);
    }

    /// 从任意张量对象零拷贝创建 MLX 数组 (统一内存架构 Unified Memory Architecture, UMA)
    pub fn fromTensorZeroCopy(t: anytype) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (!t.isContiguous()) return error.NonContiguousTensor;
        return fromSliceZeroCopy(t.data[0..t.shape.numel()], t.shape.dims[0..t.shape.len]);
    }

    /// 将 MLX 数组中的计算结果写回任意张量对象
    pub fn copyToTensor(self: Array, t: anytype) !void {
        if (!has_mlx) return error.MlxNotSupported;
        if (!t.isContiguous()) return error.NonContiguousTensor;
        return self.copyToSlice(t.data[0..t.shape.numel()]);
    }

    /// 释放数组显存与内部引用
    pub fn deinit(self: *Array) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_array_free(self.handle);
            self.handle.ctx = null;
        }
    }

    /// 强制触发惰性求值 (Lazy Evaluation)，将计算下发至 GPU 完成
    pub fn eval(self: Array) void {
        if (has_mlx and self.handle.ctx != null) {
            _ = c.mlx_array_eval(self.handle);
        }
    }

    /// 异步触发惰性求值 (Asynchronous Evaluation)，将计算入队至 GPU 流水线但不阻塞当前线程
    pub fn asyncEval(self: Array) void {
        if (has_mlx and self.handle.ctx != null) {
            const vec = c.mlx_vector_array_new_data(&self.handle, 1);
            defer _ = c.mlx_vector_array_free(vec);
            _ = c.mlx_async_eval(vec);
        }
    }

    /// 校验数组句柄有效且元素数据类型为 expected，随后触发求值使底层缓冲区就绪
    fn prepareTypedRead(self: Array, expected: DType) MlxError!void {
        if (!has_mlx) return error.MlxNotSupported;
        if (self.handle.ctx == null) return error.MlxOperationFailed;
        if (self.dtype() != expected) return error.DTypeMismatch;
        self.eval();
    }

    /// 获取求值就绪后的单精度浮点数据切片（借用数组内部缓冲区，生命周期不超过数组本身）。
    /// 数组元素类型不是 float32 时返回 `error.DTypeMismatch`（需要转换时先调用 `astype` 或改用 `copyToSlice`）。
    pub fn dataFloat32(self: Array) MlxError![]const f32 {
        if (!has_mlx) return error.MlxNotSupported;
        try self.prepareTypedRead(.float32);
        const ptr: [*]const f32 = c.mlx_array_data_float32(self.handle) orelse return error.MlxOperationFailed;
        return ptr[0..self.size()];
    }

    /// 获取求值就绪后的 32 位无符号整型数据切片（借用数组内部缓冲区）；元素类型不是 uint32 时返回 `error.DTypeMismatch`
    pub fn dataUint32(self: Array) MlxError![]const u32 {
        if (!has_mlx) return error.MlxNotSupported;
        try self.prepareTypedRead(.uint32);
        const ptr: [*]const u32 = c.mlx_array_data_uint32(self.handle) orelse return error.MlxOperationFailed;
        return ptr[0..self.size()];
    }

    /// 提取单元素 uint32 张量的值（例如 `argmax` 的结果）；元素数不为 1 时返回 `error.InvalidShape`
    pub fn itemUint32(self: Array) MlxError!u32 {
        const d = try self.dataUint32();
        if (d.len != 1) return error.InvalidShape;
        return d[0];
    }

    /// 获取张量维度数目
    pub fn ndim(self: Array) usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        return c.mlx_array_ndim(self.handle);
    }

    /// 获取指定轴的维度大小 (Dimension Size Along Axis)
    pub fn dimSize(self: Array, axis: c_int) usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        const d = c.mlx_array_dim(self.handle, axis);
        return if (d > 0) @intCast(d) else 0;
    }

    /// 矩阵或多维数组转置 (Transpose)
    pub fn transpose(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_transpose(&res.handle, self.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 重构张量维度形状 (Reshape)
    pub fn reshape(self: Array, shape_dims: []const usize, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (shape_dims.len > 8) return error.InvalidShape;
        var shape_ints: [8]c_int = undefined;
        for (shape_dims, 0..) |dim, i| {
            shape_ints[i] = @intCast(dim);
        }

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_reshape(&res.handle, self.handle, &shape_ints, shape_dims.len, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 指定轴转置 (Transpose Axes)
    pub fn transposeAxes(self: Array, axes: []const usize, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (axes.len > 8) return error.InvalidShape;
        var axes_ints: [8]c_int = undefined;
        for (axes, 0..) |ax, i| {
            axes_ints[i] = @intCast(ax);
        }

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_transpose_axes(&res.handle, self.handle, &axes_ints, axes.len, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴拼接多个数组 (Concatenate Along Axis)
    pub fn concatenateAxis(arrays: []const Array, axis: c_int, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (arrays.len == 0) return error.InvalidShape;

        const vec = try newVectorFromArrays(arrays);
        defer _ = c.mlx_vector_array_free(vec);

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_concatenate_axis(&res.handle, vec, axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴切片 (Slice Axis)
    pub fn sliceAxis(self: Array, axis: usize, start: usize, stop: usize, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var shape_buf: [8]usize = undefined;
        const nd = try self.shape(&shape_buf);
        if (axis >= nd) return error.InvalidShape;

        var start_arr: [8]c_int = undefined;
        var stop_arr: [8]c_int = undefined;
        var strides_arr: [8]c_int = undefined;
        for (0..nd) |i| {
            if (i == axis) {
                start_arr[i] = @intCast(start);
                stop_arr[i] = @intCast(stop);
            } else {
                start_arr[i] = 0;
                stop_arr[i] = @intCast(shape_buf[i]);
            }
            strides_arr[i] = 1;
        }

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_slice(
            &res.handle,
            self.handle,
            &start_arr,
            nd,
            &stop_arr,
            nd,
            &strides_arr,
            nd,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 对二维张量沿列维度 (axis 1) 快速切片 (Fast 2D Column Slice)
    /// 预置 `[1, total_cols]` 形状，消除动态 ndim 与 shape 描述符解析开销
    pub fn slice2dAxis1(self: Array, start: usize, stop: usize, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const start_arr = [_]c_int{ 0, @intCast(start) };
        const stop_arr = [_]c_int{ 1, @intCast(stop) };
        const strides_arr = [_]c_int{ 1, 1 };

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_slice(
            &res.handle,
            self.handle,
            &start_arr,
            2,
            &stop_arr,
            2,
            &strides_arr,
            2,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴以子数组切片更新 (Slice Update Axis)
    pub fn sliceUpdateAxis(
        self: Array,
        update: Array,
        axis: usize,
        start: usize,
        stop: usize,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var shape_buf: [8]usize = undefined;
        const nd = try self.shape(&shape_buf);
        if (axis >= nd) return error.InvalidShape;

        var start_arr: [8]c_int = undefined;
        var stop_arr: [8]c_int = undefined;
        var strides_arr: [8]c_int = undefined;
        for (0..nd) |i| {
            if (i == axis) {
                start_arr[i] = @intCast(start);
                stop_arr[i] = @intCast(stop);
            } else {
                start_arr[i] = 0;
                stop_arr[i] = @intCast(shape_buf[i]);
            }
            strides_arr[i] = 1;
        }

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_slice_update(
            &res.handle,
            self.handle,
            update.handle,
            &start_arr,
            nd,
            &stop_arr,
            nd,
            &strides_arr,
            nd,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 对四维键值缓存张量 (Key-Value Cache, KVCache) 沿序列轴 (axis 2) 快速切片 (Fast 4D Sequence Slice)
    /// 预置 `[1, num_kv, max_seq_len, head_dim]` 四维形状，消除运行时 ndim 与 shape 描述符解析开销
    pub fn slice4dAxis2(
        self: Array,
        num_kv: usize,
        start: usize,
        stop: usize,
        head_dim: usize,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const start_arr = [_]c_int{ 0, 0, @intCast(start), 0 };
        const stop_arr = [_]c_int{ 1, @intCast(num_kv), @intCast(stop), @intCast(head_dim) };
        const strides_arr = [_]c_int{ 1, 1, 1, 1 };

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_slice(
            &res.handle,
            self.handle,
            &start_arr,
            4,
            &stop_arr,
            4,
            &strides_arr,
            4,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 对四维键值缓存张量 (Key-Value Cache, KVCache) 沿序列轴 (axis 2) 快速定点更新 (Fast 4D Sequence Slice Update)
    /// 在固定预分配显存中直接更新指定序列范围，消除动态重新分配显存与数据拷贝开销
    pub fn sliceUpdate4dAxis2(
        self: Array,
        update: Array,
        num_kv: usize,
        start: usize,
        stop: usize,
        head_dim: usize,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const start_arr = [_]c_int{ 0, 0, @intCast(start), 0 };
        const stop_arr = [_]c_int{ 1, @intCast(num_kv), @intCast(stop), @intCast(head_dim) };
        const strides_arr = [_]c_int{ 1, 1, 1, 1 };

        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_slice_update(
            &res.handle,
            self.handle,
            update.handle,
            &start_arr,
            4,
            &stop_arr,
            4,
            &strides_arr,
            4,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴按索引提取张量子集 (Take Along Axis / GPU Embedding Lookup)
    pub fn takeAxis(self: Array, indices: Array, axis: c_int, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_take_axis(&res.handle, self.handle, indices.handle, axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴计算最大值索引 (Argmax along Axis)
    pub fn argmax(self: Array, axis: i32, keepdims: bool, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_argmax_axis(&res.handle, self.handle, axis, keepdims, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴部分排序的下标 (Argpartition along Axis)：结果中第 kth 个位置恰为排序后的第 kth 小元素下标，
    /// 其前的元素均不大于它、其后的元素均不小于它 (各段内部无序)；下标类型为 uint32
    pub fn argpartitionAxis(self: Array, kth: usize, axis: i32, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_argpartition_axis(&res.handle, self.handle, @intCast(kth), axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴按逐元素下标取值 (Take Along Axis)：indices 与 self 维数相同，结果形状与 indices 一致
    pub fn takeAlongAxis(self: Array, indices: Array, axis: i32, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_take_along_axis(&res.handle, self.handle, indices.handle, axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴按逐元素下标写入 (Put Along Axis)：返回把 values 写入 self 中 indices 位置后的新数组 (self 不变)
    pub fn putAlongAxis(self: Array, indices: Array, values: Array, axis: i32, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_put_along_axis(&res.handle, self.handle, indices.handle, values.handle, axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴的对数求和指数 (Log-Sum-Exp along Axis)：log Σ exp(x)，数值稳定
    pub fn logsumexpAxis(self: Array, axis: i32, keepdims: bool, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_logsumexp_axis(&res.handle, self.handle, axis, keepdims, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素取负 (Negative)：`res = -self`
    pub fn negative(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_negative(&res.handle, self.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素严格大于比较 (Greater)：`res = a > b` (布尔数组，支持广播)
    pub fn greater(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_greater(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素条件选择 (Where)：condition 为真处取 x，否则取 y (支持广播)
    pub fn where(condition: Array, x: Array, y: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_where(&res.handle, condition.handle, x.handle, y.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 创建引用同一底层数组的新句柄 (共享数据，不复制)；新句柄须单独 `deinit`
    pub fn share(self: Array) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        if (c.mlx_array_set(&res.handle, self.handle) != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴均等切分 (Split into equal sections)
    pub fn split(self: Array, num_splits: i32, axis: i32, stream: ?Stream) !VectorArray {
        if (!has_mlx) return error.MlxNotSupported;
        var res = VectorArray.init();
        errdefer res.deinit();
        const s = try resolveStream(stream);

        const rc = c.mlx_split(&res.handle, self.handle, num_splits, axis, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 标量乘法：`res = self * val`（自动保持输入张量的数据精度，防止 bfloat16/float16 被隐式提升为 float32）
    pub fn mulScalar(self: Array, val: f32, stream: ?Stream) !Array {
        var s_arr = try fromScalarTyped(val, self.dtype());
        defer s_arr.deinit();
        return multiply(self, s_arr, stream);
    }

    /// 数据类型转换 (Data Type Conversion / Astype)
    pub fn astype(self: Array, dt: DType, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_astype(&res.handle, self.handle, @backingInt(dt), s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 矩阵乘法：`res = a @ b`
    pub fn matmul(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_matmul(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素加法：`res = a + b`
    pub fn add(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_add(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素减法：`res = a - b`
    pub fn subtract(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_subtract(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素乘法：`res = a * b`
    pub fn multiply(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_multiply(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 高斯误差函数 (Error Function, ERF)
    pub fn erf(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_erf(&res.handle, self.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 高斯误差线性单元激活函数 (Gaussian Error Linear Unit, GELU)（自动保持输入张量精度）
    /// - `.none`：GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    /// - `.tanh`：GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))（与 mlx-lm `nn.gelu_approx` 一致）
    pub fn gelu(self: Array, approximation: GeluApproximation, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const dt = self.dtype();

        // inner = erf(x / sqrt(2)) 或 tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))
        var inner = switch (approximation) {
            .none => blk: {
                var s_scale = try fromScalarTyped(activation.gelu_erf_scale, dt);
                defer s_scale.deinit();
                var x_scaled = try multiply(self, s_scale, stream);
                defer x_scaled.deinit();
                break :blk try x_scaled.erf(stream);
            },
            .tanh => blk: {
                var x_sq = try multiply(self, self, stream);
                defer x_sq.deinit();
                var s_cubic = try fromScalarTyped(activation.gelu_tanh_cubic_coeff, dt);
                defer s_cubic.deinit();
                var cubic_term = try multiply(x_sq, s_cubic, stream);
                defer cubic_term.deinit();
                var s_one_inner = try fromScalarTyped(1.0, dt);
                defer s_one_inner.deinit();
                var poly = try add(s_one_inner, cubic_term, stream);
                defer poly.deinit();
                // x + 0.044715 * x^3 = x * (1 + 0.044715 * x^2)
                var cubic_sum = try multiply(self, poly, stream);
                defer cubic_sum.deinit();
                var s_scale = try fromScalarTyped(activation.gelu_tanh_scale, dt);
                defer s_scale.deinit();
                var u = try multiply(cubic_sum, s_scale, stream);
                defer u.deinit();
                break :blk try u.tanh(stream);
            },
        };
        defer inner.deinit();

        var s_one = try fromScalarTyped(1.0, dt);
        defer s_one.deinit();
        var one_plus_inner = try add(s_one, inner, stream);
        defer one_plus_inner.deinit();

        var s_half = try fromScalarTyped(0.5, dt);
        defer s_half.deinit();
        var half_x = try multiply(self, s_half, stream);
        defer half_x.deinit();

        return multiply(half_x, one_plus_inner, stream);
    }

    /// 门控高斯误差线性单元融合激活乘法：`out = GELU(gate) * up`
    /// 若提供了预编译的静态闭包 (Closure，须由 `compileGeluMul` 以相同的 `approximation` 编译)，则直接派发单内核融合算子，
    /// 将多次 Metal 内核启动合并为 1 次寄存器级融合执行
    pub fn fusedGeluMul(gate: Array, up: Array, approximation: GeluApproximation, compiled: ?Closure, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (compiled) |cls| {
            var in_vec = VectorArray.fromArrays(&.{ gate, up });
            defer in_vec.deinit();
            var out_vec = try cls.apply(in_vec);
            defer out_vec.deinit();
            if (out_vec.get(0)) |res| return res;
        }
        var act = try gate.gelu(approximation, stream);
        defer act.deinit();
        return multiply(act, up, stream);
    }

    /// 静态编译门控高斯误差线性单元算子融合闭包 (`out = GELU(gate) * up`，支持任意形状 `shapeless = true`)，
    /// 闭包按 `approximation` 在编译期特化
    pub fn compileGeluMul(approximation: GeluApproximation) !Closure {
        if (!has_mlx) return error.MlxNotSupported;
        return switch (approximation) {
            inline else => |approx| compile(GeluMulKernel(approx).compute, null, true),
        };
    }

    fn GeluMulKernel(comptime approximation: GeluApproximation) type {
        return struct {
            fn compute(vres: [*c]c.mlx_vector_array, in: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                if (!has_mlx) return 1;
                _ = payload;
                var gate_h: c.mlx_array = .{ .ctx = null };
                var up_h: c.mlx_array = .{ .ctx = null };
                if (c.mlx_vector_array_get(&gate_h, in, 0) != 0 or gate_h.ctx == null) return 1;
                if (c.mlx_vector_array_get(&up_h, in, 1) != 0 or up_h.ctx == null) {
                    _ = c.mlx_array_free(gate_h);
                    return 1;
                }
                // mlx_vector_array_get 为取出的元素分配了新的数组句柄，用完后需释放
                var gate = Array{ .handle = gate_h };
                defer gate.deinit();
                var up = Array{ .handle = up_h };
                defer up.deinit();

                var act = gate.gelu(approximation, null) catch return 1;
                defer act.deinit();
                var out = multiply(act, up, null) catch return 1;
                defer out.deinit();

                const res_vec = c.mlx_vector_array_new_data(&[_]c.mlx_array{out.handle}, 1);
                _ = c.mlx_vector_array_set(vres, res_vec);
                _ = c.mlx_vector_array_free(res_vec);
                return 0;
            }
        };
    }

    /// 残差相加与标量缩放融合算子：`out = (a + b) * scale`
    /// 若提供了预编译的静态闭包 (Closure)，则直接派发单内核融合算子，将 2 次逐元素内核启动与中间显存读写合并为 1 次寄存器级执行
    pub fn fusedAddMul(a: Array, b: Array, scale: Array, compiled: ?Closure, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (compiled) |cls| {
            var in_vec = VectorArray.fromArrays(&.{ a, b, scale });
            defer in_vec.deinit();
            var out_vec = try cls.apply(in_vec);
            defer out_vec.deinit();
            if (out_vec.get(0)) |res| return res;
        }
        var sum = try add(a, b, stream);
        defer sum.deinit();
        return multiply(sum, scale, stream);
    }

    /// 静态编译残差相加与标量缩放融合闭包 (`out = (a + b) * scale`，支持任意形状 `shapeless = true`)
    pub fn compileAddMul() !Closure {
        if (!has_mlx) return error.MlxNotSupported;
        const Helper = struct {
            fn compute(vres: [*c]c.mlx_vector_array, in: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                if (!has_mlx) return 1;
                _ = payload;
                // mlx_vector_array_get 为取出的元素分配了新的数组句柄，用完后需释放（deinit 对空句柄无操作）
                var a_arr = Array.init();
                defer a_arr.deinit();
                var b_arr = Array.init();
                defer b_arr.deinit();
                var s_arr = Array.init();
                defer s_arr.deinit();
                if (c.mlx_vector_array_get(&a_arr.handle, in, 0) != 0 or a_arr.handle.ctx == null) return 1;
                if (c.mlx_vector_array_get(&b_arr.handle, in, 1) != 0 or b_arr.handle.ctx == null) return 1;
                if (c.mlx_vector_array_get(&s_arr.handle, in, 2) != 0 or s_arr.handle.ctx == null) return 1;

                var sum = add(a_arr, b_arr, null) catch return 1;
                defer sum.deinit();
                var out = multiply(sum, s_arr, null) catch return 1;
                defer out.deinit();

                const res_vec = c.mlx_vector_array_new_data(&[_]c.mlx_array{out.handle}, 1);
                _ = c.mlx_vector_array_set(vres, res_vec);
                _ = c.mlx_vector_array_free(res_vec);
                return 0;
            }
        };
        return compile(Helper.compute, null, true);
    }

    /// 门控高斯误差线性单元激活 (Gated Gaussian Error Linear Unit, GeGLU)
    /// 沿最后一轴将门控与上升投影切分为两半，计算 `out = GELU(gate) * up` (GELU 按 `approximation` 选择形式)
    pub fn geglu(self: Array, intermediate_size: usize, approximation: GeluApproximation, stream: ?Stream) !Array {
        const last_axis = self.ndim() - 1;
        var gate = try self.sliceAxis(last_axis, 0, intermediate_size, stream);
        defer gate.deinit();
        var up = try self.sliceAxis(last_axis, intermediate_size, intermediate_size * 2, stream);
        defer up.deinit();
        var act = try gate.gelu(approximation, stream);
        defer act.deinit();
        return multiply(act, up, stream);
    }

    /// 批量评估多个 MLX 数组 (多算子并行批处理调度，消除重复的主机-设备同步开销)
    /// 构建向量容器失败时逐个求值，保证返回后所有数组均已就绪
    pub fn evalMultiple(arrays: []const Array) void {
        if (!has_mlx or arrays.len == 0) return;
        const vec = newVectorFromArrays(arrays) catch {
            for (arrays) |arr| arr.eval();
            return;
        };
        defer _ = c.mlx_vector_array_free(vec);
        _ = c.mlx_eval(vec);
    }

    /// Softmax 归一化算子
    pub fn softmax(a: Array, axis: c_int, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const rc = c.mlx_softmax_axis(&res.handle, a.handle, axis, false, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 快速均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    pub fn fastRmsNorm(x: Array, weight: ?Array, eps: f32, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const w_handle = if (weight) |w| w.handle else c.mlx_array{ .ctx = null };
        const rc = c.mlx_fast_rms_norm(&res.handle, x.handle, w_handle, eps, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 快速旋转位置编码 (Rotary Position Embedding, RoPE)
    pub fn fastRoPE(
        x: Array,
        dims: usize,
        traditional: bool,
        base: ?f32,
        scale: f32,
        offset: usize,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const opt_base = if (base) |b|
            c.mlx_optional_float{ .value = b, .has_value = true }
        else
            c.mlx_optional_float{ .value = 10000.0, .has_value = false };

        const null_arr = c.mlx_array{ .ctx = null };
        const rc = c.mlx_fast_rope(
            &res.handle,
            x.handle,
            @intCast(dims),
            traditional,
            opt_base,
            scale,
            @intCast(offset),
            null_arr,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 以自定义旋转周期表执行快速旋转位置编码 (Rotary Position Embedding with Custom Frequencies)：
    /// freqs 为 [dims / 2] 的单精度周期表，第 i 对分量的旋转角为 (offset + t) / freqs[i]（周期为 +inf 时该对保持恒等）
    pub fn fastRoPEWithFreqs(
        x: Array,
        dims: usize,
        traditional: bool,
        scale: f32,
        offset: usize,
        freqs: Array,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const no_base = c.mlx_optional_float{ .value = 0.0, .has_value = false };
        const rc = c.mlx_fast_rope(
            &res.handle,
            x.handle,
            @intCast(dims),
            traditional,
            no_base,
            scale,
            @intCast(offset),
            freqs.handle,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 快速缩放点积注意力 (Scaled Dot-Product Attention, SDPA) / FlashAttention
    pub fn fastScaledDotProductAttention(
        queries: Array,
        keys: Array,
        values: Array,
        scale: f32,
        is_causal: bool,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = try resolveStream(stream);

        const mask_mode: [*c]const u8 = if (is_causal) "causal" else "";
        const empty_arr = c.mlx_array{ .ctx = null };
        const rc = c.mlx_fast_scaled_dot_product_attention(
            &res.handle,
            queries.handle,
            keys.handle,
            values.handle,
            scale,
            mask_mode,
            empty_arr,
            empty_arr,
            false,
            s,
        );
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 量化矩阵乘法硬件加速 (Quantized Matrix Multiplication, QMM)
    /// 借助图形处理器 (Graphics Processing Unit, GPU) 硬件流水线执行量化矩阵点积运算
    pub fn quantizedMatmul(
        x: Array,
        w: Array,
        scales: Array,
        biases: ?Array,
        do_transpose: bool,
        group_size: i32,
        bits: i32,
        mode: [*:0]const u8,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res: c.mlx_array = .{ .ctx = null };
        const s = try resolveStream(stream);

        const gs: c.mlx_optional_int = .{ .value = group_size, .has_value = true };
        const b: c.mlx_optional_int = .{ .value = bits, .has_value = true };
        const bias_handle: c.mlx_array = if (biases) |bi| bi.handle else .{ .ctx = null };

        const status = c.mlx_quantized_matmul(
            &res,
            x.handle,
            w.handle,
            scales.handle,
            bias_handle,
            do_transpose,
            gs,
            b,
            mode,
            s,
        );
        if (status != 0 or res.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = res };
    }

    /// 反量化低比特数组 (Dequantize Quantized Weights via GPU Metal Kernel)
    pub fn dequantize(
        w: Array,
        scales: Array,
        biases: ?Array,
        group_size: i32,
        bits: i32,
        mode: [*:0]const u8,
        stream: ?Stream,
    ) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res: c.mlx_array = .{ .ctx = null };
        const s = try resolveStream(stream);

        const gs: c.mlx_optional_int = .{ .value = group_size, .has_value = true };
        const b: c.mlx_optional_int = .{ .value = bits, .has_value = true };
        const bias_handle: c.mlx_array = if (biases) |bi| bi.handle else .{ .ctx = null };
        const opt_none: c.mlx_optional_dtype = .{ .value = 0, .has_value = false };

        const status = c.mlx_dequantize(
            &res,
            w.handle,
            scales.handle,
            bias_handle,
            gs,
            b,
            mode,
            .{ .ctx = null },
            opt_none,
            s,
        );
        if (status != 0 or res.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = res };
    }

    /// 双曲正切算子 (Hyperbolic Tangent, Tanh)
    pub fn tanh(a: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res: c.mlx_array = .{ .ctx = null };
        const s = try resolveStream(stream);

        const status = c.mlx_tanh(&res, a.handle, s);
        if (status != 0 or res.ctx == null) return error.MlxOperationFailed;
        return .{ .handle = res };
    }

    /// 低比特量化权重元组 (Quantized Weights Output)
    pub const QuantizedOutput = struct {
        weights: Array,
        scales: Array,
        biases: ?Array,

        pub fn deinit(self: *QuantizedOutput) void {
            self.weights.deinit();
            self.scales.deinit();
            if (self.biases) |*b| b.deinit();
        }
    };

    /// 借助图形处理器 (Graphics Processing Unit, GPU) 硬件流水线执行量化矩阵转换 (GPU-accelerated Quantize)
    pub fn quantize(
        w: Array,
        group_size: i32,
        bits: i32,
        mode: [*:0]const u8,
        stream: ?Stream,
    ) !QuantizedOutput {
        if (!has_mlx) return error.MlxNotSupported;
        var vec = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(vec);

        const s = try resolveStream(stream);

        const gs = c.mlx_optional_int{ .value = group_size, .has_value = true };
        const b = c.mlx_optional_int{ .value = bits, .has_value = true };
        const null_arr = c.mlx_array{ .ctx = null };

        const status = c.mlx_quantize(&vec, w.handle, gs, b, mode, null_arr, s);
        if (status != 0 or vec.ctx == null) return error.MlxOperationFailed;

        const sz = c.mlx_vector_array_size(vec);
        if (sz < 2) return error.MlxOperationFailed;

        var w_arr = Array.init();
        errdefer w_arr.deinit();
        if (c.mlx_vector_array_get(&w_arr.handle, vec, 0) != 0 or w_arr.handle.ctx == null) return error.MlxOperationFailed;

        var s_arr = Array.init();
        errdefer s_arr.deinit();
        if (c.mlx_vector_array_get(&s_arr.handle, vec, 1) != 0 or s_arr.handle.ctx == null) return error.MlxOperationFailed;

        var b_arr_opt: ?Array = null;
        if (sz >= 3) {
            var b_arr = Array.init();
            if (c.mlx_vector_array_get(&b_arr.handle, vec, 2) != 0 or b_arr.handle.ctx == null) {
                b_arr.deinit();
                return error.MlxOperationFailed;
            }
            b_arr_opt = b_arr;
        }

        return .{
            .weights = w_arr,
            .scales = s_arr,
            .biases = b_arr_opt,
        };
    }
};
