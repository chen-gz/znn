//! Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速后端
//!
//! 基于 Apple 官方 `mlx-c` C API，将矩阵乘法、注意力机制、归一化与旋转位置编码卸载至
//! Apple Silicon 统一内存的图形处理器 (Graphics Processing Unit, GPU) 执行。
//! 依赖工程化 vendored 的 `deps/mlx-c/` 子项目，构建期零外部 Homebrew 依赖。

const std = @import("std");
const builtin = @import("builtin");

/// 当前构建目标是否支持 Apple 机器学习拓展 (Machine Learning Explore, MLX)
pub const has_mlx: bool = (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64);

pub const c = if (has_mlx) @cImport({
    @cInclude("mlx/c/mlx.h");
}) else struct {};

pub const MlxError = error{
    MlxNotSupported,
    MlxInitializationFailed,
    MlxOperationFailed,
    InvalidShape,
    OutOfMemory,
    ShapeMismatch,
    NonContiguousTensor,
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

    /// 从单精度浮点标量创建 MLX 标量数组
    pub fn fromScalarF32(val: f32) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        const arr = c.mlx_array_new_float32(val);
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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);
        const rc = c.mlx_contiguous(&res.handle, self.handle, false, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 将 GPU 计算完毕后的浮点数据安全复制到目标切片中（自动感知并处理非连续与转置布局）
    pub fn copyToSlice(self: Array, dest: []f32) !void {
        const src = self.dataFloat32();
        if (dest.len != src.len) return error.ShapeMismatch;

        if (self.isContiguousRowMajor()) {
            @memcpy(dest, src);
            return;
        }

        const n = self.ndim();
        const c_shape = c.mlx_array_shape(self.handle);
        const c_strides = c.mlx_array_strides(self.handle);

        // 针对注意力算子常见的 4D 张量 (Batch, Head, SeqLen, HeadDim) 极速分块复制 (底层 HeadDim 连续)
        if (n == 4 and c_strides[3] == 1) {
            const B = @as(usize, @intCast(c_shape[0]));
            const H = @as(usize, @intCast(c_shape[1]));
            const T = @as(usize, @intCast(c_shape[2]));
            const D = @as(usize, @intCast(c_shape[3]));
            const sB = c_strides[0];
            const sH = c_strides[1];
            const sT = c_strides[2];

            for (0..B) |b| {
                for (0..H) |h| {
                    for (0..T) |t| {
                        const in_off = b * sB + h * sH + t * sT;
                        const out_off = b * (H * T * D) + h * (T * D) + t * D;
                        @memcpy(dest[out_off .. out_off + D], src[in_off .. in_off + D]);
                    }
                }
            }
            return;
        }

        // 通用 N 维非连续步长重排回写
        var coord = [_]usize{0} ** 8;
        for (0..src.len) |out_idx| {
            var in_idx: usize = 0;
            for (0..n) |d| {
                in_idx += coord[d] * c_strides[d];
            }
            dest[out_idx] = src[in_idx];

            var dim_idx = n;
            while (dim_idx > 0) {
                dim_idx -= 1;
                coord[dim_idx] += 1;
                if (coord[dim_idx] < @as(usize, @intCast(c_shape[dim_idx]))) {
                    break;
                }
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

    /// 获取就绪后的单精度浮点切片指针
    pub fn dataFloat32(self: Array) []const f32 {
        if (!has_mlx or self.handle.ctx == null) return &.{};
        self.eval();
        const ptr = c.mlx_array_data_float32(self.handle);
        const len = c.mlx_array_size(self.handle);
        return ptr[0..len];
    }

    /// 获取张量维度数目
    pub fn ndim(self: Array) usize {
        if (!has_mlx or self.handle.ctx == null) return 0;
        return c.mlx_array_ndim(self.handle);
    }

    fn maybeFreeStream(s: c.mlx_stream, custom: ?Stream) void {
        if (custom == null) {
            _ = c.mlx_stream_free(s);
        }
    }

    /// 矩阵或多维数组转置 (Transpose)
    pub fn transpose(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_transpose_axes(&res.handle, self.handle, &axes_ints, axes.len, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 沿指定轴拼接多个数组 (Concatenate Along Axis)
    pub fn concatenateAxis(arrays: []const Array, axis: c_int, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        if (arrays.len == 0) return error.InvalidShape;

        var handles_buf: [16]c.mlx_array = undefined;
        const raw_handles = if (arrays.len <= 16)
            handles_buf[0..arrays.len]
        else
            try std.heap.c_allocator.alloc(c.mlx_array, arrays.len);
        defer if (arrays.len > 16) std.heap.c_allocator.free(raw_handles);

        for (arrays, 0..) |arr, i| {
            raw_handles[i] = arr.handle;
        }
        const vec = c.mlx_vector_array_new_data(raw_handles.ptr, raw_handles.len);
        defer _ = c.mlx_vector_array_free(vec);

        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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

    /// 标量乘法：`res = self * val`
    pub fn mulScalar(self: Array, val: f32, stream: ?Stream) !Array {
        var s_arr = try fromScalarF32(val);
        defer s_arr.deinit();
        return multiply(self, s_arr, stream);
    }

    /// 数据类型转换 (Data Type Conversion / Astype)
    pub fn astype(self: Array, dtype: DType, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_astype(&res.handle, self.handle, @intFromEnum(dtype), s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 矩阵乘法：`res = a @ b`
    pub fn matmul(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_matmul(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素加法：`res = a + b`
    pub fn add(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_add(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素减法：`res = a - b`
    pub fn subtract(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_subtract(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 逐元素乘法：`res = a * b`
    pub fn multiply(a: Array, b: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_multiply(&res.handle, a.handle, b.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 高斯误差函数 (Error Function, ERF)
    pub fn erf(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        const rc = c.mlx_erf(&res.handle, self.handle, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 高斯误差线性单元激活函数 (Gaussian Error Linear Unit, GELU)
    /// 数学公式：GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    pub fn gelu(self: Array, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var s_scale = try fromScalarF32(0.7071067811865475);
        defer s_scale.deinit();
        var x_scaled = try multiply(self, s_scale, stream);
        defer x_scaled.deinit();

        var erf_val = try x_scaled.erf(stream);
        defer erf_val.deinit();

        var s_one = try fromScalarF32(1.0);
        defer s_one.deinit();
        var one_plus_erf = try add(s_one, erf_val, stream);
        defer one_plus_erf.deinit();

        var s_half = try fromScalarF32(0.5);
        defer s_half.deinit();
        var half_x = try multiply(self, s_half, stream);
        defer half_x.deinit();

        return multiply(half_x, one_plus_erf, stream);
    }

    /// 批量评估多个 MLX 数组 (多算子并行批处理调度，消除重复的主机-设备同步开销)
    pub fn evalMultiple(arrays: []const Array) void {
        if (!has_mlx or arrays.len == 0) return;
        var buf: [32]c.mlx_array = undefined;
        const count = @min(arrays.len, buf.len);
        for (0..count) |i| {
            buf[i] = arrays[i].handle;
        }
        const vec = c.mlx_vector_array_new_data(&buf, count);
        defer _ = c.mlx_vector_array_free(vec);
        _ = c.mlx_eval(vec);
    }

    /// Softmax 归一化算子
    pub fn softmax(a: Array, axis: c_int, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

        var axes = [_]c_int{axis};
        const rc = c.mlx_softmax(&res.handle, a.handle, &axes, 1, s);
        if (rc != 0 or res.handle.ctx == null) return error.MlxOperationFailed;
        return res;
    }

    /// 快速均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    pub fn fastRmsNorm(x: Array, weight: ?Array, eps: f32, stream: ?Stream) !Array {
        if (!has_mlx) return error.MlxNotSupported;
        var res = Array.init();
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
        const s = if (stream) |st| st.handle else c.mlx_default_gpu_stream_new();
        defer maybeFreeStream(s, stream);

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
};

test "MLX GPU backend matmul and fast kernels" {
    if (!has_mlx) return;

    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    // 1. 验证矩阵乘法 (GEMM)
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var a = try Array.fromSlice(&a_data, &.{ 2, 2 });
    defer a.deinit();

    var c_mat = try Array.matmul(a, a, stream);
    defer c_mat.deinit();
    c_mat.eval();

    const out = c_mat.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out[3], 1e-4);

    // 2. 验证快速均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    var x_norm = try Array.fromSlice(&[_]f32{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, 4 });
    defer x_norm.deinit();
    var w_norm = try Array.fromSlice(&[_]f32{ 2.0, 2.0, 2.0, 2.0 }, &.{ 4 });
    defer w_norm.deinit();
    var norm_res = try Array.fastRmsNorm(x_norm, w_norm, 1e-5, stream);
    defer norm_res.deinit();
    norm_res.eval();
    const norm_out = norm_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), norm_out.len);
    // mean(x^2) = (1+4+9+16)/4 = 7.5. rsqrt(7.5) = 0.36514837. With weight 2.0: x * 0.7302967
    try std.testing.expectApproxEqAbs(@as(f32, 0.7302967), norm_out[0], 1e-4);

    // 2.1 验证多维高阶张量快速均方根层归一化 (4D RMSNorm)
    var x_4d_norm = try Array.fromSlice(&[_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0 }, &.{ 1, 2, 1, 4 });
    defer x_4d_norm.deinit();
    var norm_4d_res = try Array.fastRmsNorm(x_4d_norm, w_norm, 1e-5, stream);
    defer norm_4d_res.deinit();
    norm_4d_res.eval();
    const norm_4d_out = norm_4d_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 8), norm_4d_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7302967), norm_4d_out[0], 1e-4);

    // 3. 验证快速缩放点积注意力 (Scaled Dot-Product Attention, SDPA)
    const q_data = [_]f32{0.1} ** 64;
    var q = try Array.fromSlice(&q_data, &.{ 1, 2, 4, 8 });
    defer q.deinit();
    var sdpa_res = try Array.fastScaledDotProductAttention(q, q, q, 1.0 / @sqrt(8.0), false, stream);
    defer sdpa_res.deinit();
    sdpa_res.eval();
    const sdpa_out = sdpa_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 64), sdpa_out.len);
    // 3.1 验证单步增量解码 (Decode Step) 与分组查询注意力 (Grouped-Query Attention, GQA)
    const q_step = [_]f32{0.2} ** 16; // [1, 2 heads, 1, 8 dim]
    var q_dec = try Array.fromSlice(&q_step, &.{ 1, 2, 1, 8 });
    defer q_dec.deinit();

    const kv_cached = [_]f32{0.1} ** 32; // [1, 1 kv head, 4 seq_len, 8 dim]
    var k_dec = try Array.fromSlice(&kv_cached, &.{ 1, 1, 4, 8 });
    defer k_dec.deinit();
    var v_dec = try Array.fromSlice(&kv_cached, &.{ 1, 1, 4, 8 });
    defer v_dec.deinit();

    var gqa_sdpa = try Array.fastScaledDotProductAttention(q_dec, k_dec, v_dec, 1.0 / @sqrt(8.0), false, stream);
    defer gqa_sdpa.deinit();
    gqa_sdpa.eval();
    const gqa_out = gqa_sdpa.dataFloat32();
    try std.testing.expectEqual(@as(usize, 16), gqa_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), gqa_out[0], 1e-4);

    // 3.2 验证旋转位置编码 (Rotary Position Embedding, RoPE)
    const rope_in = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var rope_arr = try Array.fromSlice(&rope_in, &.{ 1, 1, 1, 4 });
    defer rope_arr.deinit();
    var rope_res = try Array.fastRoPE(rope_arr, 4, false, 10000.0, 1.0, 1, stream);
    defer rope_res.deinit();
    rope_res.eval();
    const rope_out = rope_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), rope_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, -1.9841107), rope_out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9599), rope_out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.4623779), rope_out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0198), rope_out[3], 1e-4);

    // 3.3 验证数组重构 (Reshape)、轴拼接 (Concatenate) 与轴切片 (Slice)
    var arr_a = try Array.fromSlice(&[_]f32{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, 1, 2, 2 });
    defer arr_a.deinit();
    var arr_b = try Array.fromSlice(&[_]f32{ 5.0, 6.0 }, &.{ 1, 1, 1, 2 });
    defer arr_b.deinit();

    var concat_res = try Array.concatenateAxis(&.{ arr_a, arr_b }, 2, stream);
    defer concat_res.deinit();
    concat_res.eval();
    var concat_dims: [4]usize = undefined;
    _ = try concat_res.shape(&concat_dims);
    try std.testing.expectEqual(@as(usize, 3), concat_dims[2]);

    var slice_res = try concat_res.sliceAxis(2, 1, 3, stream);
    defer slice_res.deinit();
    slice_res.eval();
    const slice_out = slice_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), slice_out.len);
    try std.testing.expectEqual(@as(f32, 3.0), slice_out[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice_out[1]);
    try std.testing.expectEqual(@as(f32, 5.0), slice_out[2]);
    try std.testing.expectEqual(@as(f32, 6.0), slice_out[3]);

    var mul_scalar_res = try slice_res.mulScalar(2.0, stream);
    defer mul_scalar_res.deinit();
    mul_scalar_res.eval();
    const mul_out = mul_scalar_res.dataFloat32();
    try std.testing.expectEqual(@as(f32, 6.0), mul_out[0]);
    try std.testing.expectEqual(@as(f32, 12.0), mul_out[3]);

    // 4. 验证零拷贝创建与形状查询 (Zero-Copy & Shape Query)
    var host_buf = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };
    var zc_arr = try Array.fromSliceZeroCopy(&host_buf, &.{ 2, 3 });
    defer zc_arr.deinit();
    try std.testing.expectEqual(@as(usize, 6), zc_arr.size());
    try std.testing.expectEqual(@as(usize, 2), zc_arr.ndim());
    var dims: [4]usize = undefined;
    const n = try zc_arr.shape(&dims);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(usize, 2), dims[0]);
    try std.testing.expectEqual(@as(usize, 3), dims[1]);

    var copy_dest = [_]f32{0.0} ** 6;
    try zc_arr.copyToSlice(&copy_dest);
    try std.testing.expectEqualSlices(f32, &host_buf, &copy_dest);

    // 6. 验证由 Q4Block 转换为 MLX 格式后调用 mlx_quantized_matmul
    var q4_weights: [32]f32 = undefined;
    for (&q4_weights, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) - 16.0;
    }
    // 量化为 Q4Block
    var q4_block: [1]extern struct {
        scale: f16,
        qs: [16]u8,
    } = undefined;

    // 使用标准 Q4Block 算法量化
    var amax: f32 = 0.0;
    for (q4_weights) |v| {
        const av = @abs(v);
        if (av > amax) amax = av;
    }
    const scale_f32 = amax / 7.0;
    const inv_scale = if (scale_f32 > 0.0) 1.0 / scale_f32 else 0.0;
    q4_block[0].scale = @floatCast(scale_f32);
    for (0..16) |j| {
        const v0 = q4_weights[j];
        const v1 = q4_weights[j + 16];
        const q0_f = std.math.clamp(std.math.round(v0 * inv_scale) + 8.0, 0.0, 15.0);
        const q1_f = std.math.clamp(std.math.round(v1 * inv_scale) + 8.0, 0.0, 15.0);
        const q0: u8 = @intFromFloat(q0_f);
        const q1: u8 = @intFromFloat(q1_f);
        q4_block[0].qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }

    // 计算 CPU 基准点积 (x = [1, 1, ..., 1])
    var x_ones: [32]f32 = undefined;
    for (&x_ones) |*v| v.* = 1.0;
    var expected_dot: f32 = 0.0;
    const d: f32 = @floatCast(q4_block[0].scale);
    for (0..16) |j| {
        const byte = q4_block[0].qs[j];
        const q0: f32 = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
        const q1: f32 = @as(f32, @floatFromInt(byte >> 4)) - 8.0;
        expected_dot += (q0 * d) * x_ones[j];
        expected_dot += (q1 * d) * x_ones[j + 16];
    }

    // 打包为 MLX uint32 格式
    var mlx_u32: [4]u32 = undefined;
    const qs = q4_block[0].qs;
    var w0: u32 = 0;
    var w1: u32 = 0;
    var w2: u32 = 0;
    var w3: u32 = 0;
    for (0..8) |j| {
        const shift: u5 = @intCast(j * 4);
        w0 |= @as(u32, qs[j] & 0x0F) << shift;
        w1 |= @as(u32, qs[j + 8] & 0x0F) << shift;
        w2 |= @as(u32, qs[j] >> 4) << shift;
        w3 |= @as(u32, qs[j + 8] >> 4) << shift;
    }
    mlx_u32[0] = w0;
    mlx_u32[1] = w1;
    mlx_u32[2] = w2;
    mlx_u32[3] = w3;

    const w_shape = [_]c_int{ 1, 4 }; // [out_features=1, in_features/8 = 4]
    const mlx_w_arr = c.mlx_array_new_data(@ptrCast(&mlx_u32), &w_shape, 2, @intFromEnum(DType.uint32));
    var mlx_w = Array{ .handle = mlx_w_arr };
    defer mlx_w.deinit();

    var mlx_scale_val = [1]f16{q4_block[0].scale};
    var mlx_bias_val = [1]f16{@floatCast(-8.0 * d)};
    const sb_shape = [_]c_int{ 1, 1 };

    const mlx_s_arr = c.mlx_array_new_data(@ptrCast(&mlx_scale_val), &sb_shape, 2, @intFromEnum(DType.float16));
    var mlx_s = Array{ .handle = mlx_s_arr };
    defer mlx_s.deinit();

    const mlx_b_arr = c.mlx_array_new_data(@ptrCast(&mlx_bias_val), &sb_shape, 2, @intFromEnum(DType.float16));
    var mlx_b = Array{ .handle = mlx_b_arr };
    defer mlx_b.deinit();

    var mlx_x = try Array.fromSlice(&x_ones, &.{ 1, 32 });
    defer mlx_x.deinit();

    const group_size: c.mlx_optional_int = .{ .value = 32, .has_value = true };
    const bits: c.mlx_optional_int = .{ .value = 4, .has_value = true };

    var qmm_res: c.mlx_array = .{ .ctx = null };
    const qmm_st = c.mlx_quantized_matmul(
        &qmm_res,
        mlx_x.handle,
        mlx_w.handle,
        mlx_s.handle,
        mlx_b.handle,
        true, // transpose
        group_size,
        bits,
        "affine",
        stream.handle,
    );
    try std.testing.expectEqual(@as(c_int, 0), qmm_st);
    var qmm_arr = Array{ .handle = qmm_res };
    defer qmm_arr.deinit();
    qmm_arr.eval();
    stream.synchronize();

    const qmm_out = qmm_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 1), qmm_out.len);
    try std.testing.expectApproxEqAbs(expected_dot, qmm_out[0], 1e-2);

    // 4. 验证高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 与批量评估
    const gelu_inputs = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var x_gelu = try Array.fromSlice(&gelu_inputs, &.{ 1, 4 });
    defer x_gelu.deinit();

    var y_gelu = try x_gelu.gelu(stream);
    defer y_gelu.deinit();

    Array.evalMultiple(&.{y_gelu});
    const gelu_out = y_gelu.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), gelu_out.len);
    // GELU(0) = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu_out[0], 1e-4);
    // GELU(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) = 0.8413447
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu_out[1], 1e-4);
    // GELU(-1) = 0.5 * (-1) * (1 + erf(-1/sqrt(2))) = -0.158655
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), gelu_out[2], 1e-4);
}

