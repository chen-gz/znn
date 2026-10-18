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
    var norm_res = try Array.fastRmsNorm(x_norm, null, 1e-5, stream);
    defer norm_res.deinit();
    norm_res.eval();
    const norm_out = norm_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), norm_out.len);

    // 3. 验证快速缩放点积注意力 (Scaled Dot-Product Attention, SDPA)
    const q_data = [_]f32{0.1} ** 64;
    var q = try Array.fromSlice(&q_data, &.{ 1, 2, 4, 8 });
    defer q.deinit();
    var sdpa_res = try Array.fastScaledDotProductAttention(q, q, q, 1.0 / @sqrt(8.0), false, stream);
    defer sdpa_res.deinit();
    sdpa_res.eval();
    const sdpa_out = sdpa_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 64), sdpa_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), sdpa_out[0], 1e-4);

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
}
