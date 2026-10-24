//! Backend 模块单元测试套件 (Backend Unit Tests)
//!
//! 统一验证 Apple Silicon Metal GPU 硬件加速计算后端 (`mlx.zig`)、模型算子卸载基础设施 (`offload.zig`) 与采样基础设施 (`sampling.zig`)。

const std = @import("std");
const mlx = @import("mlx.zig");
const offload = @import("offload.zig");
const sampling = @import("sampling.zig");
const activation = @import("activation.zig");
const attention_mask = @import("attention_mask.zig");
const cblas = @import("cblas.zig");
const Executor = @import("executor.zig").Executor;

const has_mlx = mlx.has_mlx;
const c = mlx.c;
const DType = mlx.DType;
const Stream = mlx.Stream;
const Array = mlx.Array;
const VectorArray = mlx.VectorArray;
const compile = mlx.compile;

const isAvailable = offload.isAvailable;
const LmHeadWeights = offload.LmHeadWeights;
const Q4Weights = offload.Q4Weights;
const GpuKvCache = offload.GpuKvCache;
const q4MatMul = offload.q4MatMul;

test "MLX GPU backend matmul and fast kernels" {
    if (!has_mlx) return;

    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    // 1. 验证矩阵乘法 (General Matrix Multiply, GEMM)
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var a = try Array.fromSlice(&a_data, &.{ 2, 2 });
    defer a.deinit();

    var c_mat = try Array.matmul(a, a, stream);
    defer c_mat.deinit();
    c_mat.eval();

    const out = try c_mat.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out[3], 1e-4);

    // 2. 验证快速均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    var x_norm = try Array.fromSlice(&[_]f32{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, 4 });
    defer x_norm.deinit();
    var w_norm = try Array.fromSlice(&[_]f32{ 2.0, 2.0, 2.0, 2.0 }, &.{4});
    defer w_norm.deinit();
    var norm_res = try Array.fastRmsNorm(x_norm, w_norm, 1e-5, stream);
    defer norm_res.deinit();
    norm_res.eval();
    const norm_out = try norm_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), norm_out.len);
    // mean(x^2) = (1+4+9+16)/4 = 7.5. rsqrt(7.5) = 0.36514837. With weight 2.0: x * 0.7302967
    try std.testing.expectApproxEqAbs(@as(f32, 0.7302967), norm_out[0], 1e-4);

    // 2.1 验证多维高阶张量快速均方根层归一化 (4D RMSNorm)
    var x_4d_norm = try Array.fromSlice(&[_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0 }, &.{ 1, 2, 1, 4 });
    defer x_4d_norm.deinit();
    var norm_4d_res = try Array.fastRmsNorm(x_4d_norm, w_norm, 1e-5, stream);
    defer norm_4d_res.deinit();
    norm_4d_res.eval();
    const norm_4d_out = try norm_4d_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 8), norm_4d_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7302967), norm_4d_out[0], 1e-4);

    // 3. 验证快速缩放点积注意力 (Scaled Dot-Product Attention, SDPA)
    const q_data: [64]f32 = @splat(0.1);
    var q = try Array.fromSlice(&q_data, &.{ 1, 2, 4, 8 });
    defer q.deinit();
    var sdpa_res = try Array.fastScaledDotProductAttention(q, q, q, 1.0 / @sqrt(8.0), .none, stream);
    defer sdpa_res.deinit();
    sdpa_res.eval();
    const sdpa_out = try sdpa_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 64), sdpa_out.len);
    // 3.1 验证单步增量解码 (Decode Step) 与分组查询注意力 (Grouped-Query Attention, GQA)
    const q_step: [16]f32 = @splat(0.2); // [1, 2 heads, 1, 8 dim]
    var q_dec = try Array.fromSlice(&q_step, &.{ 1, 2, 1, 8 });
    defer q_dec.deinit();

    const kv_cached: [32]f32 = @splat(0.1); // [1, 1 kv head, 4 seq_len, 8 dim]
    var k_dec = try Array.fromSlice(&kv_cached, &.{ 1, 1, 4, 8 });
    defer k_dec.deinit();
    var v_dec = try Array.fromSlice(&kv_cached, &.{ 1, 1, 4, 8 });
    defer v_dec.deinit();

    var gqa_sdpa = try Array.fastScaledDotProductAttention(q_dec, k_dec, v_dec, 1.0 / @sqrt(8.0), .none, stream);
    defer gqa_sdpa.deinit();
    gqa_sdpa.eval();
    const gqa_out = try gqa_sdpa.dataFloat32();
    try std.testing.expectEqual(@as(usize, 16), gqa_out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), gqa_out[0], 1e-4);

    // 3.1.1 验证带前缀因果注意力 (Causal SDPA with M > 1 and K > M)
    const q_prefix: [16]f32 = @splat(0.2); // [1, 1 head, 2 seq, 8 dim]
    var q_pref_arr = try Array.fromSlice(&q_prefix, &.{ 1, 1, 2, 8 });
    defer q_pref_arr.deinit();
    const kv_prefix: [40]f32 = @splat(0.1); // [1, 1 head, 5 seq, 8 dim]
    var k_pref_arr = try Array.fromSlice(&kv_prefix, &.{ 1, 1, 5, 8 });
    defer k_pref_arr.deinit();
    var v_pref_arr = try Array.fromSlice(&kv_prefix, &.{ 1, 1, 5, 8 });
    defer v_pref_arr.deinit();
    var pref_sdpa = try Array.fastScaledDotProductAttention(q_pref_arr, k_pref_arr, v_pref_arr, 1.0 / @sqrt(8.0), .causal, stream);
    defer pref_sdpa.deinit();
    pref_sdpa.eval();
    const pref_out = try pref_sdpa.dataFloat32();
    try std.testing.expectEqual(@as(usize, 16), pref_out.len);

    // 3.2 验证旋转位置编码 (Rotary Position Embedding, RoPE)
    const rope_in = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var rope_arr = try Array.fromSlice(&rope_in, &.{ 1, 1, 1, 4 });
    defer rope_arr.deinit();
    var rope_res = try Array.fastRoPE(rope_arr, 4, false, 10000.0, 1.0, 1, stream);
    defer rope_res.deinit();
    rope_res.eval();
    const rope_out = try rope_res.dataFloat32();
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
    const slice_out = try slice_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), slice_out.len);
    try std.testing.expectEqual(@as(f32, 3.0), slice_out[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice_out[1]);
    try std.testing.expectEqual(@as(f32, 5.0), slice_out[2]);
    try std.testing.expectEqual(@as(f32, 6.0), slice_out[3]);

    var mul_scalar_res = try slice_res.mulScalar(2.0, stream);
    defer mul_scalar_res.deinit();
    mul_scalar_res.eval();
    const mul_out = try mul_scalar_res.dataFloat32();
    try std.testing.expectEqual(@as(f32, 6.0), mul_out[0]);
    try std.testing.expectEqual(@as(f32, 12.0), mul_out[3]);

    // 3.4 验证全零数组 (Zeros) 与切片原地更新 (Slice Update)
    var zero_base = try Array.zeros(&.{ 1, 1, 4, 2 }, .float32, stream);
    defer zero_base.deinit();

    var update_piece = try Array.fromSlice(&[_]f32{ 7.0, 8.0 }, &.{ 1, 1, 1, 2 });
    defer update_piece.deinit();

    var updated_arr = try zero_base.sliceUpdateAxis(update_piece, 2, 2, 3, stream);
    defer updated_arr.deinit();
    updated_arr.eval();
    const up_data = try updated_arr.dataFloat32();
    try std.testing.expectEqual(@as(f32, 0.0), up_data[0]); // [0, 0]
    try std.testing.expectEqual(@as(f32, 0.0), up_data[1]); // [0, 1]
    try std.testing.expectEqual(@as(f32, 0.0), up_data[2]); // [1, 0]
    try std.testing.expectEqual(@as(f32, 0.0), up_data[3]); // [1, 1]
    try std.testing.expectEqual(@as(f32, 7.0), up_data[4]); // [2, 0]
    try std.testing.expectEqual(@as(f32, 8.0), up_data[5]); // [2, 1]
    try std.testing.expectEqual(@as(f32, 0.0), up_data[6]); // [3, 0]
    try std.testing.expectEqual(@as(f32, 0.0), up_data[7]); // [3, 1]

    // 验证 SDPA 接收从 sliceAxis 产生的非全长缓存切片
    var q_single = try Array.fromSlice(&[_]f32{ 1.0, 1.0 }, &.{ 1, 1, 1, 2 });
    defer q_single.deinit();
    var k_sub = try updated_arr.sliceAxis(2, 0, 3, stream);
    defer k_sub.deinit();
    var v_sub = try updated_arr.sliceAxis(2, 0, 3, stream);
    defer v_sub.deinit();
    var sdpa_sub = try Array.fastScaledDotProductAttention(q_single, k_sub, v_sub, 1.0, .none, stream);
    defer sdpa_sub.deinit();
    sdpa_sub.eval();
    const sdpa_sub_out = try sdpa_sub.dataFloat32();
    try std.testing.expectEqual(@as(usize, 2), sdpa_sub_out.len);

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

    var copy_dest: [6]f32 = @splat(0.0);
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
    const mlx_w_arr = c.mlx_array_new_data(@ptrCast(&mlx_u32), &w_shape, 2, @backingInt(DType.uint32));
    var mlx_w = Array{ .handle = mlx_w_arr };
    defer mlx_w.deinit();

    var mlx_scale_val = [1]f16{q4_block[0].scale};
    var mlx_bias_val = [1]f16{@floatCast(-8.0 * d)};
    const sb_shape = [_]c_int{ 1, 1 };

    const mlx_s_arr = c.mlx_array_new_data(@ptrCast(&mlx_scale_val), &sb_shape, 2, @backingInt(DType.float16));
    var mlx_s = Array{ .handle = mlx_s_arr };
    defer mlx_s.deinit();

    const mlx_b_arr = c.mlx_array_new_data(@ptrCast(&mlx_bias_val), &sb_shape, 2, @backingInt(DType.float16));
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

    const qmm_out = try qmm_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 1), qmm_out.len);
    try std.testing.expectApproxEqAbs(expected_dot, qmm_out[0], 1e-2);

    // 4. 验证高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 与批量评估
    const gelu_inputs = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var x_gelu = try Array.fromSlice(&gelu_inputs, &.{ 1, 4 });
    defer x_gelu.deinit();

    var y_gelu = try x_gelu.gelu(.none, stream);
    defer y_gelu.deinit();
    var y_gelu_tanh = try x_gelu.gelu(.tanh, stream);
    defer y_gelu_tanh.deinit();

    Array.evalMultiple(&.{ y_gelu, y_gelu_tanh });
    const gelu_out = try y_gelu.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), gelu_out.len);
    // GELU(0) = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu_out[0], 1e-4);
    // GELU(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) = 0.8413447
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu_out[1], 1e-4);
    // GELU(-1) = 0.5 * (-1) * (1 + erf(-1/sqrt(2))) = -0.158655
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), gelu_out[2], 1e-4);
    // 双曲正切近似形式与 CPU 标量参考实现一致：GELU_tanh(1) = 0.8411920，GELU_tanh(-1) = -0.1588080
    const gelu_tanh_out = try y_gelu_tanh.dataFloat32();
    for (gelu_inputs, gelu_tanh_out) |x_val, y_val| {
        try std.testing.expectApproxEqAbs(activation.gelu(.tanh, x_val), y_val, 1e-5);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.8411920), gelu_tanh_out[1], 1e-5);

    // 5. 验证原生 GPU quantize 与 quantizedMatmul 端到端协同
    var test_w_data: [64]f32 = undefined;
    for (0..64) |i| {
        test_w_data[i] = @floatFromInt(i);
    }
    var test_w = try Array.fromSlice(&test_w_data, &.{ 1, 64 });
    defer test_w.deinit();

    var test_w_bf16 = try test_w.astype(.bfloat16, stream);
    defer test_w_bf16.deinit();

    var q8_pkg = try Array.quantize(test_w_bf16, 32, 8, "affine", stream);
    defer q8_pkg.deinit();

    var test_x_data: [64]f32 = undefined;
    @memset(&test_x_data, 1.0);
    var test_x = try Array.fromSlice(&test_x_data, &.{ 1, 64 });
    defer test_x.deinit();

    var qmm_q8 = try test_x.quantizedMatmul(q8_pkg.weights, q8_pkg.scales, q8_pkg.biases, true, 32, 8, "affine", stream);
    defer qmm_q8.deinit();
    qmm_q8.eval();
    stream.synchronize();

    // 6. 验证四维键值缓存 (Key-Value Cache, KVCache) 预分配切片与定点更新
    // 形状 [1, 2, 4, 2] (num_kv=2, max_len=4, head_dim=2)
    var cache_4d = try Array.zeros(&.{ 1, 2, 4, 2 }, .float32, stream);
    defer cache_4d.deinit();

    // 构造单步更新 token [1, 2, 1, 2]
    const upd_data = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var upd_tok = try Array.fromSlice(&upd_data, &.{ 1, 2, 1, 2 });
    defer upd_tok.deinit();

    // 更新到位置 1 (pos = 1)
    const updated_cache = try cache_4d.sliceUpdate4dAxis2(upd_tok, 2, 1, 2, 2, stream);
    cache_4d.deinit();
    cache_4d = updated_cache;

    // 提取前 2 个词元 [1, 2, 2, 2]
    var view_2 = try cache_4d.slice4dAxis2(2, 0, 2, 2, stream);
    defer view_2.deinit();

    var view_c = try view_2.contiguous(stream);
    defer view_c.deinit();
    view_c.eval();
    stream.synchronize();

    const view_data = try view_c.dataFloat32();
    try std.testing.expectEqual(@as(usize, 8), view_data.len);
    // kv=0, t=0 -> 0.0, 0.0
    try std.testing.expectEqual(@as(f32, 0.0), view_data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), view_data[1]);
    // kv=0, t=1 -> 10.0, 20.0
    try std.testing.expectEqual(@as(f32, 10.0), view_data[2]);
    try std.testing.expectEqual(@as(f32, 20.0), view_data[3]);
    // kv=1, t=0 -> 0.0, 0.0
    try std.testing.expectEqual(@as(f32, 0.0), view_data[4]);
    try std.testing.expectEqual(@as(f32, 0.0), view_data[5]);
    // kv=1, t=1 -> 30.0, 40.0
    try std.testing.expectEqual(@as(f32, 30.0), view_data[6]);
    // 7. 验证切片视图直接送入 fastScaledDotProductAttention 正常执行
    var q_probe = try Array.zeros(&.{ 1, 2, 1, 2 }, .float32, stream);
    defer q_probe.deinit();
    var attn = try Array.fastScaledDotProductAttention(q_probe, view_2, view_2, 1.0, .none, stream);
    defer attn.deinit();
    attn.eval();
    stream.synchronize();

    // 8. 验证连续多步增量写入 (Sequential Multi-step KVCache Updates)
    const upd2_data = [_]f32{ 50.0, 60.0, 70.0, 80.0 };
    var upd2_tok = try Array.fromSlice(&upd2_data, &.{ 1, 2, 1, 2 });
    defer upd2_tok.deinit();

    const cache_step2 = try cache_4d.sliceUpdate4dAxis2(upd2_tok, 2, 2, 3, 2, stream);
    cache_4d.deinit();
    cache_4d = cache_step2;

    var view_3 = try cache_4d.slice4dAxis2(2, 0, 3, 2, stream);
    defer view_3.deinit();

    var attn2 = try Array.fastScaledDotProductAttention(q_probe, view_3, view_3, 1.0, .none, stream);
    defer attn2.deinit();
    attn2.eval();
    stream.synchronize();

    // 9. 验证 takeAxis 沿指定轴批量查表 (Take Axis / GPU Embedding Lookup)
    const emb_table_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [3, 2]
    var emb_table = try Array.fromSlice(&emb_table_data, &.{ 3, 2 });
    defer emb_table.deinit();

    const idx_data = [_]i32{ 2, 0 };
    var idx_arr = try Array.fromSliceI32(&idx_data, &.{2});
    defer idx_arr.deinit();

    var taken = try emb_table.takeAxis(idx_arr, 0, stream);
    defer taken.deinit();
    taken.eval();
    const taken_out = try taken.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), taken_out.len);
    // idx 2 -> 5.0, 6.0; idx 0 -> 1.0, 2.0
    try std.testing.expectEqual(@as(f32, 5.0), taken_out[0]);
    try std.testing.expectEqual(@as(f32, 6.0), taken_out[1]);
    try std.testing.expectEqual(@as(f32, 1.0), taken_out[2]);
    try std.testing.expectEqual(@as(f32, 2.0), taken_out[3]);

    // 10. 验证 geglu (Gated GELU) 激活计算
    const geglu_in = [_]f32{ 0.0, 1.0, 2.0, 3.0 }; // gate=[0.0, 1.0], up=[2.0, 3.0]
    var geglu_arr = try Array.fromSlice(&geglu_in, &.{ 1, 4 });
    defer geglu_arr.deinit();

    var geglu_res = try geglu_arr.geglu(2, .none, stream);
    defer geglu_res.deinit();
    geglu_res.eval();
    const geglu_out = try geglu_res.dataFloat32();
    try std.testing.expectEqual(@as(usize, 2), geglu_out.len);
    // GELU(0) * 2.0 = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), geglu_out[0], 1e-4);
    // GELU(1) * 3.0 = 0.8413447 * 3.0 = 2.524034
    try std.testing.expectApproxEqAbs(@as(f32, 2.524034), geglu_out[1], 1e-4);

    // 11. 验证计算图函数闭包静态编译 (Closure & compile)
    const Helper = struct {
        fn compute(vres: [*c]c.mlx_vector_array, in: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
            if (!has_mlx) return 1;
            _ = payload;
            var x_handle: c.mlx_array = .{ .ctx = null };
            if (c.mlx_vector_array_get(&x_handle, in, 0) != 0 or x_handle.ctx == null) return 1;
            const x = Array{ .handle = x_handle };

            var gelu_res = x.gelu(.none, null) catch return 1;
            defer gelu_res.deinit();

            var one = Array.fromScalarF32(1.0) catch return 1;
            defer one.deinit();

            var res_arr = Array.add(gelu_res, one, null) catch return 1;
            defer res_arr.deinit();

            const res_vec = c.mlx_vector_array_new_data(&[_]c.mlx_array{res_arr.handle}, 1);
            _ = c.mlx_vector_array_set(vres, res_vec);
            _ = c.mlx_vector_array_free(res_vec);
            return 0;
        }
    };

    var compiled = try compile(Helper.compute, null, false);
    defer compiled.deinit();

    const in_comp_data = [_]f32{ 0.0, 1.0, 2.0, -1.0 };
    var in_comp_arr = try Array.fromSlice(&in_comp_data, &.{ 1, 4 });
    defer in_comp_arr.deinit();

    var in_comp_vec = VectorArray.fromArrays(&.{in_comp_arr});
    defer in_comp_vec.deinit();

    var out_comp_vec = try compiled.apply(in_comp_vec);
    defer out_comp_vec.deinit();

    try std.testing.expectEqual(@as(usize, 1), out_comp_vec.size());
    var out_comp_arr = out_comp_vec.get(0).?;
    defer out_comp_arr.deinit();
    out_comp_arr.eval();

    const out_comp_slice = try out_comp_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), out_comp_slice.len);
    // x=0: gelu(0) + 1.0 = 0.0 + 1.0 = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_comp_slice[0], 1e-3);
    // x=1: gelu(1) + 1.0 = 0.8413447 + 1.0 = 1.8413447
    try std.testing.expectApproxEqAbs(@as(f32, 1.8413447), out_comp_slice[1], 1e-3);

    // 12. 验证脑浮点数 (Brain Floating Point 16-bit, bfloat16) 标量精度保持与融合门控激活 (fusedGeluMul)
    const bf16_bits = Array.f32ToBf16Bits(2.0);
    try std.testing.expectEqual(@as(u16, 0x4000), bf16_bits);

    var s_bf16 = try Array.fromScalarBf16(2.0);
    defer s_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, s_bf16.dtype());

    var x_bf16 = try in_comp_arr.astype(.bfloat16, stream);
    defer x_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, x_bf16.dtype());

    var scaled_bf16 = try x_bf16.mulScalar(2.0, stream);
    defer scaled_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, scaled_bf16.dtype());

    var gelu_bf16 = try x_bf16.gelu(.tanh, stream);
    defer gelu_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, gelu_bf16.dtype());

    // 两种近似方式的融合闭包分别与逐算子回退路径的结果一致
    inline for (.{ activation.GeluApproximation.none, activation.GeluApproximation.tanh }) |approx| {
        var compiled_gm = try Array.compileGeluMul(approx);
        defer compiled_gm.deinit();
        var fused_gm_bf16 = try Array.fusedGeluMul(x_bf16, scaled_bf16, approx, compiled_gm, stream);
        defer fused_gm_bf16.deinit();
        try std.testing.expectEqual(DType.bfloat16, fused_gm_bf16.dtype());

        var fused_f32 = try Array.fusedGeluMul(in_comp_arr, in_comp_arr, approx, compiled_gm, stream);
        defer fused_f32.deinit();
        var eager_f32 = try Array.fusedGeluMul(in_comp_arr, in_comp_arr, approx, null, stream);
        defer eager_f32.deinit();
        Array.evalMultiple(&.{ fused_f32, eager_f32 });
        const fused_vals = try fused_f32.dataFloat32();
        const eager_vals = try eager_f32.dataFloat32();
        for (in_comp_data, fused_vals, eager_vals) |x_val, f_val, e_val| {
            try std.testing.expectApproxEqAbs(e_val, f_val, 1e-5);
            try std.testing.expectApproxEqAbs(activation.gelu(approx, x_val) * x_val, e_val, 1e-5);
        }
    }
}

test "backend offload infrastructure" {
    if (!isAvailable()) return;

    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    // 1. 验证 LmHeadWeights 词表投影卸载
    // 构造 4 个词元、维度为 2 的测试词表 (BF16)
    // token 0: [1.0, 0.0] -> 0x3f80, 0x0000
    // token 1: [0.0, 2.0] -> 0x0000, 0x4000
    // token 2: [3.0, 3.0] -> 0x4040, 0x4040 (最大值目标)
    // token 3: [0.5, 0.5] -> 0x3f00, 0x3f00
    const embed_u16 = [_]u16{
        0x3f80, 0x0000,
        0x0000, 0x4000,
        0x4040, 0x4040,
        0x3f00, 0x3f00,
    };

    var lm_head = try LmHeadWeights.init(&embed_u16, 4, 2, stream);
    defer lm_head.deinit();

    const h_last = [_]f32{ 1.0, 1.0 };
    var h_arr = try mlx.Array.fromSlice(&h_last, &.{ 1, 2 });
    defer h_arr.deinit();
    var logits_arr = try lm_head.forwardArray(h_arr, stream);
    defer logits_arr.deinit();
    const logits = try logits_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), logits.len);
    // 无重复惩罚时，token 2 得分 1.0*3.0 + 1.0*3.0 = 6.0 最高
    try std.testing.expectEqual(@as(usize, 2), sampling.argmaxWithPenalty(logits, 4, &.{}, &.{}, 1.0));
    // 对 token 2 施加极端惩罚 10.0，使得分变为 0.6，此时 token 1 得分 2.0 胜出
    try std.testing.expectEqual(@as(usize, 1), sampling.argmaxWithPenalty(logits, 4, &.{}, &.{2}, 10.0));

    // GPU 加性排除掩码与 CPU 采样排除同一组词元：排除 token 2 后 token 1 胜出
    const excluded = [_]sampling.TokenRange{.single(2)};
    var mask = try offload.TokenExclusionMask.init(std.testing.allocator, 4, &excluded);
    defer mask.deinit();
    var logits_bf16 = try lm_head.forwardArrayBf16(h_arr, stream);
    defer logits_bf16.deinit();
    var best_arr = try mask.argmax(logits_bf16, stream);
    defer best_arr.deinit();
    try std.testing.expectEqual(@as(u32, 1), try best_arr.itemUint32());
    try std.testing.expectEqual(@as(usize, 1), sampling.argmaxWithPenalty(logits, 4, &excluded, &.{}, 1.0));

    // 2. 验证 Q4Weights 硬件量化矩阵乘法卸载
    var raw_block: [18]u8 = undefined;
    const scale_bits: u16 = 0x3c00; // f16 = 1.0
    std.mem.writeInt(u16, raw_block[0..2], scale_bits, .little);
    // 所有权重置为 (10 - 8) = 2.0 (低 4 位为 10 即 0xA，高 4 位为 10 即 0xA)
    @memset(raw_block[2..18], 0xAA);

    var q4_w = try Q4Weights.initFromBlocks(std.testing.allocator, &raw_block, 32, 1, stream);
    defer q4_w.deinit();

    var x_vec: [32]f32 = undefined;
    for (&x_vec) |*v| v.* = 1.0;
    var out_res: [1]f32 = undefined;

    const DummyTensor = struct {
        data: []f32,
        shape: struct {
            dims: [2]usize,
            len: usize = 2,
        },
    };
    var x_t = DummyTensor{ .data = &x_vec, .shape = .{ .dims = .{ 1, 32 }, .len = 2 } };
    var out_t = DummyTensor{ .data = &out_res, .shape = .{ .dims = .{ 1, 1 }, .len = 2 } };

    try q4MatMul(&x_t, &out_t, &q4_w, stream);
    stream.synchronize();

    // 32 个 1.0 与 2.0 的点积 = 64.0
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out_res[0], 1e-2);

    // 3. 验证融合量化矩阵 (Fused Q4Weights: initFromBlockSlices)
    var raw_block_a: [18]u8 = undefined;
    std.mem.writeInt(u16, raw_block_a[0..2], scale_bits, .little);
    @memset(raw_block_a[2..18], 0xAA); // weight = 2.0

    var raw_block_b: [18]u8 = undefined;
    std.mem.writeInt(u16, raw_block_b[0..2], scale_bits, .little);
    @memset(raw_block_b[2..18], 0xBB); // weight = 11 - 8 = 3.0

    var fused_q4 = try Q4Weights.initFromBlockSlices(
        std.testing.allocator,
        &.{ &raw_block_a, &raw_block_b },
        32,
        &.{ 1, 1 },
        stream,
    );
    defer fused_q4.deinit();

    var in_arr = try mlx.Array.fromSlice(&x_vec, &.{ 1, 32 });
    defer in_arr.deinit();

    var in_bf16 = try in_arr.astype(.bfloat16, stream);
    defer in_bf16.deinit();

    var norm_bf16 = try mlx.Array.fastRmsNorm(in_bf16, null, 1e-6, stream);
    defer norm_bf16.deinit();
    norm_bf16.eval();
    stream.synchronize();

    var fused_bf16_out = try fused_q4.forwardArray(in_bf16, stream);
    defer fused_bf16_out.deinit();
    fused_bf16_out.eval();
    stream.synchronize();

    var fused_out = try fused_q4.forwardArray(in_arr, stream);
    defer fused_out.deinit();

    var shape_buf: [8]usize = undefined;
    const fused_nd = try fused_out.shape(&shape_buf);
    try std.testing.expectEqual(@as(usize, 2), fused_nd);
    try std.testing.expectEqual(@as(usize, 1), shape_buf[0]);
    try std.testing.expectEqual(@as(usize, 2), shape_buf[1]);

    // 切片验证: sliceAxis 产生子张量并支持后续算子
    var slice_a = try fused_out.sliceAxis(1, 0, 1, stream);
    defer slice_a.deinit();
    var slice_b = try fused_out.sliceAxis(1, 1, 2, stream);
    defer slice_b.deinit();

    // 验证 gelu 与 multiply
    var act = try slice_a.gelu(.none, stream);
    defer act.deinit();
    var mul = try mlx.Array.multiply(act, slice_b, stream);
    defer mul.deinit();

    fused_out.eval();
    const f_data = try fused_out.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), f_data[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), f_data[1], 1e-2);
}

test "sampling argmaxWithPenalty dynamic vocab and negative logits" {
    const pad_only = [_]sampling.TokenRange{.single(0)};
    const logits = [_]f32{ 1.0, 2.0, 6.0, -0.5 };
    // 无重复惩罚时，token 2 (6.0) 最大
    try std.testing.expectEqual(@as(usize, 2), sampling.argmaxWithPenalty(&logits, 4, &pad_only, &.{}, 1.0));
    // 对 token 2 施加 10.0 惩罚 -> 6.0 / 10.0 = 0.6，此时 token 1 (2.0) 最大
    try std.testing.expectEqual(@as(usize, 1), sampling.argmaxWithPenalty(&logits, 4, &pad_only, &.{2}, 10.0));
    // 对正数与负数同时施加惩罚，验证负数乘以 penalty 的分支与越界 token 过滤
    const neg_logits = [_]f32{ -2.0, -1.0, -0.5, -3.0 };
    try std.testing.expectEqual(@as(usize, 1), sampling.argmaxWithPenalty(&neg_logits, 4, &pad_only, &.{ 2, 999999 }, 4.0));
    // 不排除任何词元时 token 0 同样可以被选中
    const first_max = [_]f32{ 9.0, 2.0, 6.0, -0.5 };
    try std.testing.expectEqual(@as(usize, 0), sampling.argmaxWithPenalty(&first_max, 4, &.{}, &.{}, 1.0));
    try std.testing.expectEqual(@as(usize, 2), sampling.argmaxWithPenalty(&first_max, 4, &pad_only, &.{}, 1.0));
}

test "sampling exclusion ranges and BF16 additive mask agree" {
    const excluded = [_]sampling.TokenRange{ .single(0), .{ .first = 5, .last = 7 }, .{ .first = 30, .last = 40 } };
    var mask: [10]u16 = undefined;
    sampling.fillExclusionMaskBf16(&mask, &excluded);
    for (mask, 0..) |bits, v| {
        const expected: u16 = if (sampling.isExcluded(&excluded, v)) sampling.bf16_neg_inf_bits else sampling.bf16_zero_bits;
        try std.testing.expectEqual(expected, bits);
    }
    // BF16 位模式 0xFF80 即单精度 -inf
    try std.testing.expect(std.math.isNegativeInf(@as(f32, @bitCast(@as(u32, sampling.bf16_neg_inf_bits) << 16))));
    try std.testing.expect(sampling.anyExcluded(&excluded, 7, 9));
    try std.testing.expect(!sampling.anyExcluded(&excluded, 8, 30));
    try std.testing.expect(sampling.anyExcluded(&excluded, 8, 31));
    try std.testing.expectEqual(@as(f32, 2.0), sampling.applyRepetitionPenalty(4.0, 2.0));
    try std.testing.expectEqual(@as(f32, -8.0), sampling.applyRepetitionPenalty(-4.0, 2.0));
    try std.testing.expect(!sampling.penaltyActive(1.0));
    try std.testing.expect(!sampling.penaltyActive(0.9));
}

test "LmHeadWeights rejects quantized modes the vocabulary shape cannot support" {
    // Q4 (分组 32) 与 Q8 (分组 64) 要求 hidden_size 整除分组长度，不满足时返回错误而不是静默退回 BF16
    try LmHeadWeights.validateMode(.q4, 128, 64);
    try LmHeadWeights.validateMode(.q8, 128, 3840);
    try LmHeadWeights.validateMode(.bf16, 128, 30);
    try std.testing.expectError(error.UnsupportedQuantMode, LmHeadWeights.validateMode(.q4, 128, 48));
    try std.testing.expectError(error.UnsupportedQuantMode, LmHeadWeights.validateMode(.q8, 128, 32));
    try std.testing.expectError(error.UnsupportedQuantMode, LmHeadWeights.validateMode(.q8, 128, 96));
    try std.testing.expectError(error.InvalidShape, LmHeadWeights.validateMode(.bf16, 0, 32));
    // 构建入口在任何平台上都先校验形状，再检查硬件加速可用性
    const embed: [(4 * 48)]u16 = @splat(0);
    try std.testing.expectError(error.UnsupportedQuantMode, LmHeadWeights.initWithMode(&embed, 4, 48, .q4, null));
}

test "MLX typed data accessors validate dtype and the default stream is created once" {
    if (!has_mlx) return;
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    var arr = try Array.fromSlice(&data, &.{3});
    defer arr.deinit();
    try std.testing.expectEqualSlices(f32, &data, try arr.dataFloat32());
    try std.testing.expectError(error.DTypeMismatch, arr.dataUint32());

    var as_bf16 = try arr.astype(.bfloat16, null);
    defer as_bf16.deinit();
    try std.testing.expectError(error.DTypeMismatch, as_bf16.dataFloat32());

    var idx = try arr.argmax(0, false, null);
    defer idx.deinit();
    try std.testing.expectEqual(@as(u32, 2), try idx.itemUint32());
    var pair = try Array.fromSliceU32(&.{ 1, 2 }, &.{2});
    defer pair.deinit();
    try std.testing.expectError(error.InvalidShape, pair.itemUint32());

    // 多次未指定命令流的算子调用复用同一个缓存的默认 GPU 命令流；释放后可再次惰性创建
    var sum = try Array.add(arr, arr, null);
    defer sum.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 4.0, 6.0 }, try sum.dataFloat32());
    mlx.releaseDefaultStream();
    var sum2 = try Array.add(sum, arr, null);
    defer sum2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 6.0, 9.0 }, try sum2.dataFloat32());
}

test "Q4Weights inside compiled closure" {
    if (!isAvailable()) return;
    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    var raw_block: [18]u8 = undefined;
    const scale_bits: u16 = 0x3C00; // 1.0 in f16
    std.mem.writeInt(u16, raw_block[0..2], scale_bits, .little);
    @memset(raw_block[2..18], 0x88); // weight = 8 - 8 = 0

    var q4_w = try Q4Weights.initFromBlocks(std.testing.allocator, &raw_block, 32, 1, stream);
    defer q4_w.deinit();

    const Payload = struct {
        w: *const Q4Weights,
        stream: ?Stream,
    };
    var payload = Payload{ .w = &q4_w, .stream = stream };

    const Helper = struct {
        fn compute(vres: [*c]mlx.c.mlx_vector_array, in: mlx.c.mlx_vector_array, pl: ?*anyopaque) callconv(.c) c_int {
            if (!mlx.has_mlx) return 1;
            const p: *const Payload = @ptrCast(@alignCast(pl.?));
            var x_handle: mlx.c.mlx_array = .{ .ctx = null };
            if (mlx.c.mlx_vector_array_get(&x_handle, in, 0) != 0 or x_handle.ctx == null) return 1;
            const x = mlx.Array{ .handle = x_handle };

            var res = p.w.forwardArray(x, p.stream) catch return 1;
            defer res.deinit();

            const res_vec = mlx.c.mlx_vector_array_new_data(&[_]mlx.c.mlx_array{res.handle}, 1);
            _ = mlx.c.mlx_vector_array_set(vres, res_vec);
            _ = mlx.c.mlx_vector_array_free(res_vec);
            return 0;
        }
    };

    var compiled = try mlx.compile(Helper.compute, &payload, false);
    defer compiled.deinit();

    var x_vec: [32]f32 = undefined;
    for (&x_vec) |*v| v.* = 1.0;
    var in_arr = try mlx.Array.fromSlice(&x_vec, &.{ 1, 32 });
    defer in_arr.deinit();

    var in_vec = mlx.VectorArray.fromArrays(&.{in_arr});
    defer in_vec.deinit();

    var out_vec = try compiled.apply(in_vec);
    defer out_vec.deinit();

    try std.testing.expectEqual(@as(usize, 1), out_vec.size());
    var out_arr = out_vec.get(0).?;
    defer out_arr.deinit();
    out_arr.eval();
    const data = try out_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 1), data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 1e-3);
}

test "Q4Weights initFromDirectBuffers fast path" {
    if (!isAvailable()) return;
    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    var raw_block: [18]u8 = undefined;
    const scale_bits: u16 = 0x3C00; // 1.0 in f16
    std.mem.writeInt(u16, raw_block[0..2], scale_bits, .little);
    @memset(raw_block[2..18], 0x88); // weight = 8 - 8 = 0

    var w_q_buf: [4]u32 = undefined;
    var scales_buf: [1]f16 = undefined;
    var biases_buf: [1]f16 = undefined;

    Q4Weights.packBlocksToDirectBuffers(&.{&raw_block}, &.{1}, 32, &w_q_buf, &scales_buf, &biases_buf);

    var q4_direct = try Q4Weights.initFromDirectBuffers(
        std.mem.sliceAsBytes(&w_q_buf),
        std.mem.sliceAsBytes(&scales_buf),
        std.mem.sliceAsBytes(&biases_buf),
        32,
        1,
        stream,
    );
    defer q4_direct.deinit();

    var x_vec: [32]f32 = undefined;
    for (&x_vec) |*v| v.* = 1.0;
    var in_arr = try mlx.Array.fromSlice(&x_vec, &.{ 1, 32 });
    defer in_arr.deinit();

    var out_arr = try q4_direct.forwardArray(in_arr, stream);
    defer out_arr.deinit();
    out_arr.eval();
    const data = try out_arr.dataFloat32();
    try std.testing.expectEqual(@as(usize, 1), data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 1e-3);
}

test "GpuKvCache rotating sliding-window wrap-around range calculation" {
    // 1. 局部滑动窗口层在未满窗口容量时按线性偏移写入并切片
    const r_prefill = try GpuKvCache.computeWriteRange(512, true, 512, 0, 12);
    try std.testing.expectEqual(@as(usize, 0), r_prefill.write_start);
    try std.testing.expectEqual(@as(usize, 12), r_prefill.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_prefill.slice_start);
    try std.testing.expectEqual(@as(usize, 12), r_prefill.slice_stop);
    try std.testing.expect(!r_prefill.use_full_buffer);

    // 2. 局部滑动窗口层恰好填满第 512 个词元槽位 (pos = 511, seq_len = 1) 时启用整块缓冲区零切片复用
    const r_exact = try GpuKvCache.computeWriteRange(512, true, 512, 511, 1);
    try std.testing.expectEqual(@as(usize, 511), r_exact.write_start);
    try std.testing.expectEqual(@as(usize, 512), r_exact.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_exact.slice_start);
    try std.testing.expectEqual(@as(usize, 512), r_exact.slice_stop);
    try std.testing.expect(r_exact.use_full_buffer);

    // 3. 局部滑动窗口层超出 512 窗口 (pos = 512, 513, 1025) 时环形回绕覆写最旧槽位并保持整块缓冲区零切片复用
    const r_wrap0 = try GpuKvCache.computeWriteRange(512, true, 512, 512, 1);
    try std.testing.expectEqual(@as(usize, 0), r_wrap0.write_start);
    try std.testing.expectEqual(@as(usize, 1), r_wrap0.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_wrap0.slice_start);
    try std.testing.expectEqual(@as(usize, 512), r_wrap0.slice_stop);
    try std.testing.expect(r_wrap0.use_full_buffer);

    const r_wrap1 = try GpuKvCache.computeWriteRange(512, true, 512, 1025, 1);
    try std.testing.expectEqual(@as(usize, 1), r_wrap1.write_start);
    try std.testing.expectEqual(@as(usize, 2), r_wrap1.write_stop);
    try std.testing.expect(r_wrap1.use_full_buffer);

    // 4. 全局全注意力层 (is_sliding = false) 保持线性追加增长
    const r_global = try GpuKvCache.computeWriteRange(768, false, 512, 512, 1);
    try std.testing.expectEqual(@as(usize, 512), r_global.write_start);
    try std.testing.expectEqual(@as(usize, 513), r_global.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_global.slice_start);
    try std.testing.expectEqual(@as(usize, 513), r_global.slice_stop);
    try std.testing.expect(!r_global.use_full_buffer);

    // 5. 滑动窗口层预填充长度超过窗口 (T = 600 > 512)：线性写入全部键，读取 [0, 600) 并须施加滑动窗口掩码
    const r_long = try GpuKvCache.computeWriteRange(512, true, 512, 0, 600);
    try std.testing.expectEqual(@as(usize, 0), r_long.write_start);
    try std.testing.expectEqual(@as(usize, 600), r_long.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_long.slice_start);
    try std.testing.expectEqual(@as(usize, 600), r_long.slice_stop);
    try std.testing.expect(!r_long.use_full_buffer);
    try std.testing.expect(GpuKvCache.visibleKeyRange(true, 512, 0, 600).needs_window_mask);

    // 6. 长预填充之后的单步解码 (容量已扩至 768，不再环形覆写)：恰好读取最近 512 个键，无需掩码
    const r_after = try GpuKvCache.computeWriteRange(768, true, 512, 600, 1);
    try std.testing.expectEqual(@as(usize, 600), r_after.write_start);
    try std.testing.expectEqual(@as(usize, 89), r_after.slice_start);
    try std.testing.expectEqual(@as(usize, 601), r_after.slice_stop);
    try std.testing.expect(!GpuKvCache.visibleKeyRange(true, 512, 600, 1).needs_window_mask);

    // 7. 线性布局下的多词元步 (pos = 600, T = 5)：保留首个查询窗口内的 511 个历史键 [89, 605) 并施加掩码
    const r_chunk = try GpuKvCache.computeWriteRange(768, true, 512, 600, 5);
    try std.testing.expectEqual(@as(usize, 89), r_chunk.slice_start);
    try std.testing.expectEqual(@as(usize, 605), r_chunk.slice_stop);
    try std.testing.expect(GpuKvCache.visibleKeyRange(true, 512, 600, 5).needs_window_mask);

    // 8. 容量恰为窗口：位置 512 处尚未环形覆写，多词元步仍可线性追加；一旦越过 512 (已环形覆写) 则拒绝多词元步
    const r_edge = try GpuKvCache.computeWriteRange(512, true, 512, 512, 4);
    try std.testing.expectEqual(@as(usize, 512), r_edge.write_start);
    try std.testing.expectEqual(@as(usize, 1), r_edge.slice_start);
    try std.testing.expectEqual(@as(usize, 516), r_edge.slice_stop);
    try std.testing.expectError(error.SlidingWindowCacheWrapped, GpuKvCache.computeWriteRange(512, true, 512, 513, 4));
    try std.testing.expectError(error.SlidingWindowCacheWrapped, GpuKvCache.computeWriteRange(512, true, 512, 1025, 2));
}

test "GpuKvCache visible key range with the attention mask gives every query exactly its sliding window" {
    // 对每个查询 p ∈ [pos, pos + T)，读取的键范围 [start, stop) 与所选掩码 (滑动窗口掩码或右下对齐因果掩码) 的交集
    // 必须恰为 {k : 0 <= p - k < window} (与 mlx-lm 及 CPU F32 预填充 / 解码路径的窗口语义一致)
    const windows = [_]usize{ 1, 3, 4, 7 };
    for (windows) |window| {
        for (0..12) |pos| {
            for (1..10) |seq_len| {
                const range = GpuKvCache.visibleKeyRange(true, window, pos, seq_len);
                try std.testing.expectEqual(pos + seq_len, range.stop);
                const tk = range.stop - range.start;
                for (0..seq_len) |i| {
                    const p = pos + i;
                    for (0..p + 1) |k| {
                        const expected = p - k < window;
                        const in_view = k >= range.start and k < range.stop;
                        // 右下对齐因果掩码：第 i 个查询可见视图中前 i + tk - seq_len + 1 个键
                        const causal_visible = in_view and (k - range.start) <= i + tk - seq_len;
                        const window_visible = in_view and (attention_mask.Mask{ .sliding = window }).isVisible(p, k);
                        const got = if (range.needs_window_mask) window_visible else causal_visible;
                        try std.testing.expectEqual(expected, got);
                    }
                }
            }
        }
    }

    // 全局注意力层：读取全部历史键，右下对齐因果掩码即精确
    const global = GpuKvCache.visibleKeyRange(false, 4, 10, 6);
    try std.testing.expectEqual(@as(usize, 0), global.start);
    try std.testing.expectEqual(@as(usize, 16), global.stop);
    try std.testing.expect(!global.needs_window_mask);
}

test "fillAdditiveMask with a sliding window marks exactly the keys inside each query window" {
    const tq = 4;
    const tk = 6;
    var mask: [tq * tk]f32 = undefined;
    // 查询绝对位置 5..8，键绝对位置 3..8，窗口 3
    attention_mask.fillAdditiveMask(&mask, tq, tk, 5, 3, .{ .sliding = 3 });
    const expected = [tq][tk]bool{
        .{ true, true, true, false, false, false }, // p = 5: k ∈ {3, 4, 5}
        .{ false, true, true, true, false, false }, // p = 6: k ∈ {4, 5, 6}
        .{ false, false, true, true, true, false }, // p = 7: k ∈ {5, 6, 7}
        .{ false, false, false, true, true, true }, // p = 8: k ∈ {6, 7, 8}
    };
    for (0..tq) |i| {
        for (0..tk) |j| {
            const want: f32 = if (expected[i][j]) 0.0 else -1e9;
            try std.testing.expectEqual(want, mask[i * tk + j]);
        }
    }
}

test "Executor parallelFor and parallelReduce with grain near maxInt(usize) do not overflow" {
    const Count = struct {
        hits: *std.atomic.Value(usize),
        fn body(ctx: @This(), begin: usize, end: usize) void {
            _ = ctx.hits.fetchAdd(end - begin, .monotonic);
        }
        fn map(ctx: @This(), begin: usize, end: usize) usize {
            _ = ctx;
            return end - begin;
        }
        fn add(a: usize, b: usize) usize {
            return a + b;
        }
    };

    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const execs = [_]Executor{ Executor.serial, .{ .io = threaded.io(), .max_tasks = 4 } };
    const max = std.math.maxInt(usize);
    for (execs) |exec| {
        for ([_]usize{ max, max - 1, max / 2 + 1 }) |grain| {
            var hits = std.atomic.Value(usize).init(0);
            try exec.parallelFor(10, grain, Count{ .hits = &hits }, Count.body);
            try std.testing.expectEqual(@as(usize, 10), hits.load(.monotonic));

            const total = try exec.parallelReduce(usize, 10, grain, Count{ .hits = &hits }, Count.map, Count.add, 0);
            try std.testing.expectEqual(@as(usize, 10), total);
        }
        // 区间长度本身接近上限时，最后一块的结束位置也不得回绕
        var big_hits = std.atomic.Value(usize).init(0);
        try exec.parallelFor(max, max / 2 + 1, Count{ .hits = &big_hits }, Count.body);
        try std.testing.expectEqual(max, big_hits.load(.monotonic));
    }
}

/// 测试用排除集合：0 号填充词元与跨越 64 词块边界的一段闭区间
const test_excluded = [_]sampling.TokenRange{ .single(0), .{ .first = 258_880, .last = 258_884 } };

/// 逐元素顺序扫描的贪婪采样参考实现：与 `argmaxWithPenalty` 的语义逐位对照
fn referenceArgmaxWithPenalty(logits: []const f32, vocab_size: usize, recent: []const usize, penalty: f32) usize {
    var best: f32 = -std.math.inf(f32);
    var best_id: usize = 0;
    for (0..vocab_size) |v| {
        if (sampling.isExcluded(&test_excluded, v)) continue;
        var val = logits[v];
        if (penalty > 1.0 and std.mem.indexOfScalar(usize, recent, v) != null) {
            val = if (val > 0.0) val / penalty else val * penalty;
        }
        if (val > best) {
            best = val;
            best_id = v;
        }
    }
    return best_id;
}

test "sampling argmaxWithPenalty handles vocabularies larger than one bitset window without heap allocation" {
    const allocator = std.testing.allocator;
    // 词表超过单个 262,144 词元位图窗口，旧实现在堆分配失败时会越界读取栈上位图
    const vocab: usize = 300_000;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    @memset(logits, -1.0);
    logits[290_000] = 8.0; // 第二个窗口内的最大值
    logits[1234] = 5.0;

    try std.testing.expectEqual(@as(usize, 290_000), sampling.argmaxWithPenalty(logits, vocab, &test_excluded, &.{}, 1.0));
    // 第二个窗口内的最近词元同样必须被惩罚：8 / 2 = 4 < 5
    try std.testing.expectEqual(@as(usize, 1234), sampling.argmaxWithPenalty(logits, vocab, &test_excluded, &.{ 290_000, 290_000, vocab + 7 }, 2.0));

    // 排除词元即使取得最大值也不会被选中
    logits[0] = 100.0;
    logits[258_882] = 50.0;
    try std.testing.expectEqual(@as(usize, 290_000), sampling.argmaxWithPenalty(logits, vocab, &test_excluded, &.{}, 1.0));
}

test "sampling argmaxWithPenalty matches sequential scan with ties, NaN and penalties" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    for ([_]usize{ 1, 63, 64, 65, 1000, 4097, 262_200 }) |vocab| {
        const logits = try allocator.alloc(f32, vocab);
        defer allocator.free(logits);
        for (0..5) |_| {
            // 量化到少量离散值以制造大量并列最大值，并混入 NaN 与 -Inf
            for (logits) |*l| {
                const r = rnd.uintLessThan(u32, 40);
                l.* = switch (r) {
                    0 => std.math.nan(f32),
                    1 => -std.math.inf(f32),
                    else => @as(f32, @floatFromInt(@as(i32, @intCast(r)) - 20)) * 0.5,
                };
            }
            var recent: [48]usize = undefined;
            for (&recent) |*t| t.* = rnd.uintLessThan(usize, vocab + 10);
            for ([_]f32{ 1.0, 1.3, 4.0 }) |penalty| {
                try std.testing.expectEqual(
                    referenceArgmaxWithPenalty(logits, vocab, &recent, penalty),
                    sampling.argmaxWithPenalty(logits, vocab, &test_excluded, &recent, penalty),
                );
            }
        }
    }
}

test "QuantLayout validates quantization parameters and computes buffer sizes" {
    const QuantLayout = offload.QuantLayout;
    // 4-bit / 分组 64：每个 uint32 打包 8 个权重
    const l4 = try QuantLayout.init(128, 3, 64, 4, .bfloat16);
    try std.testing.expectEqual([2]usize{ 3, 16 }, l4.w_dims);
    try std.testing.expectEqual([2]usize{ 3, 2 }, l4.sb_dims);
    try std.testing.expectEqual(@as(usize, 3 * 16 * 4), l4.w_q_bytes);
    try std.testing.expectEqual(@as(usize, 3 * 2 * 2), l4.sb_bytes);

    const l8 = try QuantLayout.init(64, 2, 32, 8, .float32);
    try std.testing.expectEqual([2]usize{ 2, 16 }, l8.w_dims);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), l8.sb_bytes);

    // 位宽不整除 32（旧实现在 @divExact 处触发未定义行为）、分组长度非法或输入维度不整除分组长度时返回错误
    try std.testing.expectError(error.InvalidShape, QuantLayout.init(96, 1, 32, 3, .float16));
    try std.testing.expectError(error.InvalidShape, QuantLayout.init(64, 1, 0, 4, .float16));
    try std.testing.expectError(error.InvalidShape, QuantLayout.init(96, 1, 64, 4, .float16));
    try std.testing.expectError(error.InvalidShape, QuantLayout.init(64, 1, 32, 4, .int32));
}

test "MLX copyToSlice handles strided slice views and VectorArray keeps more than 64 arrays" {
    if (!has_mlx) return;
    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    // [4, 8] 的列切片 [:, 2:4] 是非连续视图：元素总数为 8，但按步长寻址的跨度为 3 * 8 + 3 + 1 = 28
    var base_data: [32]f32 = undefined;
    for (&base_data, 0..) |*v, i| v.* = @floatFromInt(i);
    var base = try Array.fromSlice(&base_data, &.{ 4, 8 });
    defer base.deinit();
    var view = try base.sliceAxis(1, 2, 4, stream);
    defer view.deinit();
    var out: [8]f32 = undefined;
    try view.copyToSlice(&out);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 10, 11, 18, 19, 26, 27 }, &out);

    // 转置视图（最内层维度非连续）
    var t_view = try base.transpose(stream);
    defer t_view.deinit();
    var t_out: [32]f32 = undefined;
    try t_view.copyToSlice(&t_out);
    for (0..8) |r| {
        for (0..4) |col| try std.testing.expectEqual(base_data[col * 8 + r], t_out[r * 4 + col]);
    }

    // 脑浮点 16 (Brain Floating Point 16, BF16) 数组先转换为单精度再复制
    var bf = try base.astype(.bfloat16, stream);
    defer bf.deinit();
    var bf_out: [32]f32 = undefined;
    try bf.copyToSlice(&bf_out);
    try std.testing.expectEqualSlices(f32, &base_data, &bf_out);

    var many: [70]Array = undefined;
    for (&many) |*a| a.* = base;
    var vec = VectorArray.fromArrays(&many);
    defer vec.deinit();
    try std.testing.expectEqual(@as(usize, 70), vec.size());
}

// ============================================================================
// 随机采样 (Stochastic Sampling)：Top-K / Top-P / 温度 / 种子 / 投机采样校验
// ============================================================================

/// Hugging Face 语义的参考终结分布：温度 → 前 K 截断 (Top-K) → 核采样 (Nucleus Sampling, Top-P)，对完整词表排序计算
fn referenceProbabilities(allocator: std.mem.Allocator, logits: []const f32, opts: sampling.SamplingOptions) ![]f64 {
    const n = logits.len;
    const probs = try allocator.alloc(f64, n);
    @memset(probs, 0.0);
    const order = try allocator.alloc(sampling.Candidate, n);
    defer allocator.free(order);
    var m: usize = 0;
    for (logits, 0..) |l, v| {
        if (std.math.isNan(l) or l == -std.math.inf(f32)) continue;
        order[m] = .{ .id = @intCast(v), .logit = l };
        m += 1;
    }
    const sorted = order[0..m];
    std.mem.sort(sampling.Candidate, sorted, {}, struct {
        fn lt(_: void, a: sampling.Candidate, b: sampling.Candidate) bool {
            return a.before(b);
        }
    }.lt);
    const kept_k = if (opts.top_k > 0) sorted[0..@min(opts.top_k, m)] else sorted;
    const inv_t: f64 = 1.0 / @as(f64, opts.temperature);
    const top: f64 = kept_k[0].logit;
    var z: f64 = 0.0;
    for (kept_k) |cand| z += @exp((@as(f64, cand.logit) - top) * inv_t);
    var kept: usize = kept_k.len;
    if (opts.top_p < 1.0) {
        var cum: f64 = 0.0;
        kept = 0;
        while (kept < kept_k.len) : (kept += 1) {
            if (kept > 0 and cum >= @as(f64, opts.top_p) * z) break;
            cum += @exp((@as(f64, kept_k[kept].logit) - top) * inv_t);
        }
    }
    var mass: f64 = 0.0;
    for (kept_k[0..kept]) |cand| mass += @exp((@as(f64, cand.logit) - top) * inv_t);
    for (kept_k[0..kept]) |cand| probs[cand.id] = @exp((@as(f64, cand.logit) - top) * inv_t) / mass;
    return probs;
}

/// 皮尔逊卡方统计量 (Pearson Chi-Square Statistic)：只统计期望概率大于 0 的类别，期望为 0 的类别出现即视为失败
fn chiSquare(counts: []const usize, probs: []const f64, draws: usize) !f64 {
    var chi2: f64 = 0.0;
    for (counts, probs) |count, p| {
        if (p == 0.0) {
            try std.testing.expectEqual(@as(usize, 0), count);
            continue;
        }
        const expected = p * @as(f64, @floatFromInt(draws));
        const diff = @as(f64, @floatFromInt(count)) - expected;
        chi2 += diff * diff / expected;
    }
    return chi2;
}

test "sampling options default to full softmax sampling and detect greedy decoding" {
    const opts = sampling.SamplingOptions.defaultOptions();
    try std.testing.expectEqual(@as(f32, 1.0), opts.temperature);
    try std.testing.expectEqual(@as(usize, 0), opts.top_k);
    try std.testing.expectEqual(@as(f32, 1.0), opts.top_p);
    try std.testing.expectEqual(@as(f32, 1.0), opts.repetition_penalty);
    try std.testing.expect(!opts.isGreedy());
    try std.testing.expect(sampling.SamplingOptions.greedy.isPlainGreedy());
    try std.testing.expect((sampling.SamplingOptions{ .temperature = -1.0 }).isGreedy());
    try std.testing.expect((sampling.SamplingOptions{ .temperature = std.math.nan(f32) }).isGreedy());
    try std.testing.expect((sampling.SamplingOptions{ .top_k = 1 }).isGreedy());
    try std.testing.expect(!(sampling.SamplingOptions{ .temperature = 0.0, .repetition_penalty = 1.2 }).isPlainGreedy());

    try std.testing.expectEqual(@as(usize, 1), sampling.SamplingOptions.greedy.candidateCount(1000));
    try std.testing.expectEqual(@as(usize, 64), (sampling.SamplingOptions{ .top_k = 64 }).candidateCount(1000));
    try std.testing.expectEqual(@as(usize, 10), (sampling.SamplingOptions{ .top_k = 64 }).candidateCount(10));
    try std.testing.expectEqual(sampling.bounded_candidate_count, opts.candidateCount(262_144));
    try std.testing.expect((sampling.SamplingOptions{ .top_p = 0.9 }).needsLogNormalizer());
    try std.testing.expect(!(sampling.SamplingOptions{ .top_p = 0.9, .top_k = 40 }).needsLogNormalizer());
}

test "sampling temperature <= 0 and top_k == 1 reproduce greedy argmax" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const vocab: usize = 5000;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    for (0..4) |_| {
        for (logits) |*l| l.* = rnd.floatNorm(f32) * 3.0;
        logits[rnd.uintLessThan(usize, vocab)] = -std.math.inf(f32);
        logits[rnd.uintLessThan(usize, vocab)] = std.math.nan(f32);
        const expected = sampling.argmaxWithPenalty(logits, vocab, &.{}, &.{}, 1.0);
        for ([_]sampling.SamplingOptions{
            .{ .temperature = 0.0 },
            .{ .temperature = -2.0, .top_k = 50 },
            .{ .temperature = 1.5, .top_k = 1 },
            .{ .temperature = 0.7, .top_k = 1, .top_p = 0.5 },
        }) |opts| {
            var sampler = sampling.Sampler.init(opts);
            for (0..5) |_| {
                try std.testing.expectEqual(expected, try sampler.sampleLogits(Executor.serial, allocator, logits));
            }
            // 候选集终结步骤同样只取最优候选
            var cands = [_]sampling.Candidate{
                .{ .id = 3, .logit = 1.0 }, .{ .id = 9, .logit = 4.0 }, .{ .id = 1, .logit = 4.0 }, .{ .id = 2, .logit = -1.0 },
            };
            try std.testing.expectEqual(@as(u32, 1), sampler.sampleCandidates(&cands, null));
        }
    }
}

test "sampling tiny top_p always picks the most likely token" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{ 0.5, 2.0, 1.9, -1.0, 1.95, -std.math.inf(f32), 0.0, 1.0 };
    for ([_]usize{ 0, 3, 64 }) |top_k| {
        var sampler = sampling.Sampler.init(.{ .temperature = 1.0, .top_k = top_k, .top_p = 1e-6, .seed = top_k + 1 });
        for (0..300) |_| {
            try std.testing.expectEqual(@as(usize, 1), try sampler.sampleLogits(Executor.serial, allocator, &logits));
        }
    }
}

test "sampling top_k restricts the support to the k most likely tokens" {
    const allocator = std.testing.allocator;
    // 8 个词元概率接近，前 3 名为 2、5、7
    const logits = [_]f32{ 0.10, 0.20, 0.90, 0.30, 0.15, 0.85, 0.05, 0.80 };
    var sampler = sampling.Sampler.init(.{ .temperature = 1.0, .top_k = 3, .seed = 11 });
    var counts: [8]usize = @splat(0);
    for (0..3000) |_| counts[try sampler.sampleLogits(Executor.serial, allocator, &logits)] += 1;
    for (counts, 0..) |count, v| {
        if (v == 2 or v == 5 or v == 7) {
            try std.testing.expect(count > 700);
        } else {
            try std.testing.expectEqual(@as(usize, 0), count);
        }
    }
}

test "sampling frequencies approximate the processed softmax distribution for a fixed seed" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{ 1.0, 0.2, -0.5, 2.0, 0.0, -std.math.inf(f32), 1.4, -1.2, 0.7, 0.3 };
    const draws: usize = 40_000;
    for ([_]sampling.SamplingOptions{
        .{ .temperature = 1.0, .seed = 1 },
        .{ .temperature = 0.6, .seed = 2 },
        .{ .temperature = 1.3, .top_k = 4, .seed = 3 },
        .{ .temperature = 1.0, .top_p = 0.8, .seed = 4 },
        .{ .temperature = 0.9, .top_k = 6, .top_p = 0.9, .seed = 5 },
    }) |opts| {
        const probs = try referenceProbabilities(allocator, &logits, opts);
        defer allocator.free(probs);
        var sampler = sampling.Sampler.init(opts);
        var counts: [logits.len]usize = @splat(0);
        for (0..draws) |_| counts[try sampler.sampleLogits(Executor.serial, allocator, &logits)] += 1;
        // 自由度不超过 8：卡方统计量超过 35 的概率约 2e-5
        try std.testing.expect(try chiSquare(&counts, probs, draws) < 35.0);
    }
}

test "sampling applies the repetition penalty once per token before excluding tokens" {
    const allocator = std.testing.allocator;
    var logits = [_]f32{ 4.0, -2.0, 3.0, 2.0, 0.0, 6.0 };
    const recent = [_]usize{ 2, 1, 2, 2, 99, 4 };
    const excluded = [_]sampling.TokenRange{ .single(0), .single(5) };
    var reference = logits;
    try sampling.applyRepetitionPenaltyInPlace(allocator, &logits, &recent, 2.0);
    // 正值除以系数、负值乘以系数，重复出现的词元只惩罚一次，越界词元被忽略
    try std.testing.expectEqualSlices(f32, &.{ 4.0, -4.0, 1.5, 2.0, 0.0, 6.0 }, &logits);
    sampling.maskExcludedInPlace(&logits, &excluded);
    try std.testing.expect(logits[0] == -std.math.inf(f32) and logits[5] == -std.math.inf(f32));

    // 惩罚与排除之后的贪婪结果与融合实现 `argmaxWithPenalty` 一致 (词元 2 受罚后词元 3 胜出)
    var greedy = sampling.Sampler.init(.{ .temperature = 0.0 });
    const fused = sampling.argmaxWithPenalty(&reference, reference.len, &excluded, &recent, 2.0);
    try std.testing.expectEqual(@as(usize, 2), sampling.argmaxWithPenalty(&reference, reference.len, &excluded, &.{}, 2.0));
    try std.testing.expectEqual(fused, try greedy.sampleLogits(Executor.serial, allocator, &logits));
    try std.testing.expectEqual(@as(usize, 3), fused);

    // 系数不大于 1 时不施加惩罚
    try sampling.applyRepetitionPenaltyInPlace(allocator, &reference, &recent, 1.0);
    try std.testing.expectEqual(@as(f32, 3.0), reference[2]);
}

test "sampling with the same seed reproduces the same token sequence" {
    const allocator = std.testing.allocator;
    var logits: [300]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 2.0;
    const opts = sampling.SamplingOptions{ .temperature = 1.0, .top_k = 64, .top_p = 0.95, .seed = 1234 };
    var a = sampling.Sampler.init(opts);
    var b = sampling.Sampler.init(opts);
    var c_other = sampling.Sampler.init(.{ .temperature = 1.0, .top_k = 64, .top_p = 0.95, .seed = 4321 });
    var differs = false;
    for (0..200) |_| {
        const ta = try a.sampleLogits(Executor.serial, allocator, &logits);
        try std.testing.expectEqual(ta, try b.sampleLogits(Executor.serial, allocator, &logits));
        if (ta != try c_other.sampleLogits(Executor.serial, allocator, &logits)) differs = true;
    }
    try std.testing.expect(differs);
}

test "sampling selectTopK matches a full sort and does not depend on the thread count" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const parallel = Executor{ .io = threaded.io(), .max_tasks = 4 };

    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    const vocab: usize = 70_001;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    // 离散取值制造大量并列，混入 NaN 与 -inf
    for (logits) |*l| {
        const r = rnd.uintLessThan(u32, 200);
        l.* = switch (r) {
            0 => std.math.nan(f32),
            1 => -std.math.inf(f32),
            else => @as(f32, @floatFromInt(r)) * 0.25,
        };
    }
    const all = try allocator.alloc(sampling.Candidate, vocab);
    defer allocator.free(all);
    var usable: usize = 0;
    for (logits, 0..) |l, v| {
        if (std.math.isNan(l) or l == -std.math.inf(f32)) continue;
        all[usable] = .{ .id = @intCast(v), .logit = l };
        usable += 1;
    }
    std.mem.sort(sampling.Candidate, all[0..usable], {}, struct {
        fn lt(_: void, a: sampling.Candidate, b: sampling.Candidate) bool {
            return a.before(b);
        }
    }.lt);

    for ([_]usize{ 1, 64, 1000, 40_000 }) |k| {
        const buf_serial = try allocator.alloc(sampling.Candidate, k);
        defer allocator.free(buf_serial);
        const buf_parallel = try allocator.alloc(sampling.Candidate, k);
        defer allocator.free(buf_parallel);
        const serial_sel = try sampling.selectTopK(Executor.serial, allocator, logits, buf_serial);
        const parallel_sel = try sampling.selectTopK(parallel, allocator, logits, buf_parallel);
        try std.testing.expectEqual(k, serial_sel.len);
        _ = sampling.buildDistribution(serial_sel, .{ .temperature = 1.0 }, null);
        _ = sampling.buildDistribution(parallel_sel, .{ .temperature = 1.0 }, null);
        try std.testing.expectEqualSlices(sampling.Candidate, all[0..k], serial_sel);
        try std.testing.expectEqualSlices(sampling.Candidate, all[0..k], parallel_sel);
    }

    // 可采样词元少于 k 时只返回全部可采样词元
    const small = [_]f32{ 1.0, -std.math.inf(f32), 2.0 };
    var buf: [8]sampling.Candidate = undefined;
    try std.testing.expectEqual(@as(usize, 2), (try sampling.selectTopK(Executor.serial, allocator, &small, &buf)).len);
}

test "sampling nucleus partial selection over a large vocabulary matches the full-vocabulary reference" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const vocab: usize = 20_000;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    // 较平坦的分布：核采样阈值需要远多于首轮 256 个候选，触发候选数扩大
    for (logits) |*l| l.* = rnd.floatNorm(f32) * 0.5;
    const all = try allocator.alloc(sampling.Candidate, vocab);
    defer allocator.free(all);

    for ([_]f32{ 0.5, 0.95 }) |top_p| {
        const opts = sampling.SamplingOptions{ .temperature = 0.8, .top_p = top_p, .seed = 77 };
        var fast = sampling.Sampler.init(opts);
        var reference = sampling.Sampler.init(opts);
        for (0..50) |_| {
            for (all, logits, 0..) |*cand, l, v| cand.* = .{ .id = @intCast(v), .logit = l };
            const expected = reference.sampleCandidates(all, null);
            try std.testing.expectEqual(@as(usize, expected), try fast.sampleLogits(Executor.serial, allocator, logits));
        }
    }

    // 完整词表的对数配分函数与直接计算一致
    var direct: f64 = 0.0;
    for (logits) |l| direct += @exp(@as(f64, l) / 0.8);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(@log(direct))), try sampling.logSumExpScaled(Executor.serial, logits, 1.0 / 0.8), 1e-5);
}

test "sampling candidate finisher gives identical tokens for identical candidate sets in any order" {
    const base = [_]sampling.Candidate{
        .{ .id = 10, .logit = 2.0 },               .{ .id = 4, .logit = 1.5 },   .{ .id = 7, .logit = 1.5 },
        .{ .id = 1, .logit = 0.2 },                .{ .id = 30, .logit = -0.4 }, .{ .id = 2, .logit = 1.9 },
        .{ .id = 8, .logit = -std.math.inf(f32) }, .{ .id = 5, .logit = 0.9 },
    };
    const opts = sampling.SamplingOptions{ .temperature = 1.1, .top_k = 6, .top_p = 0.9, .seed = 2024 };
    var a = sampling.Sampler.init(opts);
    var b = sampling.Sampler.init(opts);
    var prng = std.Random.DefaultPrng.init(3);
    for (0..300) |_| {
        var x = base;
        var y = base;
        prng.random().shuffle(sampling.Candidate, &y);
        try std.testing.expectEqual(a.sampleCandidates(&x, null), b.sampleCandidates(&y, null));
    }
}

test "sampling speculative draft verification keeps the target distribution unbiased" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{ 1.2, 0.3, -0.7, 2.0, 0.1, -std.math.inf(f32) };
    const opts = sampling.SamplingOptions{ .temperature = 1.0, .top_k = 5, .top_p = 0.97 };
    const probs = try referenceProbabilities(allocator, &logits, opts);
    defer allocator.free(probs);
    const draws: usize = 40_000;

    // 草稿分别为高概率词元、低概率词元、被核采样截断的词元与排除词元
    for ([_]u32{ 3, 1, 2, 5 }, 0..) |draft, round| {
        var sampler = sampling.Sampler.init(.{ .temperature = 1.0, .top_k = 5, .top_p = 0.97, .seed = 100 + round });
        var counts: [logits.len]usize = @splat(0);
        var accepted: usize = 0;
        for (0..draws) |_| {
            var cands: [logits.len]sampling.Candidate = undefined;
            for (&cands, logits, 0..) |*cand, l, v| cand.* = .{ .id = @intCast(v), .logit = l };
            const dist = sampler.distribution(&cands, null);
            switch (sampler.verifyDraft(dist, draft)) {
                .accepted => {
                    accepted += 1;
                    counts[draft] += 1;
                },
                .rejected => |tok| {
                    try std.testing.expect(tok != draft);
                    counts[tok] += 1;
                },
            }
        }
        try std.testing.expect(try chiSquare(&counts, probs, draws) < 30.0);
        const accept_rate = @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(draws));
        try std.testing.expectApproxEqAbs(probs[draft], accept_rate, 0.015);
    }

    // 贪婪终结分布退化为精确匹配规则，且不消耗随机数
    var greedy = sampling.Sampler.init(.{ .temperature = 0.0, .seed = 9 });
    const before_state = greedy.prng.s;
    var cands: [logits.len]sampling.Candidate = undefined;
    for (&cands, logits, 0..) |*cand, l, v| cand.* = .{ .id = @intCast(v), .logit = l };
    const dist = greedy.distribution(&cands, null);
    try std.testing.expectEqual(sampling.DraftVerdict.accepted, greedy.verifyDraft(dist, 3));
    try std.testing.expectEqual(sampling.DraftVerdict{ .rejected = 3 }, greedy.verifyDraft(dist, 0));
    try std.testing.expectEqual(before_state, greedy.prng.s);
}

test "MLX candidate selection bindings: argpartition, take/put along axis, logsumexp and where" {
    if (!has_mlx) return;
    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    const data = [_]f32{ 0.5, 3.0, -1.0, 2.0, 2.5, -4.0 };
    var logits = try Array.fromSlice(&data, &.{ 1, 6 });
    defer logits.deinit();

    // 取负后部分排序：前 3 个下标即对数几率最大的 3 个词元 (集合 {1, 4, 3})
    var neg = try logits.negative(stream);
    defer neg.deinit();
    var part = try neg.argpartitionAxis(2, -1, stream);
    defer part.deinit();
    var top_view = try part.sliceAxis(1, 0, 3, stream);
    defer top_view.deinit();
    var top_ids = try top_view.contiguous(stream);
    defer top_ids.deinit();
    var top_vals = try logits.takeAlongAxis(top_ids, -1, stream);
    defer top_vals.deinit();
    var cands: [3]sampling.Candidate = undefined;
    for (try top_ids.dataUint32(), try top_vals.dataFloat32(), &cands) |id, v, *cand| cand.* = .{ .id = id, .logit = v };
    _ = sampling.buildDistribution(&cands, .{}, null);
    try std.testing.expectEqualSlices(sampling.Candidate, &.{
        .{ .id = 1, .logit = 3.0 }, .{ .id = 4, .logit = 2.5 }, .{ .id = 3, .logit = 2.0 },
    }, &cands);

    // 按下标写回重复惩罚后的值：正值除以 2，负值乘以 2
    const idx_data = [_]i32{ 1, 2 };
    var idx = try Array.fromSliceI32(&idx_data, &.{ 1, 2 });
    defer idx.deinit();
    var gathered = try logits.takeAlongAxis(idx, -1, stream);
    defer gathered.deinit();
    var zero = try Array.fromScalarF32(0.0);
    defer zero.deinit();
    var positive = try Array.greater(gathered, zero, stream);
    defer positive.deinit();
    var shrunk = try gathered.mulScalar(0.5, stream);
    defer shrunk.deinit();
    var grown = try gathered.mulScalar(2.0, stream);
    defer grown.deinit();
    var penalized = try Array.where(positive, shrunk, grown, stream);
    defer penalized.deinit();
    var updated = try logits.putAlongAxis(idx, penalized, -1, stream);
    defer updated.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1.5, -2.0, 2.0, 2.5, -4.0 }, try updated.dataFloat32());

    var lse = try logits.logsumexpAxis(-1, false, stream);
    defer lse.deinit();
    var direct: f32 = 0.0;
    for (data) |v| direct += @exp(v);
    try std.testing.expectApproxEqRel(@log(direct), (try lse.dataFloat32())[0], 1e-5);

    var shared = try logits.share();
    defer shared.deinit();
    try std.testing.expectEqualSlices(f32, &data, try shared.dataFloat32());
}

/// 合成注意力输入：第 p 个位置、第 h 个头、第 d 维的取值 (已按脑浮点数 (Brain Floating Point 16-bit, bfloat16) 舍入，
/// 使主机端参考实现与 GPU 读取完全相同的输入)
fn syntheticAttnValue(kind: f32, p: usize, h: usize, d: usize) f32 {
    const x = @sin(kind * 1.7 + @as(f32, @floatFromInt(p)) * 1.3 + @as(f32, @floatFromInt(h)) * 0.7 + @as(f32, @floatFromInt(d)) * 0.37);
    const bits: u32 = @as(u32, Array.f32ToBf16Bits(x)) << 16;
    return @bitCast(bits);
}

/// 构建本步 `[1, heads, seq_len, head_dim]` 的 bfloat16 输入张量 (绝对位置 pos..pos + seq_len)
fn syntheticAttnStep(allocator: std.mem.Allocator, kind: f32, heads: usize, pos: usize, seq_len: usize, head_dim: usize, stream: Stream) !Array {
    const buf = try allocator.alloc(f32, heads * seq_len * head_dim);
    defer allocator.free(buf);
    for (0..heads) |h| {
        for (0..seq_len) |t| {
            for (0..head_dim) |d| buf[(h * seq_len + t) * head_dim + d] = syntheticAttnValue(kind, pos + t, h, d);
        }
    }
    var f32_arr = try Array.fromSlice(buf, &.{ 1, heads, seq_len, head_dim });
    defer f32_arr.deinit();
    return f32_arr.astype(.bfloat16, stream);
}

test "GpuKvCache sliding-window steps match a per-query reference (prefill longer than the window, ring decode, chunked steps)" {
    if (!isAvailable()) return;
    const allocator = std.testing.allocator;
    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    const num_heads = 4;
    const num_kv = 2;
    const head_dim = 8;
    const window = 3;
    const groups = num_heads / num_kv;
    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    const q_kind: f32 = 0.0;
    const k_kind: f32 = 1.0;
    const v_kind: f32 = 2.0;

    // 每个场景为一串 (pos, seq_len) 步：长预填充 + 解码、短预填充 + 环形覆写解码、线性布局下的分块多词元步
    const Step = struct { pos: usize, len: usize };
    const scenarios = [_][]const Step{
        &.{ .{ .pos = 0, .len = 7 }, .{ .pos = 7, .len = 1 }, .{ .pos = 8, .len = 1 }, .{ .pos = 9, .len = 1 } },
        &.{ .{ .pos = 0, .len = 2 }, .{ .pos = 2, .len = 1 }, .{ .pos = 3, .len = 1 }, .{ .pos = 4, .len = 1 }, .{ .pos = 5, .len = 1 }, .{ .pos = 6, .len = 1 } },
        &.{ .{ .pos = 0, .len = 5 }, .{ .pos = 5, .len = 4 }, .{ .pos = 9, .len = 1 }, .{ .pos = 10, .len = 2 } },
    };

    for (scenarios) |steps| {
        var cache = try GpuKvCache.init(window, num_kv, head_dim, stream);
        defer cache.deinit();
        for (steps) |step| {
            var q = try syntheticAttnStep(allocator, q_kind, num_heads, step.pos, step.len, head_dim, stream);
            defer q.deinit();
            var k_new = try syntheticAttnStep(allocator, k_kind, num_kv, step.pos, step.len, head_dim, stream);
            defer k_new.deinit();
            var v_new = try syntheticAttnStep(allocator, v_kind, num_kv, step.pos, step.len, head_dim, stream);
            defer v_new.deinit();

            const view = try cache.writeStep(k_new, v_new, true, window, step.pos, step.len, stream);
            defer if (view.is_slice) {
                var vk = view.k;
                vk.deinit();
                var vv = view.v;
                vv.deinit();
            };
            var out = try offload.cachedAttention(allocator, q, view.k, view.v, scale, true, window, step.pos, step.len, stream);
            defer out.deinit();
            var out_f32 = try out.astype(.float32, stream);
            defer out_f32.deinit();
            out_f32.eval();
            const got = try out_f32.dataFloat32();
            try std.testing.expectEqual(@as(usize, num_heads * step.len * head_dim), got.len);

            // 逐查询参考：绝对位置 p 的查询只与 [max(0, p - window + 1), p] 内的键做 Softmax 加权
            for (0..num_heads) |h| {
                const kv_h = h / groups;
                for (0..step.len) |t| {
                    const p = step.pos + t;
                    const k_lo = if (p + 1 > window) p + 1 - window else 0;
                    var scores: [window]f32 = undefined;
                    var max_score: f32 = -std.math.inf(f32);
                    for (k_lo..p + 1, 0..) |kp, idx| {
                        var dot: f32 = 0.0;
                        for (0..head_dim) |d| dot += syntheticAttnValue(q_kind, p, h, d) * syntheticAttnValue(k_kind, kp, kv_h, d);
                        scores[idx] = dot * scale;
                        max_score = @max(max_score, scores[idx]);
                    }
                    const n_keys = p + 1 - k_lo;
                    var sum: f32 = 0.0;
                    for (scores[0..n_keys]) |*sc| {
                        sc.* = @exp(sc.* - max_score);
                        sum += sc.*;
                    }
                    for (0..head_dim) |d| {
                        var expected: f32 = 0.0;
                        for (k_lo..p + 1, 0..) |kp, idx| expected += scores[idx] / sum * syntheticAttnValue(v_kind, kp, kv_h, d);
                        try std.testing.expectApproxEqAbs(expected, got[(h * step.len + t) * head_dim + d], 3e-2);
                    }
                }
            }
        }
    }

    // 环形覆写之后的多词元步被拒绝 (缓存槽位已不按绝对位置排列)
    var ring = try GpuKvCache.init(window, num_kv, head_dim, stream);
    defer ring.deinit();
    for (0..window + 2) |p| {
        var k_new = try syntheticAttnStep(allocator, k_kind, num_kv, p, 1, head_dim, stream);
        defer k_new.deinit();
        var v_new = try syntheticAttnStep(allocator, v_kind, num_kv, p, 1, head_dim, stream);
        defer v_new.deinit();
        const view = try ring.writeStep(k_new, v_new, true, window, p, 1, stream);
        if (view.is_slice) {
            var vk = view.k;
            vk.deinit();
            var vv = view.v;
            vv.deinit();
        }
    }
    var k_chunk = try syntheticAttnStep(allocator, k_kind, num_kv, window + 2, 2, head_dim, stream);
    defer k_chunk.deinit();
    var v_chunk = try syntheticAttnStep(allocator, v_kind, num_kv, window + 2, 2, head_dim, stream);
    defer v_chunk.deinit();
    try std.testing.expectError(error.SlidingWindowCacheWrapped, ring.writeStep(k_chunk, v_chunk, true, window, window + 2, 2, stream));
}
