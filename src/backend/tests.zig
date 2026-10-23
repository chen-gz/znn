//! Backend 模块单元测试套件 (Backend Unit Tests)
//!
//! 统一验证 Apple Silicon Metal GPU 硬件加速计算后端 (`mlx.zig`) 与模型算子卸载基础设施 (`offload.zig`)。

const std = @import("std");
const mlx = @import("mlx.zig");
const offload = @import("offload.zig");
const sampling = @import("sampling.zig");
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
    var sdpa_res = try Array.fastScaledDotProductAttention(q, q, q, 1.0 / @sqrt(8.0), false, stream);
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

    var gqa_sdpa = try Array.fastScaledDotProductAttention(q_dec, k_dec, v_dec, 1.0 / @sqrt(8.0), false, stream);
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
    var pref_sdpa = try Array.fastScaledDotProductAttention(q_pref_arr, k_pref_arr, v_pref_arr, 1.0 / @sqrt(8.0), true, stream);
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
    var sdpa_sub = try Array.fastScaledDotProductAttention(q_single, k_sub, v_sub, 1.0, false, stream);
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

    var y_gelu = try x_gelu.gelu(stream);
    defer y_gelu.deinit();

    Array.evalMultiple(&.{y_gelu});
    const gelu_out = try y_gelu.dataFloat32();
    try std.testing.expectEqual(@as(usize, 4), gelu_out.len);
    // GELU(0) = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu_out[0], 1e-4);
    // GELU(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) = 0.8413447
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu_out[1], 1e-4);
    // GELU(-1) = 0.5 * (-1) * (1 + erf(-1/sqrt(2))) = -0.158655
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), gelu_out[2], 1e-4);

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
    var attn = try Array.fastScaledDotProductAttention(q_probe, view_2, view_2, 1.0, false, stream);
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

    var attn2 = try Array.fastScaledDotProductAttention(q_probe, view_3, view_3, 1.0, false, stream);
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

    var geglu_res = try geglu_arr.geglu(2, stream);
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

            var gelu_res = x.gelu(null) catch return 1;
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

    var gelu_bf16 = try x_bf16.gelu(stream);
    defer gelu_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, gelu_bf16.dtype());

    var compiled_gm = try Array.compileGeluMul();
    defer compiled_gm.deinit();
    var fused_gm_bf16 = try Array.fusedGeluMul(x_bf16, scaled_bf16, compiled_gm, stream);
    defer fused_gm_bf16.deinit();
    try std.testing.expectEqual(DType.bfloat16, fused_gm_bf16.dtype());
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
    var act = try slice_a.gelu(stream);
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
    const r_prefill = GpuKvCache.computeWriteRange(512, true, 512, 0, 12);
    try std.testing.expectEqual(@as(usize, 0), r_prefill.write_start);
    try std.testing.expectEqual(@as(usize, 12), r_prefill.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_prefill.slice_start);
    try std.testing.expectEqual(@as(usize, 12), r_prefill.slice_stop);
    try std.testing.expect(!r_prefill.use_full_buffer);

    // 2. 局部滑动窗口层恰好填满第 512 个词元槽位 (pos = 511, seq_len = 1) 时启用整块缓冲区零切片复用
    const r_exact = GpuKvCache.computeWriteRange(512, true, 512, 511, 1);
    try std.testing.expectEqual(@as(usize, 511), r_exact.write_start);
    try std.testing.expectEqual(@as(usize, 512), r_exact.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_exact.slice_start);
    try std.testing.expectEqual(@as(usize, 512), r_exact.slice_stop);
    try std.testing.expect(r_exact.use_full_buffer);

    // 3. 局部滑动窗口层超出 512 窗口 (pos = 512, 513, 1025) 时环形回绕覆写最旧槽位并保持整块缓冲区零切片复用
    const r_wrap0 = GpuKvCache.computeWriteRange(512, true, 512, 512, 1);
    try std.testing.expectEqual(@as(usize, 0), r_wrap0.write_start);
    try std.testing.expectEqual(@as(usize, 1), r_wrap0.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_wrap0.slice_start);
    try std.testing.expectEqual(@as(usize, 512), r_wrap0.slice_stop);
    try std.testing.expect(r_wrap0.use_full_buffer);

    const r_wrap1 = GpuKvCache.computeWriteRange(512, true, 512, 1025, 1);
    try std.testing.expectEqual(@as(usize, 1), r_wrap1.write_start);
    try std.testing.expectEqual(@as(usize, 2), r_wrap1.write_stop);
    try std.testing.expect(r_wrap1.use_full_buffer);

    // 4. 全局全注意力层 (is_sliding = false) 保持线性追加增长
    const r_global = GpuKvCache.computeWriteRange(768, false, 512, 512, 1);
    try std.testing.expectEqual(@as(usize, 512), r_global.write_start);
    try std.testing.expectEqual(@as(usize, 513), r_global.write_stop);
    try std.testing.expectEqual(@as(usize, 0), r_global.slice_start);
    try std.testing.expectEqual(@as(usize, 513), r_global.slice_stop);
    try std.testing.expect(!r_global.use_full_buffer);
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
