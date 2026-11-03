const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const vec_math = @import("../backend.zig").vec_math;
pub const tensor = tensor_mod;
const Tensor = tensor_mod.Tensor;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
const Op = types.Op;
const backward_core = @import("backward_core.zig");
const backward_nn = @import("backward_nn.zig");
const backward_math = @import("backward_math.zig");

// 算子前向重演与反向求导的分发：按 `Op.op_type` 调用对应的张量内核与反向内核

/// 重新执行算子的前向计算，根据最新输入更新输出张量的数据
pub fn forward(self: *Op, allocator: std.mem.Allocator) !void {
    switch (self.op_type) {
        .MatMul => {
            try tensor_mod.matmulInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .Q4MatMul => {
            const ctx = self.context.Q4MatMul;
            const x = self.inputs[0];
            const x_contig = if (x.isContiguous()) x else try x.contiguous(allocator);
            defer if (x_contig != x) x_contig.deinit(allocator);
            try tensor_mod.quant.q4MatMulInto(.serial, self.outputs[0], x_contig, ctx.blocks, ctx.in_features, ctx.out_features);
        },
        .Relu => {
            try tensor_mod.reluInto(self.outputs[0], self.inputs[0]);
        },
        .Gelu, .GeluTanh => {
            try tensor_mod.geluInto(self.outputs[0], self.inputs[0], self.op_type.geluApproximation().?);
        },
        .Sigmoid => {
            try tensor_mod.sigmoidInto(self.outputs[0], self.inputs[0]);
        },
        .Tanh => {
            try tensor_mod.tanhInto(self.outputs[0], self.inputs[0]);
        },
        .Softcap => {
            try tensor_mod.softcapInto(self.outputs[0], self.inputs[0], self.context.Softcap.cap);
        },
        .LeakyRelu => {
            try tensor_mod.leakyReluInto(self.outputs[0], self.inputs[0], self.context.LeakyRelu.alpha);
        },
        .Silu => {
            try tensor_mod.siluInto(self.outputs[0], self.inputs[0]);
        },
        .SoftmaxCrossEntropy => {
            const logits = self.inputs[0];
            const loss = self.outputs[0];
            const ctx = &self.context.SoftmaxCrossEntropy;
            const targets = ctx.targets;
            const probs = ctx.probs;

            const B_size = logits.shape.dims[0];
            const D = logits.shape.dims[1];
            var loss_sum: f32 = 0.0;
            var total_w: f32 = 0.0;

            for (0..B_size) |i| {
                const w: f32 = if (ctx.mask) |m| m[i] else 1.0;
                if (w <= 0.0) continue;

                const row = logits.data[i * D .. (i + 1) * D];
                const p_row = if (probs.len == B_size * D) probs[i * D .. (i + 1) * D] else null;
                const row_loss = try tensor_mod.losses.ceRow(row, targets[i], p_row, null, 0.0);
                loss_sum += row_loss * w;
                total_w += w;
            }

            ctx.total_weight = total_w;
            loss.data[0] = if (total_w > 0.0) loss_sum / total_w else 0.0;
        },
        .DpoLoss => {
            const pi_chosen = self.inputs[0];
            const pi_rejected = self.inputs[1];
            const loss = self.outputs[0];
            const ctx = self.context.DpoLoss;
            const N = pi_chosen.data.len;

            var total_loss: f32 = 0.0;
            for (0..N) |i| {
                total_loss += tensor_mod.losses.dpoTerm(
                    pi_chosen.data[i],
                    pi_rejected.data[i],
                    ctx.ref_chosen[i],
                    ctx.ref_rejected[i],
                    ctx.beta,
                    null,
                );
            }
            loss.data[0] = if (N > 0) total_loss / @as(f32, @floatFromInt(N)) else 0.0;
        },
        .GrpoLoss => {
            const old_logps = self.inputs[0];
            const new_logps = self.inputs[1];
            const loss = self.outputs[0];
            const ctx = self.context.GrpoLoss;
            const N = old_logps.data.len;

            var total_obj: f32 = 0.0;
            for (0..N) |i| {
                const ref = if (ctx.ref_logps) |refs| refs[i] else null;
                total_obj += tensor_mod.losses.grpoTerm(
                    new_logps.data[i],
                    old_logps.data[i],
                    ctx.advantages[i],
                    ref,
                    ctx.beta,
                    ctx.clip_eps,
                    null,
                );
            }
            loss.data[0] = if (N > 0) -(total_obj / @as(f32, @floatFromInt(N))) else 0.0;
        },
        .BceWithLogitsLoss => {
            try tensor_mod.bceWithLogitsLossInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .BceLoss => {
            try tensor_mod.bceLossInto(self.outputs[0], self.inputs[0], self.inputs[1], self.context.BceLoss.eps);
        },
        .Reshape => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (C.data.ptr != A.data.ptr) {
                if (!A.isContiguous()) {
                    tensor_mod.shape.gatherStrided(C.data, A.data, A.shape, A.strides);
                } else {
                    try tensor_mod.copyInto(C, A);
                }
            }
        },
        .Transpose => {
            const ctx = self.context.Transpose;
            try tensor_mod.transposeInto(self.outputs[0], self.inputs[0], ctx.dim0, ctx.dim1);
        },
        .Concat => {
            try tensor_mod.concatInto(self.outputs[0], self.inputs, self.context.Concat.dim, allocator);
        },
        .Split => {
            try tensor_mod.splitInto(self.outputs, self.inputs[0], self.context.Split.dim, allocator);
        },
        .RepeatKV => {
            try tensor_mod.repeatKVInto(self.outputs[0], self.inputs[0], self.context.RepeatKV.groups, allocator);
        },
        .MseLoss => {
            try tensor_mod.mseLossInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .MulScalar => {
            try tensor_mod.mulScalarInto(self.outputs[0], self.inputs[0], self.context.MulScalar.val);
        },
        .DivScalar => {
            try tensor_mod.divScalarInto(self.outputs[0], self.inputs[0], self.context.DivScalar.val);
        },
        .AddScalar => {
            try tensor_mod.addScalarInto(self.outputs[0], self.inputs[0], self.context.AddScalar.val);
        },
        .SubScalar => {
            try tensor_mod.subScalarInto(self.outputs[0], self.inputs[0], self.context.SubScalar.val);
        },
        .Add => {
            try tensor_mod.addInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .Sub => {
            try tensor_mod.subInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .Mul => {
            try tensor_mod.mulInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .Div => {
            try tensor_mod.divInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .Conv1D => {
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const ctx = self.context.Conv1D;
            try tensor_mod.conv1dInto(self.outputs[0], self.inputs[0], self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding, .groups = ctx.groups }, allocator);
        },
        .Conv2D => {
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const ctx = self.context.Conv2D;
            try tensor_mod.conv2dInto(self.outputs[0], self.inputs[0], self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding, .groups = ctx.groups }, allocator);
        },
        .ConvTranspose1D => {
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const ctx = self.context.ConvTranspose1D;
            try tensor_mod.convTranspose1dInto(self.outputs[0], self.inputs[0], self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding }, allocator);
        },
        .ConvTranspose2D => {
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const ctx = self.context.ConvTranspose2D;
            try tensor_mod.convTranspose2dInto(self.outputs[0], self.inputs[0], self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding }, allocator);
        },
        .MaxPool1D => {
            const ctx = self.context.MaxPool1D;
            try tensor_mod.maxpool1dInto(self.outputs[0], self.inputs[0], ctx.pool_size, .{ .stride = ctx.stride, .padding = ctx.padding });
        },
        .MaxPool2D => {
            const ctx = self.context.MaxPool2D;
            try tensor_mod.maxpool2dInto(self.outputs[0], self.inputs[0], ctx.pool_size, .{ .stride = ctx.stride, .padding = ctx.padding });
        },
        .AvgPool1D => {
            const ctx = self.context.AvgPool1D;
            try tensor_mod.avgpool1dInto(self.outputs[0], self.inputs[0], ctx.kernel_size, .{ .stride = ctx.stride, .padding = ctx.padding });
        },
        .AvgPool2D => {
            const ctx = self.context.AvgPool2D;
            try tensor_mod.avgpool2dInto(self.outputs[0], self.inputs[0], ctx.kernel_size, .{ .stride = ctx.stride, .padding = ctx.padding });
        },
        .AdaptiveAvgPool1D => {
            const ctx = self.context.AdaptiveAvgPool1D;
            try tensor_mod.adaptiveAvgPool1dInto(self.outputs[0], self.inputs[0], ctx.output_size);
        },
        .AdaptiveAvgPool2D => {
            const ctx = self.context.AdaptiveAvgPool2D;
            try tensor_mod.adaptiveAvgPool2dInto(self.outputs[0], self.inputs[0], ctx.output_size);
        },
        .Softmax => {
            try tensor_mod.softmaxInto(self.outputs[0], self.inputs[0], null);
        },
        .RmsNorm => {
            const ctx = self.context.RmsNorm;
            const weight: ?*Tensor = if (self.inputs.len > 1) self.inputs[1] else null;
            const inv_rms: ?[]f32 = if (ctx.inv_rms.len > 0) ctx.inv_rms else null;
            try tensor_mod.rmsNormInto(self.outputs[0], self.inputs[0], weight, ctx.options, inv_rms);
        },
        .LayerNorm => {
            const ctx = self.context.LayerNorm;
            const save_mean: ?[]f32 = if (ctx.save_mean.len > 0) ctx.save_mean else null;
            const save_inv_std: ?[]f32 = if (ctx.save_inv_std.len > 0) ctx.save_inv_std else null;
            try tensor_mod.layerNormInto(self.outputs[0], self.inputs[0], self.inputs[1], self.inputs[2], ctx.eps, save_mean, save_inv_std);
        },
        .BatchNorm1d => {
            // 训练模式重演必须重新计算当前批次的统计量 (并更新运行统计量)，否则反向传播会使用过期的批统计量
            const ctx = self.context.BatchNorm1d;
            try tensor_mod.batchNorm1dInto(
                self.outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.inputs[2],
                ctx.running_mean,
                ctx.running_var,
                ctx.training,
                ctx.eps,
                ctx.momentum,
                ctx.save_mean,
                ctx.save_inv_std,
                allocator,
            );
        },
        .BatchNorm2d => {
            // 训练模式重演必须重新计算当前批次的统计量 (并更新运行统计量)，否则反向传播会使用过期的批统计量
            const ctx = self.context.BatchNorm2d;
            try tensor_mod.batchNorm2dInto(
                self.outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.inputs[2],
                ctx.running_mean,
                ctx.running_var,
                ctx.training,
                ctx.eps,
                ctx.momentum,
                ctx.save_mean,
                ctx.save_inv_std,
                allocator,
            );
        },
        .GroupNorm => {
            const ctx = self.context.GroupNorm;
            try tensor_mod.groupNormInto(
                self.outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.inputs[2],
                ctx.num_groups,
                ctx.eps,
                ctx.save_mean,
                ctx.save_inv_std,
                allocator,
            );
        },
        .Dropout => {
            try tensor_mod.applyDropoutMaskInto(self.outputs[0], self.inputs[0], self.context.Dropout.mask_scale, allocator);
        },
        .RoPE => {
            const ctx = self.context.RoPE;
            try tensor_mod.ropeInto(self.outputs[0], self.inputs[0], ctx.start_pos, ctx.options, allocator);
        },
        .BatchMatMul => {
            try tensor_mod.batchMatMulInto(self.outputs[0], self.inputs[0], self.inputs[1], allocator);
        },
        .Attention => {
            const ctx = self.context.Attention;
            const probs: ?[]f32 = if (ctx.probs.len > 0) ctx.probs else null;
            try tensor_mod.attention_kernels.attentionInto(self.inputs[0], self.inputs[1], self.inputs[2], self.outputs[0], probs, ctx.options, allocator);
        },
        .Embedding => {
            try tensor_mod.embeddingInto(self.outputs[0], self.inputs[0], self.inputs[1]);
        },
        .TopK => {
            try self.inputs[0].topKInto(self.context.TopK.k, self.outputs[0], self.outputs[1]);
        },
        .GatherMatMul => {
            const ctx = self.context.GatherMatMul;
            const E = ctx.num_experts;
            const weights = self.inputs[2 .. 2 + E];
            const biases: ?[]const *Tensor = if (ctx.has_bias) self.inputs[2 + E .. 2 + 2 * E] else null;
            try self.inputs[0].gatherMatMulInto(self.inputs[1], weights, biases, self.outputs[0], allocator);
        },
        .L2Loss => {
            try tensor_mod.l2LossInto(self.outputs[0], self.inputs[0], self.context.L2Loss.lambda);
        },
        .L1Loss => {
            try tensor_mod.l1LossInto(self.outputs[0], self.inputs[0], self.context.L1Loss.lambda);
        },
        .Sqrt => {
            try tensor_mod.sqrtInto(self.outputs[0], self.inputs[0]);
        },
        .Exp => {
            try tensor_mod.expInto(self.outputs[0], self.inputs[0]);
        },
        .Log => {
            try tensor_mod.logInto(self.outputs[0], self.inputs[0]);
        },
        .Abs => {
            try tensor_mod.absInto(self.outputs[0], self.inputs[0]);
        },
        .Sum => {
            const ctx = self.context.Sum;
            try tensor_mod.sumInto(self.outputs[0], self.inputs[0], ctx.axis, ctx.keepdims, allocator);
        },
        .Mean => {
            const ctx = self.context.Mean;
            try tensor_mod.meanInto(self.outputs[0], self.inputs[0], ctx.axis, ctx.keepdims, allocator);
        },
        .Where => {
            const X = self.inputs[0];
            const Y = self.inputs[1];
            const C = self.outputs[0];
            const mask = self.context.Where.mask;
            const x_strides = tensor_mod.computeBroadcastStrides(X.shape, X.strides, C.shape);
            const y_strides = tensor_mod.computeBroadcastStrides(Y.shape, Y.strides, C.shape);
            const rank = C.shape.len;
            var indices: [8]usize = @splat(0);
            for (0..C.data.len) |c_flat| {
                var x_flat: usize = 0;
                var y_flat: usize = 0;
                for (0..rank) |d| {
                    x_flat += indices[d] * x_strides.dims[d];
                    y_flat += indices[d] * y_strides.dims[d];
                }
                C.data[c_flat] = if (mask[c_flat]) X.data[x_flat] else Y.data[y_flat];
                C.shape.incrementCoord(&indices);
            }
        },
        .MaskedFill => {
            const X = self.inputs[0];
            const C = self.outputs[0];
            const ctx = self.context.MaskedFill;
            var coord: [8]usize = @splat(0);
            const len = X.shape.len;
            for (0..C.data.len) |i| {
                if (ctx.mask[i]) {
                    C.data[i] = ctx.value;
                } else {
                    var src_idx: usize = 0;
                    for (0..len) |d| src_idx += coord[d] * X.strides.dims[d];
                    C.data[i] = X.data[src_idx];
                }
                X.shape.incrementCoord(&coord);
            }
        },
        .Slice => {
            const X = self.inputs[0];
            const C = self.outputs[0];
            const ctx = self.context.Slice;
            var coord: [8]usize = @splat(0);
            for (0..C.data.len) |dest_i| {
                var src_idx: usize = ctx.offset;
                for (0..ctx.rank) |d| src_idx += coord[d] * ctx.strides[d];
                C.data[dest_i] = X.data[src_idx];
                C.shape.incrementCoord(&coord);
            }
        },
        .Gather => {
            const ctx = self.context.Gather;
            try tensor_mod.gatherInto(self.outputs[0], self.inputs[0], ctx.dim, self.inputs[1]);
        },
        .Scatter => {
            const ctx = self.context.Scatter;
            try tensor_mod.scatterInto(self.outputs[0], self.inputs[0], ctx.dim, self.inputs[1], self.inputs[2]);
        },
        .ScatterAdd => {
            const ctx = self.context.ScatterAdd;
            if (self.outputs[0].isContiguous() and self.inputs[0].isContiguous()) {
                @memcpy(self.outputs[0].data, self.inputs[0].data);
            } else {
                tensor_mod.shape.gatherStrided(self.outputs[0].data, self.inputs[0].data, self.inputs[0].shape, self.inputs[0].strides);
            }
            try tensor_mod.scatterAddInto(self.outputs[0], ctx.dim, self.inputs[1], self.inputs[2]);
        },
        .IndexSelect => {
            const ctx = self.context.IndexSelect;
            try tensor_mod.indexSelectInto(self.outputs[0], self.inputs[0], ctx.dim, self.inputs[1]);
        },
    }
}

/// 执行算子的反向传播计算，更新其输入节点的梯度
pub fn backward(self: *Op, scratch_allocator: std.mem.Allocator) !void {
    switch (self.op_type) {
        .MatMul,
        .Relu,
        .Gelu,
        .GeluTanh,
        .Sigmoid,
        .Tanh,
        .Softcap,
        .LeakyRelu,
        .Silu,
        .SoftmaxCrossEntropy,
        .DpoLoss,
        .GrpoLoss,
        .BceWithLogitsLoss,
        .BceLoss,
        .Reshape,
        .Transpose,
        .Concat,
        .Split,
        .RepeatKV,
        .MseLoss,
        .MulScalar,
        .DivScalar,
        .AddScalar,
        .SubScalar,
        .Add,
        .Sub,
        .Mul,
        .Div,
        => try backward_core.backwardCore(self, scratch_allocator),

        .Conv1D,
        .Conv2D,
        .ConvTranspose1D,
        .ConvTranspose2D,
        .MaxPool1D,
        .MaxPool2D,
        .AvgPool1D,
        .AvgPool2D,
        .AdaptiveAvgPool1D,
        .AdaptiveAvgPool2D,
        .Softmax,
        .RmsNorm,
        .LayerNorm,
        .BatchNorm1d,
        .BatchNorm2d,
        .GroupNorm,
        .Dropout,
        .RoPE,
        .BatchMatMul,
        .Attention,
        .Embedding,
        .TopK,
        .GatherMatMul,
        .Q4MatMul,
        => try backward_nn.backwardNN(self, scratch_allocator),

        .L2Loss,
        .L1Loss,
        .Sqrt,
        .Exp,
        .Log,
        .Abs,
        .Sum,
        .Mean,
        .Where,
        .MaskedFill,
        .Slice,
        .Gather,
        .Scatter,
        .ScatterAdd,
        .IndexSelect,
        => try backward_math.backwardMath(self, scratch_allocator),
    }
}
