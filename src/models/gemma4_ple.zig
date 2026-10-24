//! Gemma 4 逐层输入增强模块 (Per-Layer Embeddings, PLE Modules)
//!
//! 遵循 PyTorch / JAX 纯 Module 设计范式：
//! - 每个组件均为纯粹的 struct，只持有子模块与参数张量，前向计算签名统一为 `forward(self, graph, ...) !*Tensor`；
//! - 内部完全解耦底层硬件细节（零硬件专有数组依赖），计算全部交由计算图 `Graph` 调度。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const RMSNorm = nn.RMSNorm;
const deinitModel = nn.core.deinitModel;
const enterModuleScope = nn.core.enterModuleScope;

/// 单个解码层的 PLE 门控上投影模块 (Gemma4PleLayer)
/// 负责在注意力与门控 MLP 后，将逐层嵌入切片 (ple_slice) 调制并投影注入主干隐状态
pub fn Gemma4PleLayer(comptime Proj: type) type {
    return struct {
        const Self = @This();

        ple_gate: Proj,
        ple_proj: Proj,
        ple_norm: RMSNorm,
        hidden_activation: backend.GeluApproximation = .tanh,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4PleLayer",

        pub fn init(
            allocator: std.mem.Allocator,
            hidden_size: usize,
            ple_dim: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
        ) !Self {
            var ple_gate = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, ple_dim, .{ .bias = false })
            else
                try Proj.init(allocator, hidden_size, ple_dim);
            errdefer if (@hasDecl(Proj, "deinit")) ple_gate.deinit(allocator) else deinitModel(&ple_gate, allocator);

            var ple_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, ple_dim, hidden_size, .{ .bias = false })
            else
                try Proj.init(allocator, ple_dim, hidden_size);
            errdefer if (@hasDecl(Proj, "deinit")) ple_proj.deinit(allocator) else deinitModel(&ple_proj, allocator);

            const ple_norm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&ple_norm, allocator);

            return Self{
                .ple_gate = ple_gate,
                .ple_proj = ple_proj,
                .ple_norm = ple_norm,
                .hidden_activation = hidden_activation,
            };
        }

        pub fn initEmpty(
            hidden_size: usize,
            ple_dim: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
        ) Self {
            const ple_gate = if (@hasDecl(Proj, "initEmpty"))
                Proj.initEmpty(hidden_size, ple_dim)
            else
                undefined;
            const ple_proj = if (@hasDecl(Proj, "initEmpty"))
                Proj.initEmpty(ple_dim, hidden_size)
            else
                undefined;
            const ple_norm = RMSNorm.initUnscaled(.{ .eps = rms_norm_eps });

            return Self{
                .ple_gate = ple_gate,
                .ple_proj = ple_proj,
                .ple_norm = ple_norm,
                .hidden_activation = hidden_activation,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (@hasDecl(Proj, "deinit")) self.ple_gate.deinit(allocator) else deinitModel(&self.ple_gate, allocator);
            if (@hasDecl(Proj, "deinit")) self.ple_proj.deinit(allocator) else deinitModel(&self.ple_proj, allocator);
            deinitModel(&self.ple_norm, allocator);
        }

        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            if (@hasDecl(Proj, "enableOffload")) {
                try self.ple_gate.enableOffload(allocator, null);
                try self.ple_proj.enableOffload(allocator, null);
            }
        }

        /// 前向传播：计算调制后的逐层特征并与主隐状态残差累加
        /// x: [..., hidden_size]
        /// ple_slice: [..., ple_dim]
        /// layer_scalar: [1]
        pub fn forward(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            ple_slice: *Tensor,
            layer_scalar: *Tensor,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            // 1. 门控投影: [..., hidden_size] -> [..., ple_dim]
            const gate_raw = try self.ple_gate.forward(graph, x);
            // 2. GELU 激活函数
            const gate_act = try graph.gelu(gate_raw, self.hidden_activation);
            // 3. 与逐层切片进行逐元素点乘调制
            const modulated = try graph.mul(gate_act, ple_slice);
            // 4. 上投影映射回隐藏维度: [..., ple_dim] -> [..., hidden_size]
            const projected = try self.ple_proj.forward(graph, modulated);
            // 5. RMSNorm 归一化
            const normed = try self.ple_norm.forward(graph, projected);
            // 6. 残差与层缩放: (x + normed) * layer_scalar
            const sum = try graph.add(x, normed);
            return try graph.mulScalar(sum, layer_scalar.data[0]);
        }
    };
}

/// 全局逐层嵌入预计算模块 (Gemma4PleEmbedding)
/// 负责在解码首步根据 Prompt 词元和基础嵌入生成全部 42 层的逐层注入张量 [1, T, num_layers, ple_dim]
pub fn Gemma4PleEmbedding(comptime Proj: type) type {
    return struct {
        const Self = @This();

        model_proj: Proj,
        proj_norm: RMSNorm,
        num_layers: usize,
        ple_dim: usize,
        embed_scale: f32,
        proj_scale: f32,
        input_scale: f32,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4PleEmbedding",

        pub fn init(
            allocator: std.mem.Allocator,
            hidden_size: usize,
            num_layers: usize,
            ple_dim: usize,
            rms_norm_eps: f32,
        ) !Self {
            const ple_total = num_layers * ple_dim;
            var model_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, ple_total, .{ .bias = false })
            else
                try Proj.init(allocator, hidden_size, ple_total);
            errdefer if (@hasDecl(Proj, "deinit")) model_proj.deinit(allocator) else deinitModel(&model_proj, allocator);

            const proj_norm = try RMSNorm.init(allocator, ple_dim, rms_norm_eps);
            errdefer deinitModel(&proj_norm, allocator);

            const embed_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
            const proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_size)));
            const input_scale = 1.0 / @sqrt(2.0);

            return Self{
                .model_proj = model_proj,
                .proj_norm = proj_norm,
                .num_layers = num_layers,
                .ple_dim = ple_dim,
                .embed_scale = embed_scale,
                .proj_scale = proj_scale,
                .input_scale = input_scale,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (@hasDecl(Proj, "deinit")) self.model_proj.deinit(allocator) else deinitModel(&self.model_proj, allocator);
            deinitModel(&self.proj_norm, allocator);
        }

        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            if (@hasDecl(Proj, "enableOffload")) {
                try self.model_proj.enableOffload(allocator, null);
            }
        }

        /// 前向传播：融合词元逐层嵌入与骨干特征投影
        /// x_cur: [1, T, hidden_size]
        /// ple_tokens: [1, T, num_layers, ple_dim]
        /// 输出: [1, T, num_layers, ple_dim]
        pub fn forward(
            self: *const Self,
            graph: *autodiff.Graph,
            x_cur: *Tensor,
            ple_tokens: *Tensor,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const T = x_cur.shape.dims[1];

            // 1. 骨干投影并缩放
            const ple_proj_raw = try self.model_proj.forward(graph, x_cur);
            const ple_proj_scaled = try graph.mulScalar(ple_proj_raw, self.proj_scale);

            // 2. 重塑为 4D 张量并归一化 [1, T, num_layers, ple_dim]
            const ple_proj_4d = try graph.reshape(ple_proj_scaled, &.{ 1, T, self.num_layers, self.ple_dim });
            const ple_proj_normed = try self.proj_norm.forward(graph, ple_proj_4d);

            // 3. 逐层词元嵌入相加并施加输入缩放: (normed + tokens) * input_scale
            const ple_sum = try graph.add(ple_proj_normed, ple_tokens);
            return try graph.mulScalar(ple_sum, self.input_scale);
        }
    };
}
