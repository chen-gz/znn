const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Graph = @import("graph.zig").Graph;

/// 从计算图中自动推导指定模块或节点的数学公式
pub fn inferModuleFormula(self: *const Graph, module_path: []const u8) []const u8 {
    // 1. 如果有显式指定的公式，直接返回
    if (self.getModuleFormula(module_path)) |form| {
        return form;
    }

    // 2. 检查计算图中的具体算子 (Ops) 是否有输出精确匹配该节点名称
    // （必须先于模块前缀继承，避免像 gpt.layers.0.attn.act_Add_20 被继承为 Attention 的全局公式）
    for (self.ops.items) |op| {
        if (op.outputs.len > 0) {
            if (op.outputs[0].name) |out_name| {
                if (std.mem.eql(u8, out_name, module_path)) {
                    return op.op_type.getFormula();
                }
            }
        }
    }

    // 3. 匹配子路径的最长前缀 (例如 "gpt.layers.0.output" -> 优先继承更长的 "gpt.layers.0" 而非根路径 "gpt")
    var best_prefix_match: ?[]const u8 = null;
    var best_prefix_len: usize = 0;
    var it = self.module_formulas.iterator();
    while (it.next()) |entry| {
        const k_len = entry.key_ptr.len;
        if (module_path.len > k_len and
            std.mem.startsWith(u8, module_path, entry.key_ptr.*) and
            module_path[k_len] == '.')
        {
            if (k_len > best_prefix_len) {
                best_prefix_len = k_len;
                best_prefix_match = entry.value_ptr.*;
            }
        }
    }
    if (best_prefix_match) |form| {
        return form;
    }

    // 4. 根据模块注册的原生类型推导标准公式 (避免脆弱的字符串模式推测)
    if (self.getModuleType(module_path)) |m_type| {
        if (std.mem.eql(u8, m_type, "Linear")) return "y = x W^T + b";
        if (std.mem.eql(u8, m_type, "Conv2D")) return "y = \\text{Conv2D}(x; W, b)";
        if (std.mem.eql(u8, m_type, "ConvTranspose2D")) return "y = \\text{ConvTranspose2D}(x; W, b)";
        if (std.mem.eql(u8, m_type, "RMSNorm")) return "y = \\text{RMSNorm}(x; \\gamma, \\epsilon)";
        if (std.mem.eql(u8, m_type, "LayerNorm")) return "y = \\text{LayerNorm}(x; \\gamma, \\beta)";
        if (std.mem.eql(u8, m_type, "BatchNorm2d")) return "y = \\text{BatchNorm2d}(x; \\gamma, \\beta)";
        if (std.mem.eql(u8, m_type, "Embedding")) return "y = \\text{Embedding}(x; W)";
        if (std.mem.eql(u8, m_type, "MLP")) return "y = \\text{GELU}(x W_{fc}^T + b_{fc}) W_{proj}^T + b_{proj}";
        if (std.mem.eql(u8, m_type, "SwiGLU")) return "y = (\\text{SiLU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";
        if (std.mem.eql(u8, m_type, "CausalSelfAttention")) return "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\cdot W_o^T + b_o";
        if (std.mem.eql(u8, m_type, "ScaledDotProductAttention")) return "\\text{AttentionCore}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V";
        if (std.mem.eql(u8, m_type, "TransformerBlock")) return "h_l = x_l + \\text{Attention}(\\text{RMSNorm}(x_l)), \\quad x_{l+1} = \\text{TransformerBlock}(x_l) = h_l + \\text{MLP}(\\text{RMSNorm}(h_l))";
        if (std.mem.eql(u8, m_type, "TransformerDecoder")) return "x_L = \\text{DecoderStack}(x_0) = (\\text{Block}_L \\circ \\dots \\circ \\text{Block}_1)(x_0)";
        if (std.mem.eql(u8, m_type, "GPT")) return "\\text{logits} = \\text{GPT}(\\text{TokenIDs}; \\theta) \\rightarrow [B, T, V]";
        if (std.mem.eql(u8, m_type, "RNNCell")) return "h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})";
        if (std.mem.eql(u8, m_type, "RNN")) return "h_{1:T} = \\text{RNN}(x_{1:T}, h_0)";
        if (std.mem.eql(u8, m_type, "LSTMCell")) return "c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, \\quad h_t = o_t \\odot \\tanh(c_t)";
        if (std.mem.eql(u8, m_type, "LSTM")) return "(h_{1:T}, c_{1:T}) = \\text{LSTM}(x_{1:T}, h_0, c_0)";
        if (std.mem.eql(u8, m_type, "StackedLSTM")) return "h^{(L)}_{1:T} = \\text{StackedLSTM}(x_{1:T})";
        if (std.mem.eql(u8, m_type, "GRUCell")) return "h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}))";
        if (std.mem.eql(u8, m_type, "GRU")) return "h_{1:T} = \\text{GRU}(x_{1:T}, h_0)";
        if (std.mem.eql(u8, m_type, "MoELayer")) return "y = \\sum_{i \\in \\text{TopK}(g(x))} p_i(x) E_i(x) + \\sum_{j} E^{\\text{shared}}_j(x)";
        if (std.mem.eql(u8, m_type, "MLALayer")) return "c_t^{KV} = x_t W^{DKV}, \\quad y = \\text{MLA}(Q, c^{KV}, k^R) W^O";
        if (std.mem.eql(u8, m_type, "LoRALinear")) return "y = x W_0 + \\frac{\\alpha}{r} (x A) B + b";
    }

    return "y = f(x; \\theta)";
}

/// 在图建立完毕后，智能探查参数节点的下游消费者算子并自动初始化权重
pub fn initWeights(self: *Graph, random: std.Random) void {
    const init_mod = @import("../nn/init.zig");

    // 搜集图内所有 Ops 的输入参数节点与 registered tensors
    const allocator = self.arena.allocator();
    var visited = std.AutoHashMap(*Tensor, void).init(allocator);
    defer visited.deinit();

    // 1. 扫描图中的 Ops 所有输入
    for (self.ops.items) |op| {
        for (op.inputs) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            initSingleTensor(self, t, random, init_mod);
        }
    }

    // 2. 扫描显式注册的 tensors
    for (self.tensors.items) |t| {
        if (visited.contains(t)) continue;
        visited.put(t, {}) catch continue;
        initSingleTensor(self, t, random, init_mod);
    }
}

pub fn isNormScaleParam(self: *const Graph, t: *const Tensor) bool {
    for (self.ops.items) |op| {
        switch (op.op_type) {
            .RmsNorm, .LayerNorm, .BatchNorm2d => {
                if (op.inputs.len >= 2 and op.inputs[1] == t) return true;
            },
            else => {},
        }
    }
    return false;
}

pub fn isEmbeddingParam(self: *const Graph, t: *const Tensor) bool {
    for (self.ops.items) |op| {
        if (op.op_type == .Embedding and op.inputs.len >= 1 and op.inputs[0] == t) return true;
    }
    return false;
}

pub fn isLoRABParam(self: *const Graph, t: *const Tensor) bool {
    if (t.name) |n| {
        if (std.mem.endsWith(u8, n, ".lora_b") or std.mem.eql(u8, n, "lora_b")) return true;
    }
    for (self.ops.items) |op| {
        if (op.op_type == .MatMul and op.inputs.len >= 2 and op.inputs[1] == t) {
            if (self.getModuleType(op.scope)) |m_type| {
                if (std.mem.eql(u8, m_type, "LoRALinear")) {
                    if (op.inputs[0].creator) |prev_op| {
                        if (prev_op.op_type == .MatMul and std.mem.eql(u8, prev_op.scope, op.scope)) return true;
                    }
                }
            }
        }
    }
    return false;
}

pub const AutoGraphParamInfo = struct {
    act_name: []const u8,
    strategy: []const u8,
};

pub fn describeAutoGraphParam(self: *Graph, t: *Tensor, strat_buf: *[64]u8) AutoGraphParamInfo {
    const init_mod = @import("../nn/init.zig");

    if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
        if (isNormScaleParam(self, t)) {
            return .{ .act_name = "scale", .strategy = "ones (1.0)" };
        }
        if (t.init_constant) |value| {
            if (value == 1.0) return .{ .act_name = "bias", .strategy = "ones (1.0)" };
            const strat = std.fmt.bufPrint(strat_buf, "constant ({d})", .{value}) catch "constant";
            return .{ .act_name = "bias", .strategy = strat };
        }
        return .{ .act_name = "bias", .strategy = "zeros (0.0)" };
    }

    if (isEmbeddingParam(self, t)) {
        return .{ .act_name = "Embedding", .strategy = "Normal (mean=0.0, std=0.02)" };
    }

    if (isLoRABParam(self, t)) {
        return .{ .act_name = "LoRA-B", .strategy = "zeros (0.0)" };
    }

    const act = detectConsumerActivation(self, t);
    const gain = init_mod.calculateGain(act);
    const act_name = switch (act) {
        .relu => "ReLU",
        .tanh => "Tanh",
        .sigmoid => "Sigmoid",
        .gelu => "GELU",
        .silu => "SiLU",
        .selu => "SELU",
        .leaky_relu => "LeakyReLU",
        .linear => "Linear (None)",
    };

    const strat = switch (act) {
        .tanh, .sigmoid => std.fmt.bufPrint(strat_buf, "Xavier Normal (gain={d:.3})", .{gain}) catch "Xavier Normal",
        .selu => "LeCun Normal",
        else => std.fmt.bufPrint(strat_buf, "He Normal (gain={d:.3})", .{gain}) catch "He Normal",
    };

    return .{ .act_name = act_name, .strategy = strat };
}

pub fn initSingleTensor(self: *Graph, t: *Tensor, random: std.Random, comptime init_mod: type) void {
    // 只对属于可训练参数（requires_grad=true 且非中间激活运算生成）的节点进行初始化
    // 如果已经被库外用户代码的 customInit 初始化过，则坚决跳过，绝不覆盖！
    if (!t.requires_grad or t.is_custom_initialized or t.creator != null) return;

    // 1. 如果是 1D 参数向量 (带结构性常量的参数按常量填充，如 LSTM 遗忘门偏置 1.0；
    //    归一化缩放因子 gamma 初始化为 1.0；其余偏置向量初始化为 0.0)
    if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
        if (t.init_constant) |value| {
            @memset(t.data, value);
            return;
        }
        if (isNormScaleParam(self, t)) {
            @memset(t.data, 1.0);
            return;
        }
        @memset(t.data, 0.0);
        return;
    }

    // 2. 如果是词嵌入表权重 (Embedding: Normal(0, 0.02)) 或 LoRA 旁路 B 矩阵 (全 0 初始化)
    if (isEmbeddingParam(self, t)) {
        init_mod.initWeights(random, t.data, t.shape.dims[0], t.shape.dims[1], .{ .normal = .{ .mean = 0.0, .std = 0.02 } });
        return;
    }
    if (isLoRABParam(self, t)) {
        @memset(t.data, 0.0);
        return;
    }

    // 3. 如果是高维权重矩阵 (Linear, Conv2D, ConvTranspose2D 等)
    // 沿计算图中的 Ops 向后探测第一个下游消费算子 (Consumer Op)
    const nonlinearity = detectConsumerActivation(self, t);
    const gain = init_mod.calculateGain(nonlinearity);

    var fan_in: usize = 1;
    var fan_out: usize = 1;
    if (t.shape.len == 2) {
        fan_in = t.shape.dims[0];
        fan_out = t.shape.dims[1];
    } else if (t.shape.len == 4) {
        // Conv2D: [out_channels, in_channels, kh, kw]
        const out_c = t.shape.dims[0];
        const in_c = t.shape.dims[1];
        const kh = t.shape.dims[2];
        const kw = t.shape.dims[3];
        fan_in = in_c * kh * kw;
        fan_out = out_c * kh * kw;
    } else {
        for (t.shape.dims[0 .. t.shape.len - 1]) |d| fan_in *= d;
        fan_out = t.shape.dims[t.shape.len - 1];
    }

    const method: init_mod.InitMethod = switch (nonlinearity) {
        .tanh, .sigmoid => .{ .xavier_normal = .{ .gain = gain } },
        .selu => .lecun_normal,
        else => .{ .he_normal = .{ .gain = gain } },
    };

    init_mod.initWeights(random, t.data, fan_in, fan_out, method);
}

/// 格式化全图各节点（包括输入数据、模型参数及中间算子输出）的详情与初始化报告为分配的字符串
pub fn formatInitReport(self: *Graph, allocator: std.mem.Allocator) ![]const u8 {
    const init_mod = @import("../nn/init.zig");
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "\n=== Graph Architecture & Initialization Report ===\n");
    try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
        "Node", "Kind", "Shape", "Status", "Inferred / Op", "Strategy / Details",
    });
    try buf.print(allocator, "{s:-<24} {s:-<12} {s:-<18} {s:-<14} {s:-<16} {s:-<28}\n", .{
        "", "", "", "", "", "",
    });

    const arena_alloc = self.arena.allocator();
    var visited = std.AutoHashMap(*Tensor, void).init(arena_alloc);
    defer visited.deinit();

    var param_idx: usize = 0;
    var input_idx: usize = 0;
    var op_idx: usize = 0;

    for (self.ops.items) |op| {
        for (op.inputs) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            try appendSingleTensorReport(self, t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
        }
        for (op.outputs) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            try appendSingleTensorReport(self, t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
        }
    }

    for (self.tensors.items) |t| {
        if (visited.contains(t)) continue;
        visited.put(t, {}) catch continue;
        try appendSingleTensorReport(self, t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
    }
    try buf.appendSlice(allocator, "=========================================================================================================\n\n");
    return buf.toOwnedSlice(allocator);
}

/// 在标准输出/调试控制台直接打印初始化详情报告
pub fn printInitReport(self: *Graph) void {
    const report = formatInitReport(self, self.backing_allocator) catch return;
    defer self.backing_allocator.free(report);
    std.debug.print("{s}", .{report});
}

/// 将计算图与模块层级结构序列化为递归的 JavaScript 对象表示法 (JavaScript Object Notation, JSON) 数据字符串 (供前端直接解析并构建完整模型拓扑)
pub fn formatJson(self: *Graph, allocator: std.mem.Allocator) ![]const u8 {
    const vis = @import("../nn/visualization.zig");
    return vis.graph_ir.generateJson(self, allocator);
}

/// 将计算图与模块层级结构直接导出保存为独立的 JavaScript 对象表示法 (JavaScript Object Notation, JSON) 文件 (如 "model_graph.json")
pub fn exportJson(self: *Graph, file_path: []const u8) !void {
    const vis = @import("../nn/visualization.zig");
    try vis.graph_ir.exportJson(self, file_path, self.backing_allocator);
}

pub fn appendSingleTensorReport(
    self: *Graph,
    t: *Tensor,
    param_idx: *usize,
    input_idx: *usize,
    op_idx: *usize,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    comptime init_mod: type,
) !void {
    _ = init_mod;
    var shape_buf: [64]u8 = undefined;
    var shape_len: usize = 0;
    shape_buf[0] = '[';
    shape_len += 1;
    for (0..t.shape.len) |d| {
        if (d > 0) {
            shape_buf[shape_len] = ',';
            shape_buf[shape_len + 1] = ' ';
            shape_len += 2;
        }
        const part = std.fmt.bufPrint(shape_buf[shape_len..], "{d}", .{t.shape.dims[d]}) catch "";
        shape_len += part.len;
    }
    shape_buf[shape_len] = ']';
    shape_len += 1;
    const shape_str = shape_buf[0..shape_len];

    // 1. 算子生成的中间计算节点 / 激活输出
    if (t.creator) |creator_op| {
        var name_buf: [48]u8 = undefined;
        const op_name = @tagName(creator_op.op_type);
        const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Node_{s}_{d}", .{ op_name, op_idx.* }) catch "Node_Op");
        op_idx.* += 1;

        var detail_buf: [64]u8 = undefined;
        const detail = std.fmt.bufPrint(&detail_buf, "produced by {s}", .{op_name}) catch "op output";

        try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
            name, "Activation", shape_str, "OP_OUTPUT", op_name, detail,
        });
        return;
    }

    // 2. 外部输入或常量张量 (非可学习参数)
    if (!t.requires_grad) {
        var name_buf: [48]u8 = undefined;
        const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Input_{d}", .{input_idx.*}) catch "Input");
        input_idx.* += 1;

        try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
            name, "Input", shape_str, "INPUT", "N/A", "user input / constant",
        });
        return;
    }

    // 3. 模型可学习参数节点
    var name_buf: [48]u8 = undefined;
    const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Param_{d}", .{param_idx.*}) catch "Param");
    param_idx.* += 1;

    if (t.is_custom_initialized) {
        try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
            name, "Param", shape_str, "CUSTOM_INIT", "N/A", "user-defined customInit",
        });
        return;
    }

    var strat_buf: [64]u8 = undefined;
    const info = describeAutoGraphParam(self, t, &strat_buf);

    try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
        name, "Param", shape_str, "AUTO_GRAPH", info.act_name, info.strategy,
    });
}

/// 顺着张量 t 往后在图的 Ops 列表中探查下游消费者的激活函数类型
pub fn detectConsumerActivation(self: *Graph, target: *Tensor) @import("../nn/init.zig").Nonlinearity {
    var current: *Tensor = target;
    var passed_projection = false;

    // 顺着当前层的线性/卷积投影、偏置加法与形状变换向后搜寻紧邻的激活函数
    var steps: usize = 0;
    while (steps < 16) : (steps += 1) {
        var advanced = false;
        for (self.ops.items) |op| {
            for (op.inputs) |inp| {
                if (inp == current) {
                    switch (op.op_type) {
                        .Relu => return .relu,
                        .Tanh => return .tanh,
                        .Sigmoid => return .sigmoid,
                        .Gelu => return .gelu,
                        .Silu => return .silu,
                        .LeakyRelu => return .{ .leaky_relu = op.context.LeakyRelu.alpha },
                        .MatMul, .Conv2D, .ConvTranspose2D => {
                            if (!passed_projection and op.outputs.len > 0) {
                                passed_projection = true;
                                current = op.outputs[0];
                                advanced = true;
                                break;
                            }
                        },
                        .AddBias, .Add, .Reshape, .Transpose => {
                            if (op.outputs.len > 0) {
                                current = op.outputs[0];
                                advanced = true;
                                break;
                            }
                        },
                        else => {},
                    }
                }
            }
            if (advanced) break;
        }
        if (!advanced) break;
    }

    // 如果下游没有接激活函数或直接进入输出/损失层，判定为 linear (gain=1.0)
    return .linear;
}
