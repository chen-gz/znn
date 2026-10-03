const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const deinitModel = core.deinitModel;
const enterModuleScope = core.enterModuleScope;
const Tensor = tensor.Tensor;
const Linear = core.Linear;

// ============================================================================
// 循环神经网络模块 (Recurrent Neural Network, RNN Modules)
// ============================================================================

/// 未传入初始状态时创建的全零状态 `[batch, hidden]`。
/// 它由模块内部生成，因此标记为常量缓冲区并命名为 `{module}.{state}` (如 `lstm.h_0`)，
/// 在模型图中不会被当作模型输入。
fn zeroState(
    graph: *autodiff.Graph,
    module_name: ?[]const u8,
    batch_size: usize,
    hidden_dim: usize,
    comptime state_fmt: []const u8,
    state_args: anytype,
) !*Tensor {
    const state = try graph.zeros(&.{ batch_size, hidden_dim }, false);
    state.is_buffer = true;
    if (module_name) |n| state.setNameFormatted("{s}." ++ state_fmt, .{n} ++ state_args);
    return state;
}

/// 单步经典 Elman 循环神经网络单元 (Recurrent Neural Network Cell, RNNCell)
/// 隐状态更新公式：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
pub const RNNCell = struct {
    input_dim: usize,
    hidden_dim: usize,
    weight_ih: Linear,
    weight_hh: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "RNNCell",

    pub const formula = "h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !RNNCell {
        const weight_ih = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&weight_ih, allocator);
        const weight_hh = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&weight_hh, allocator);

        return RNNCell{
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
        };
    }

    /// 单时间步前向：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    /// x: [batch_size, input_dim], h_prev: [batch_size, hidden_dim]
    pub fn forward(
        self: *const RNNCell,
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const x_proj = try self.weight_ih.forward(graph, x);
        const h_proj = try self.weight_hh.forward(graph, h_prev);

        const sum = try graph.add(x_proj, h_proj);

        return try graph.tanh(sum);
    }
};

/// Elman 循环神经网络 (Recurrent Neural Network, RNN) 序列前向传播返回结果
pub const RNNResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
};

/// 沿时间展开的 Elman 循环神经网络 (Recurrent Neural Network, RNN) 序列容器
pub const RNN = struct {
    cell: RNNCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    module_type: []const u8 = "RNN",

    pub const formula = "h_{1:T} = \\text{RNN}(x_{1:T}, h_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !RNN {
        const cell = try RNNCell.init(allocator, input_dim, hidden_dim);
        return RNN{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    /// 序列时序展开前向传播
    /// inputs: 长度为 seq_len 的 Tensor 切片，每个形状为 [batch_size, input_dim]
    /// h_0: 初始隐状态 [batch_size, hidden_dim]，若为 null 则自动置零
    pub fn forward(
        self: *const RNN,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !RNNResult {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try graph.arenaAllocator().alloc(*Tensor, seq_len);

        var h_curr: *Tensor = undefined;
        if (h_0) |h| {
            h_curr = h;
        } else {
            const batch_size = inputs[0].shape.dims[0];
            h_curr = try zeroState(graph, self.name, batch_size, self.hidden_dim, "h_0", .{});
        }

        for (inputs, 0..) |x_t, t| {
            const h_next = try self.cell.forward(graph, x_t, h_curr);
            outputs[t] = h_next;
            h_curr = h_next;
        }

        return .{
            .outputs = outputs,
            .h_n = h_curr,
        };
    }
};

/// 长短期记忆网络 (Long Short-Term Memory, LSTM) 单元内部状态对 (h_t, c_t)
pub const LSTMState = struct {
    h: *Tensor,
    c: *Tensor,
};

/// 单步长短期记忆网络单元 (Long Short-Term Memory Cell, LSTMCell)
/// 包含遗忘门 (f)、输入门 (i)、候选状态 (c_cand)、输出门 (o)
pub const LSTMCell = struct {
    input_dim: usize,
    hidden_dim: usize,
    // 遗忘门 (Forget Gate)
    w_ih_f: Linear,
    w_hh_f: Linear,
    // 输入门 (Input Gate)
    w_ih_i: Linear,
    w_hh_i: Linear,
    // 候选细胞状态 (Candidate Cell State)
    w_ih_c: Linear,
    w_hh_c: Linear,
    // 输出门 (Output Gate)
    w_ih_o: Linear,
    w_hh_o: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "LSTMCell",

    pub const formula = "c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, \\quad h_t = o_t \\odot \\tanh(c_t)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !LSTMCell {
        const w_ih_f = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_f, allocator);
        const w_hh_f = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_f, allocator);

        const w_ih_i = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_i, allocator);
        const w_hh_i = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_i, allocator);

        const w_ih_c = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_c, allocator);
        const w_hh_c = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_c, allocator);

        const w_ih_o = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_o, allocator);
        const w_hh_o = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_o, allocator);

        // 关键技巧：将遗忘门偏置初始化为 +1.0，促进长程梯度回传 (Gers et al., 2000)
        // 同时记录为结构性常量，使依据计算图的自动初始化保留该取值
        @memset(w_ih_f.bias.data, 1.0);
        w_ih_f.bias.init_constant = 1.0;

        return LSTMCell{
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .w_ih_f = w_ih_f,
            .w_hh_f = w_hh_f,
            .w_ih_i = w_ih_i,
            .w_hh_i = w_hh_i,
            .w_ih_c = w_ih_c,
            .w_hh_c = w_hh_c,
            .w_ih_o = w_ih_o,
            .w_hh_o = w_hh_o,
        };
    }

    /// 库内标准参数重初始化：8 个门控线性层按 options 初始化，遗忘门偏置重置为 +1.0
    pub fn resetParameters(self: *LSTMCell, random: std.Random, options: core.InitOptions) void {
        inline for (.{ "w_ih_f", "w_hh_f", "w_ih_i", "w_hh_i", "w_ih_c", "w_hh_c", "w_ih_o", "w_hh_o" }) |field_name| {
            @field(self, field_name).resetParameters(random, options);
        }
        @memset(self.w_ih_f.bias.data, 1.0);
    }

    /// 单步前向计算
    /// x: [batch_size, input_dim]
    /// h_prev: [batch_size, hidden_dim]
    /// c_prev: [batch_size, hidden_dim]
    pub fn forward(
        self: *const LSTMCell,
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
        c_prev: *Tensor,
    ) !LSTMState {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        // 1. 遗忘门: f_t = sigmoid(W_f * x + U_f * h_prev)
        const f_x = try self.w_ih_f.forward(graph, x);
        const f_h = try self.w_hh_f.forward(graph, h_prev);
        const f_sum = try graph.add(f_x, f_h);
        const f_t = try graph.sigmoid(f_sum);

        // 2. 输入门: i_t = sigmoid(W_i * x + U_i * h_prev)
        const i_x = try self.w_ih_i.forward(graph, x);
        const i_h = try self.w_hh_i.forward(graph, h_prev);
        const i_sum = try graph.add(i_x, i_h);
        const i_t = try graph.sigmoid(i_sum);

        // 3. 候选状态: c_cand = tanh(W_c * x + U_c * h_prev)
        const c_x = try self.w_ih_c.forward(graph, x);
        const c_h = try self.w_hh_c.forward(graph, h_prev);
        const c_sum = try graph.add(c_x, c_h);
        const c_cand = try graph.tanh(c_sum);

        // 4. 输出门: o_t = sigmoid(W_o * x + U_o * h_prev)
        const o_x = try self.w_ih_o.forward(graph, x);
        const o_h = try self.w_hh_o.forward(graph, h_prev);
        const o_sum = try graph.add(o_x, o_h);
        const o_t = try graph.sigmoid(o_sum);

        // 5. 细胞状态更新: C_t = f_t * C_{t-1} + i_t * c_cand
        const f_c_prev = try graph.mul(f_t, c_prev);
        const i_c_cand = try graph.mul(i_t, c_cand);
        const c_t = try graph.add(f_c_prev, i_c_cand);

        // 6. 隐状态更新: h_t = o_t * tanh(C_t)
        const tanh_c = try graph.tanh(c_t);
        const h_t = try graph.mul(o_t, tanh_c);

        return LSTMState{
            .h = h_t,
            .c = c_t,
        };
    }
};

/// 单层长短期记忆网络 (Long Short-Term Memory, LSTM) 序列前向传播返回结果
pub const LSTMResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
    c_n: *Tensor,
};

/// 沿时间展开的单层长短期记忆网络 (Long Short-Term Memory, LSTM) 序列容器
pub const LSTM = struct {
    cell: LSTMCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    module_type: []const u8 = "LSTM",

    pub const formula = "(h_{1:T}, c_{1:T}) = \\text{LSTM}(x_{1:T}, h_0, c_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !LSTM {
        const cell = try LSTMCell.init(allocator, input_dim, hidden_dim);
        return LSTM{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn forward(
        self: *const LSTM,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
        c_0: ?*Tensor,
    ) !LSTMResult {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try graph.arenaAllocator().alloc(*Tensor, seq_len);

        const batch_size = inputs[0].shape.dims[0];
        var h_curr: *Tensor = undefined;
        var c_curr: *Tensor = undefined;

        if (h_0) |h| {
            h_curr = h;
        } else {
            h_curr = try zeroState(graph, self.name, batch_size, self.hidden_dim, "h_0", .{});
        }

        if (c_0) |c| {
            c_curr = c;
        } else {
            c_curr = try zeroState(graph, self.name, batch_size, self.hidden_dim, "c_0", .{});
        }

        for (inputs, 0..) |x_t, t| {
            const state = try self.cell.forward(graph, x_t, h_curr, c_curr);
            outputs[t] = state.h;
            h_curr = state.h;
            c_curr = state.c;
        }

        return .{
            .outputs = outputs,
            .h_n = h_curr,
            .c_n = c_curr,
        };
    }
};

/// 多层堆叠长短期记忆网络 (Stacked Long Short-Term Memory, StackedLSTM) 序列前向传播返回结果
pub const StackedLSTMResult = struct {
    outputs: []*Tensor,
    h_n: []*Tensor,
    c_n: []*Tensor,
};

/// 多层堆叠长短期记忆网络 (Stacked Long Short-Term Memory, StackedLSTM) 引擎
pub const StackedLSTM = struct {
    num_layers: usize,
    input_dim: usize,
    hidden_dim: usize,
    layers: []LSTMCell,
    name: ?[]const u8 = null,
    module_type: []const u8 = "StackedLSTM",

    pub const formula = "h^{(L)}_{1:T} = \\text{StackedLSTM}(x_{1:T})";

    pub fn init(
        allocator: std.mem.Allocator,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) !StackedLSTM {
        const layers = try allocator.alloc(LSTMCell, num_layers);
        errdefer allocator.free(layers);
        var initialized_count: usize = 0;
        errdefer {
            for (0..initialized_count) |i| {
                deinitModel(&layers[i], allocator);
            }
        }
        for (0..num_layers) |i| {
            const in_d = if (i == 0) input_dim else hidden_dim;
            layers[i] = try LSTMCell.init(allocator, in_d, hidden_dim);
            initialized_count += 1;
        }
        return StackedLSTM{
            .num_layers = num_layers,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .layers = layers,
        };
    }

    pub fn forwardStep(
        self: *const StackedLSTM,
        graph: *autodiff.Graph,
        x_t: *Tensor,
        h_prevs: []const *Tensor,
        c_prevs: []const *Tensor,
        h_outs: []*Tensor,
        c_outs: []*Tensor,
    ) !void {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        var cur_in = x_t;
        for (0..self.num_layers) |l| {
            const state = try self.layers[l].forward(graph, cur_in, h_prevs[l], c_prevs[l]);
            h_outs[l] = state.h;
            c_outs[l] = state.c;
            cur_in = state.h;
        }
    }

    pub fn forwardSequence(
        self: *const StackedLSTM,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?[]const *Tensor,
        c_0: ?[]const *Tensor,
    ) !StackedLSTMResult {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const L = self.num_layers;
        const batch_size = inputs[0].shape.dims[0];

        var h_states = try graph.arenaAllocator().alloc(*Tensor, L);
        var c_states = try graph.arenaAllocator().alloc(*Tensor, L);

        for (0..L) |l| {
            if (h_0) |h_inits| {
                h_states[l] = h_inits[l];
            } else {
                h_states[l] = try zeroState(graph, self.name, batch_size, self.hidden_dim, "h_0_{d}", .{l});
            }

            if (c_0) |c_inits| {
                c_states[l] = c_inits[l];
            } else {
                c_states[l] = try zeroState(graph, self.name, batch_size, self.hidden_dim, "c_0_{d}", .{l});
            }
        }

        const outputs = try graph.arenaAllocator().alloc(*Tensor, seq_len);
        for (inputs, 0..) |x_t, t| {
            var cur_in = x_t;
            for (0..L) |l| {
                const state = try self.layers[l].forward(graph, cur_in, h_states[l], c_states[l]);
                h_states[l] = state.h;
                c_states[l] = state.c;
                cur_in = state.h;
            }
            outputs[t] = cur_in;
        }

        return .{
            .outputs = outputs,
            .h_n = h_states,
            .c_n = c_states,
        };
    }
};

/// 门控循环单元 (Gated Recurrent Unit Cell, GRUCell)
pub const GRUCell = struct {
    input_dim: usize,
    hidden_dim: usize,
    // 重置门 (Reset Gate)
    w_ih_r: Linear,
    w_hh_r: Linear,
    // 更新门 (Update Gate)
    w_ih_z: Linear,
    w_hh_z: Linear,
    // 候选隐状态 (Candidate Hidden State)
    w_ih_h: Linear,
    w_hh_h: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GRUCell",

    pub const formula = "h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}))";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !GRUCell {
        const w_ih_r = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_r, allocator);
        const w_hh_r = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_r, allocator);

        const w_ih_z = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_z, allocator);
        const w_hh_z = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_z, allocator);

        const w_ih_h = try Linear.init(allocator, input_dim, hidden_dim);
        errdefer deinitModel(&w_ih_h, allocator);
        const w_hh_h = try Linear.init(allocator, hidden_dim, hidden_dim);
        errdefer deinitModel(&w_hh_h, allocator);

        return GRUCell{
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .w_ih_r = w_ih_r,
            .w_hh_r = w_hh_r,
            .w_ih_z = w_ih_z,
            .w_hh_z = w_hh_z,
            .w_ih_h = w_ih_h,
            .w_hh_h = w_hh_h,
        };
    }

    pub fn forward(
        self: *const GRUCell,
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        // 1. 重置门: r_t = sigmoid(W_r * x + U_r * h_prev)
        const r_x = try self.w_ih_r.forward(graph, x);
        const r_h = try self.w_hh_r.forward(graph, h_prev);
        const r_sum = try graph.add(r_x, r_h);
        const r_t = try graph.sigmoid(r_sum);

        // 2. 更新门: z_t = sigmoid(W_z * x + U_z * h_prev)
        const z_x = try self.w_ih_z.forward(graph, x);
        const z_h = try self.w_hh_z.forward(graph, h_prev);
        const z_sum = try graph.add(z_x, z_h);
        const z_t = try graph.sigmoid(z_sum);

        // 3. 候选隐状态: h_tilde = tanh(W_h * x + U_h * (r_t * h_prev))
        const rh = try graph.mul(r_t, h_prev);
        const h_x = try self.w_ih_h.forward(graph, x);
        const h_h = try self.w_hh_h.forward(graph, rh);
        const cand_sum = try graph.add(h_x, h_h);
        const h_tilde = try graph.tanh(cand_sum);

        // 4. 隐状态融合: h_t = (1 - z_t) * h_prev + z_t * h_tilde
        const neg_z = try graph.mulScalar(z_t, -1.0);
        const one_minus_z = try graph.addScalar(neg_z, 1.0);

        const term1 = try graph.mul(one_minus_z, h_prev);
        const term2 = try graph.mul(z_t, h_tilde);

        return try graph.add(term1, term2);
    }
};

/// 单层门控循环单元 (Gated Recurrent Unit, GRU) 序列前向传播返回结果
pub const GRUResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
};

/// 沿时间展开的单层门控循环单元 (Gated Recurrent Unit, GRU) 序列容器
pub const GRU = struct {
    cell: GRUCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GRU",

    pub const formula = "h_{1:T} = \\text{GRU}(x_{1:T}, h_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !GRU {
        const cell = try GRUCell.init(allocator, input_dim, hidden_dim);
        return GRU{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn forward(
        self: *const GRU,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !GRUResult {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try graph.arenaAllocator().alloc(*Tensor, seq_len);

        var h_curr: *Tensor = undefined;
        if (h_0) |h| {
            h_curr = h;
        } else {
            const batch_size = inputs[0].shape.dims[0];
            h_curr = try zeroState(graph, self.name, batch_size, self.hidden_dim, "h_0", .{});
        }

        for (inputs, 0..) |x_t, t| {
            const h_next = try self.cell.forward(graph, x_t, h_curr);
            outputs[t] = h_next;
            h_curr = h_next;
        }

        return .{
            .outputs = outputs,
            .h_n = h_curr,
        };
    }
};
