const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const Tensor = tensor.Tensor;
const Linear = core.Linear;

// ============================================================================
// 循环神经网络模块 (Recurrent Neural Network Modules)
// ============================================================================

/// 单步经典 Elman RNN 单元 (RNNCell)
/// 隐状态更新公式：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
pub const RNNCell = struct {
    input_dim: usize,
    hidden_dim: usize,
    weight_ih: Linear,
    weight_hh: Linear,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !RNNCell {
        const weight_ih = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer weight_ih.deinit(allocator);
        const weight_hh = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer weight_hh.deinit(allocator);

        return RNNCell{
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
        };
    }

    pub fn deinit(self: RNNCell, allocator: std.mem.Allocator) void {
        self.weight_ih.deinit(allocator);
        self.weight_hh.deinit(allocator);
    }

    pub fn zeroGrad(self: RNNCell) void {
        self.weight_ih.zeroGrad();
        self.weight_hh.zeroGrad();
    }

    /// 单时间步前向：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    /// x: [batch_size, input_dim], h_prev: [batch_size, hidden_dim]
    pub fn forward(
        self: RNNCell,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        const x_proj = try self.weight_ih.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, x_proj);
        const h_proj = try self.weight_hh.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, h_proj);

        const sum = if (graph) |g| try g.add(x_proj, h_proj) else try x_proj.add(h_proj, allocator, null);
        defer if (graph == null) tensor.free(allocator, sum);

        return if (graph) |g| try g.tanh(sum) else try sum.tanh(allocator, null);
    }
};

/// 沿时间展开的 Elman RNN 序列容器
pub const RNN = struct {
    cell: RNNCell,
    input_dim: usize,
    hidden_dim: usize,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !RNN {
        const cell = try RNNCell.init(allocator, input_dim, hidden_dim, random);
        return RNN{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn deinit(self: RNN, allocator: std.mem.Allocator) void {
        self.cell.deinit(allocator);
    }

    pub fn zeroGrad(self: RNN) void {
        self.cell.zeroGrad();
    }

    /// 序列时序展开前向传播
    /// inputs: 长度为 seq_len 的 Tensor 切片，每个形状为 [batch_size, input_dim]
    /// h_0: 初始隐状态 [batch_size, hidden_dim]，若为 null 则自动置零
    pub fn forward(
        self: RNN,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !struct { outputs: []*Tensor, h_n: *Tensor } {
        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try allocator.alloc(*Tensor, seq_len);

        var h_curr: *Tensor = undefined;
        if (h_0) |h| {
            h_curr = h;
        } else {
            const batch_size = inputs[0].shape.dims[0];
            if (graph) |g| {
                h_curr = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
            } else {
                h_curr = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
            }
        }

        for (inputs, 0..) |x_t, t| {
            const h_next = try self.cell.forward(allocator, graph, x_t, h_curr);
            outputs[t] = h_next;
            h_curr = h_next;
        }

        return .{
            .outputs = outputs,
            .h_n = h_curr,
        };
    }
};

/// LSTM 单元内部状态对 (h_t, c_t)
pub const LSTMState = struct {
    h: *Tensor,
    c: *Tensor,
};

/// 单步长短期记忆网络单元 (LSTMCell)
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

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !LSTMCell {
        const w_ih_f = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_f.deinit(allocator);
        const w_hh_f = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_f.deinit(allocator);

        const w_ih_i = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_i.deinit(allocator);
        const w_hh_i = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_i.deinit(allocator);

        const w_ih_c = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_c.deinit(allocator);
        const w_hh_c = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_c.deinit(allocator);

        const w_ih_o = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_o.deinit(allocator);
        const w_hh_o = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_o.deinit(allocator);

        // 关键技巧：将遗忘门偏置初始化为 +1.0，促进长程梯度回传 (Gers et al., 2000)
        @memset(w_ih_f.bias.data, 1.0);

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

    pub fn deinit(self: LSTMCell, allocator: std.mem.Allocator) void {
        self.w_ih_f.deinit(allocator);
        self.w_hh_f.deinit(allocator);
        self.w_ih_i.deinit(allocator);
        self.w_hh_i.deinit(allocator);
        self.w_ih_c.deinit(allocator);
        self.w_hh_c.deinit(allocator);
        self.w_ih_o.deinit(allocator);
        self.w_hh_o.deinit(allocator);
    }

    pub fn zeroGrad(self: LSTMCell) void {
        self.w_ih_f.zeroGrad();
        self.w_hh_f.zeroGrad();
        self.w_ih_i.zeroGrad();
        self.w_hh_i.zeroGrad();
        self.w_ih_c.zeroGrad();
        self.w_hh_c.zeroGrad();
        self.w_ih_o.zeroGrad();
        self.w_hh_o.zeroGrad();
    }

    /// 单步前向计算
    /// x: [batch_size, input_dim]
    /// h_prev: [batch_size, hidden_dim]
    /// c_prev: [batch_size, hidden_dim]
    pub fn forward(
        self: LSTMCell,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
        c_prev: *Tensor,
    ) !LSTMState {
        // 1. 遗忘门: f_t = sigmoid(W_f * x + U_f * h_prev)
        const f_x = try self.w_ih_f.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, f_x);
        const f_h = try self.w_hh_f.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, f_h);
        const f_sum = if (graph) |g| try g.add(f_x, f_h) else try f_x.add(f_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, f_sum);
        const f_t = if (graph) |g| try g.sigmoid(f_sum) else try f_sum.sigmoid(allocator, null);
        defer if (graph == null) tensor.free(allocator, f_t);

        // 2. 输入门: i_t = sigmoid(W_i * x + U_i * h_prev)
        const i_x = try self.w_ih_i.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, i_x);
        const i_h = try self.w_hh_i.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, i_h);
        const i_sum = if (graph) |g| try g.add(i_x, i_h) else try i_x.add(i_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, i_sum);
        const i_t = if (graph) |g| try g.sigmoid(i_sum) else try i_sum.sigmoid(allocator, null);
        defer if (graph == null) tensor.free(allocator, i_t);

        // 3. 候选状态: c_cand = tanh(W_c * x + U_c * h_prev)
        const c_x = try self.w_ih_c.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, c_x);
        const c_h = try self.w_hh_c.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, c_h);
        const c_sum = if (graph) |g| try g.add(c_x, c_h) else try c_x.add(c_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, c_sum);
        const c_cand = if (graph) |g| try g.tanh(c_sum) else try c_sum.tanh(allocator, null);
        defer if (graph == null) tensor.free(allocator, c_cand);

        // 4. 输出门: o_t = sigmoid(W_o * x + U_o * h_prev)
        const o_x = try self.w_ih_o.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, o_x);
        const o_h = try self.w_hh_o.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, o_h);
        const o_sum = if (graph) |g| try g.add(o_x, o_h) else try o_x.add(o_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, o_sum);
        const o_t = if (graph) |g| try g.sigmoid(o_sum) else try o_sum.sigmoid(allocator, null);
        defer if (graph == null) tensor.free(allocator, o_t);

        // 5. 细胞状态更新: C_t = f_t * C_{t-1} + i_t * c_cand
        const f_c_prev = if (graph) |g| try g.mul(f_t, c_prev) else try f_t.mul(c_prev, allocator, null);
        defer if (graph == null) tensor.free(allocator, f_c_prev);
        const i_c_cand = if (graph) |g| try g.mul(i_t, c_cand) else try i_t.mul(c_cand, allocator, null);
        defer if (graph == null) tensor.free(allocator, i_c_cand);
        const c_t = if (graph) |g| try g.add(f_c_prev, i_c_cand) else try f_c_prev.add(i_c_cand, allocator, null);

        // 6. 隐状态更新: h_t = o_t * tanh(C_t)
        const tanh_c = if (graph) |g| try g.tanh(c_t) else try c_t.tanh(allocator, null);
        defer if (graph == null) tensor.free(allocator, tanh_c);
        const h_t = if (graph) |g| try g.mul(o_t, tanh_c) else try o_t.mul(tanh_c, allocator, null);

        return LSTMState{
            .h = h_t,
            .c = c_t,
        };
    }
};

/// 沿时间展开的单层 LSTM 序列容器
pub const LSTM = struct {
    cell: LSTMCell,
    input_dim: usize,
    hidden_dim: usize,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !LSTM {
        const cell = try LSTMCell.init(allocator, input_dim, hidden_dim, random);
        return LSTM{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn deinit(self: LSTM, allocator: std.mem.Allocator) void {
        self.cell.deinit(allocator);
    }

    pub fn zeroGrad(self: LSTM) void {
        self.cell.zeroGrad();
    }

    pub fn forward(
        self: LSTM,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
        c_0: ?*Tensor,
    ) !struct { outputs: []*Tensor, h_n: *Tensor, c_n: *Tensor } {
        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try allocator.alloc(*Tensor, seq_len);

        const batch_size = inputs[0].shape.dims[0];
        var h_curr: *Tensor = undefined;
        var c_curr: *Tensor = undefined;

        if (h_0) |h| {
            h_curr = h;
        } else {
            if (graph) |g| {
                h_curr = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
            } else {
                h_curr = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
            }
        }

        if (c_0) |c| {
            c_curr = c;
        } else {
            if (graph) |g| {
                c_curr = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
            } else {
                c_curr = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
            }
        }

        for (inputs, 0..) |x_t, t| {
            const state = try self.cell.forward(allocator, graph, x_t, h_curr, c_curr);
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

/// 多层堆叠 LSTM (Stacked / Deep LSTM) 引擎
pub const StackedLSTM = struct {
    num_layers: usize,
    input_dim: usize,
    hidden_dim: usize,
    layers: []LSTMCell,

    pub fn init(
        allocator: std.mem.Allocator,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        random: std.Random,
    ) !StackedLSTM {
        const layers = try allocator.alloc(LSTMCell, num_layers);
        errdefer allocator.free(layers);
        var initialized_count: usize = 0;
        errdefer {
            for (0..initialized_count) |i| {
                layers[i].deinit(allocator);
            }
        }
        for (0..num_layers) |i| {
            const in_d = if (i == 0) input_dim else hidden_dim;
            layers[i] = try LSTMCell.init(allocator, in_d, hidden_dim, random);
            initialized_count += 1;
        }
        return StackedLSTM{
            .num_layers = num_layers,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .layers = layers,
        };
    }

    pub fn deinit(self: StackedLSTM, allocator: std.mem.Allocator) void {
        for (self.layers) |layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn zeroGrad(self: StackedLSTM) void {
        for (self.layers) |layer| {
            layer.zeroGrad();
        }
    }

    pub fn forwardStep(
        self: StackedLSTM,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        x_t: *Tensor,
        h_prevs: []const *Tensor,
        c_prevs: []const *Tensor,
        h_outs: []*Tensor,
        c_outs: []*Tensor,
    ) !void {
        var cur_in = x_t;
        for (0..self.num_layers) |l| {
            const state = try self.layers[l].forward(allocator, graph, cur_in, h_prevs[l], c_prevs[l]);
            h_outs[l] = state.h;
            c_outs[l] = state.c;
            cur_in = state.h;
        }
    }

    pub fn forwardSequence(
        self: StackedLSTM,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?[]const *Tensor,
        c_0: ?[]const *Tensor,
    ) !struct {
        outputs: []*Tensor,
        h_n: []*Tensor,
        c_n: []*Tensor,
    } {
        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const L = self.num_layers;
        const batch_size = inputs[0].shape.dims[0];

        var h_states = try allocator.alloc(*Tensor, L);
        var c_states = try allocator.alloc(*Tensor, L);

        for (0..L) |l| {
            if (h_0) |h_inits| {
                h_states[l] = h_inits[l];
            } else {
                if (graph) |g| {
                    h_states[l] = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
                } else {
                    h_states[l] = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
                }
            }

            if (c_0) |c_inits| {
                c_states[l] = c_inits[l];
            } else {
                if (graph) |g| {
                    c_states[l] = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
                } else {
                    c_states[l] = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
                }
            }
        }

        const outputs = try allocator.alloc(*Tensor, seq_len);
        for (inputs, 0..) |x_t, t| {
            var cur_in = x_t;
            for (0..L) |l| {
                const state = try self.layers[l].forward(allocator, graph, cur_in, h_states[l], c_states[l]);
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

/// 门控循环单元 (GRUCell)
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

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !GRUCell {
        const w_ih_r = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_r.deinit(allocator);
        const w_hh_r = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_r.deinit(allocator);

        const w_ih_z = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_z.deinit(allocator);
        const w_hh_z = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_z.deinit(allocator);

        const w_ih_h = try Linear.init(allocator, input_dim, hidden_dim, random);
        errdefer w_ih_h.deinit(allocator);
        const w_hh_h = try Linear.init(allocator, hidden_dim, hidden_dim, random);
        errdefer w_hh_h.deinit(allocator);

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

    pub fn deinit(self: GRUCell, allocator: std.mem.Allocator) void {
        self.w_ih_r.deinit(allocator);
        self.w_hh_r.deinit(allocator);
        self.w_ih_z.deinit(allocator);
        self.w_hh_z.deinit(allocator);
        self.w_ih_h.deinit(allocator);
        self.w_hh_h.deinit(allocator);
    }

    pub fn zeroGrad(self: GRUCell) void {
        self.w_ih_r.zeroGrad();
        self.w_hh_r.zeroGrad();
        self.w_ih_z.zeroGrad();
        self.w_hh_z.zeroGrad();
        self.w_ih_h.zeroGrad();
        self.w_hh_h.zeroGrad();
    }

    pub fn forward(
        self: GRUCell,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        // 1. 重置门: r_t = sigmoid(W_r * x + U_r * h_prev)
        const r_x = try self.w_ih_r.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, r_x);
        const r_h = try self.w_hh_r.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, r_h);
        const r_sum = if (graph) |g| try g.add(r_x, r_h) else try r_x.add(r_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, r_sum);
        const r_t = if (graph) |g| try g.sigmoid(r_sum) else try r_sum.sigmoid(allocator, null);
        defer if (graph == null) tensor.free(allocator, r_t);

        // 2. 更新门: z_t = sigmoid(W_z * x + U_z * h_prev)
        const z_x = try self.w_ih_z.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, z_x);
        const z_h = try self.w_hh_z.forward(allocator, graph, h_prev);
        defer if (graph == null) tensor.free(allocator, z_h);
        const z_sum = if (graph) |g| try g.add(z_x, z_h) else try z_x.add(z_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, z_sum);
        const z_t = if (graph) |g| try g.sigmoid(z_sum) else try z_sum.sigmoid(allocator, null);
        defer if (graph == null) tensor.free(allocator, z_t);

        // 3. 候选隐状态: h_tilde = tanh(W_h * x + U_h * (r_t * h_prev))
        const rh = if (graph) |g| try g.mul(r_t, h_prev) else try r_t.mul(h_prev, allocator, null);
        defer if (graph == null) tensor.free(allocator, rh);
        const h_x = try self.w_ih_h.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, h_x);
        const h_h = try self.w_hh_h.forward(allocator, graph, rh);
        defer if (graph == null) tensor.free(allocator, h_h);
        const cand_sum = if (graph) |g| try g.add(h_x, h_h) else try h_x.add(h_h, allocator, null);
        defer if (graph == null) tensor.free(allocator, cand_sum);
        const h_tilde = if (graph) |g| try g.tanh(cand_sum) else try cand_sum.tanh(allocator, null);
        defer if (graph == null) tensor.free(allocator, h_tilde);

        // 4. 隐状态融合: h_t = (1 - z_t) * h_prev + z_t * h_tilde
        const neg_z = if (graph) |g| try g.mulScalar(z_t, -1.0) else try z_t.mulScalar(-1.0, allocator, null);
        defer if (graph == null) tensor.free(allocator, neg_z);
        const one_minus_z = if (graph) |g| try g.addScalar(neg_z, 1.0) else try neg_z.addScalar(1.0, allocator, null);
        defer if (graph == null) tensor.free(allocator, one_minus_z);

        const term1 = if (graph) |g| try g.mul(one_minus_z, h_prev) else try one_minus_z.mul(h_prev, allocator, null);
        defer if (graph == null) tensor.free(allocator, term1);
        const term2 = if (graph) |g| try g.mul(z_t, h_tilde) else try z_t.mul(h_tilde, allocator, null);
        defer if (graph == null) tensor.free(allocator, term2);

        return if (graph) |g| try g.add(term1, term2) else try term1.add(term2, allocator, null);
    }
};

/// 沿时间展开的单层 GRU 序列容器
pub const GRU = struct {
    cell: GRUCell,
    input_dim: usize,
    hidden_dim: usize,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random: std.Random) !GRU {
        const cell = try GRUCell.init(allocator, input_dim, hidden_dim, random);
        return GRU{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn deinit(self: GRU, allocator: std.mem.Allocator) void {
        self.cell.deinit(allocator);
    }

    pub fn zeroGrad(self: GRU) void {
        self.cell.zeroGrad();
    }

    pub fn forward(
        self: GRU,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !struct { outputs: []*Tensor, h_n: *Tensor } {
        std.debug.assert(inputs.len > 0);
        const seq_len = inputs.len;
        const outputs = try allocator.alloc(*Tensor, seq_len);

        var h_curr: *Tensor = undefined;
        if (h_0) |h| {
            h_curr = h;
        } else {
            const batch_size = inputs[0].shape.dims[0];
            if (graph) |g| {
                h_curr = try g.zeros(&.{ batch_size, self.hidden_dim }, false);
            } else {
                h_curr = try tensor.zeros(allocator, &.{ batch_size, self.hidden_dim });
            }
        }

        for (inputs, 0..) |x_t, t| {
            const h_next = try self.cell.forward(allocator, graph, x_t, h_curr);
            outputs[t] = h_next;
            h_curr = h_next;
        }

        return .{
            .outputs = outputs,
            .h_n = h_curr,
        };
    }
};
