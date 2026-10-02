const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const Tensor = tensor.Tensor;
const Linear = core.Linear;

// ============================================================================
// 循环神经网络模块 (Recurrent Neural Network Modules)
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

/// 单步经典 Elman RNN 单元 (RNNCell)
/// 隐状态更新公式：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
pub const RNNCell = struct {
    input_dim: usize,
    hidden_dim: usize,
    weight_ih: Linear,
    weight_hh: Linear,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "RNNCell",

    pub const formula = "h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !RNNCell {
        const weight_ih = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer weight_ih.deinit(allocator);
        const weight_hh = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer weight_hh.deinit(allocator);

        return RNNCell{
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
        };
    }

    pub fn setName(self: *RNNCell, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight_ih.setNameFormatted("{s}.weight_ih", .{self.name.?});
        self.weight_hh.setNameFormatted("{s}.weight_hh", .{self.name.?});
    }

    pub fn setNameFormatted(self: *RNNCell, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("rnn_cell");
        }
    }

    pub fn getName(self: *const RNNCell) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const RNNCell, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.weight_ih.registerFormula(graph);
            try self.weight_hh.registerFormula(graph);
        }
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
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

        const x_proj = try self.weight_ih.forward(graph, x);
        const h_proj = try self.weight_hh.forward(graph, h_prev);

        const sum = try graph.add(x_proj, h_proj);

        return try graph.tanh(sum);
    }
};

/// Elman RNN 序列前向传播返回结果
pub const RNNResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
};

/// 沿时间展开的 Elman RNN 序列容器
pub const RNN = struct {
    cell: RNNCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "RNN",

    pub const formula = "h_{1:T} = \\text{RNN}(x_{1:T}, h_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !RNN {
        const cell = try RNNCell.init(allocator, input_dim, hidden_dim, random_opt);
        return RNN{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn setName(self: *RNN, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.cell.setNameFormatted("{s}.cell", .{self.name.?});
    }

    pub fn setNameFormatted(self: *RNN, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("rnn");
        }
    }

    pub fn getName(self: *const RNN) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const RNN, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.cell.registerFormula(graph);
        }
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
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !RNNResult {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "LSTMCell",

    pub const formula = "c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, \\quad h_t = o_t \\odot \\tanh(c_t)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !LSTMCell {
        const w_ih_f = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_f.deinit(allocator);
        const w_hh_f = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_f.deinit(allocator);

        const w_ih_i = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_i.deinit(allocator);
        const w_hh_i = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_i.deinit(allocator);

        const w_ih_c = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_c.deinit(allocator);
        const w_hh_c = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_c.deinit(allocator);

        const w_ih_o = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_o.deinit(allocator);
        const w_hh_o = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_o.deinit(allocator);

        // 关键技巧：将遗忘门偏置初始化为 +1.0，促进长程梯度回传 (Gers et al., 2000)
        @memset(w_ih_f.bias.data, 1.0);
        w_ih_f.bias.is_custom_initialized = true;

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

    pub fn setName(self: *LSTMCell, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.w_ih_f.setNameFormatted("{s}.w_ih_f", .{self.name.?});
        self.w_hh_f.setNameFormatted("{s}.w_hh_f", .{self.name.?});
        self.w_ih_i.setNameFormatted("{s}.w_ih_i", .{self.name.?});
        self.w_hh_i.setNameFormatted("{s}.w_hh_i", .{self.name.?});
        self.w_ih_c.setNameFormatted("{s}.w_ih_c", .{self.name.?});
        self.w_hh_c.setNameFormatted("{s}.w_hh_c", .{self.name.?});
        self.w_ih_o.setNameFormatted("{s}.w_ih_o", .{self.name.?});
        self.w_hh_o.setNameFormatted("{s}.w_hh_o", .{self.name.?});
    }

    pub fn setNameFormatted(self: *LSTMCell, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("lstm_cell");
        }
    }

    pub fn getName(self: *const LSTMCell) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const LSTMCell, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.w_ih_f.registerFormula(graph);
            try self.w_hh_f.registerFormula(graph);
            try self.w_ih_i.registerFormula(graph);
            try self.w_hh_i.registerFormula(graph);
            try self.w_ih_c.registerFormula(graph);
            try self.w_hh_c.registerFormula(graph);
            try self.w_ih_o.registerFormula(graph);
            try self.w_hh_o.registerFormula(graph);
        }
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
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
        c_prev: *Tensor,
    ) !LSTMState {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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

/// 单层 LSTM 序列前向传播返回结果
pub const LSTMResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
    c_n: *Tensor,
};

/// 沿时间展开的单层 LSTM 序列容器
pub const LSTM = struct {
    cell: LSTMCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "LSTM",

    pub const formula = "(h_{1:T}, c_{1:T}) = \\text{LSTM}(x_{1:T}, h_0, c_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !LSTM {
        const cell = try LSTMCell.init(allocator, input_dim, hidden_dim, random_opt);
        return LSTM{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn setName(self: *LSTM, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.cell.setNameFormatted("{s}.cell", .{self.name.?});
    }

    pub fn setNameFormatted(self: *LSTM, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("lstm");
        }
    }

    pub fn getName(self: *const LSTM) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const LSTM, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.cell.registerFormula(graph);
        }
    }

    pub fn deinit(self: LSTM, allocator: std.mem.Allocator) void {
        self.cell.deinit(allocator);
    }

    pub fn zeroGrad(self: LSTM) void {
        self.cell.zeroGrad();
    }

    pub fn forward(
        self: LSTM,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
        c_0: ?*Tensor,
    ) !LSTMResult {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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

/// 多层堆叠 LSTM 序列前向传播返回结果
pub const StackedLSTMResult = struct {
    outputs: []*Tensor,
    h_n: []*Tensor,
    c_n: []*Tensor,
};

/// 多层堆叠 LSTM (Stacked / Deep LSTM) 引擎
pub const StackedLSTM = struct {
    num_layers: usize,
    input_dim: usize,
    hidden_dim: usize,
    layers: []LSTMCell,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "StackedLSTM",

    pub const formula = "h^{(L)}_{1:T} = \\text{StackedLSTM}(x_{1:T})";

    pub fn init(
        allocator: std.mem.Allocator,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        random_opt: anytype,
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
            layers[i] = try LSTMCell.init(allocator, in_d, hidden_dim, random_opt);
            initialized_count += 1;
        }
        return StackedLSTM{
            .num_layers = num_layers,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .layers = layers,
        };
    }

    pub fn setName(self: *StackedLSTM, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        for (self.layers, 0..) |*layer, i| {
            layer.setNameFormatted("{s}.layer_{d}", .{ self.name.?, i });
        }
    }

    pub fn setNameFormatted(self: *StackedLSTM, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("stacked_lstm");
        }
    }

    pub fn getName(self: *const StackedLSTM) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const StackedLSTM, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            for (self.layers) |*layer| {
                try layer.registerFormula(graph);
            }
        }
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
        graph: *autodiff.Graph,
        x_t: *Tensor,
        h_prevs: []const *Tensor,
        c_prevs: []const *Tensor,
        h_outs: []*Tensor,
        c_outs: []*Tensor,
    ) !void {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

        var cur_in = x_t;
        for (0..self.num_layers) |l| {
            const state = try self.layers[l].forward(graph, cur_in, h_prevs[l], c_prevs[l]);
            h_outs[l] = state.h;
            c_outs[l] = state.c;
            cur_in = state.h;
        }
    }

    pub fn forwardSequence(
        self: StackedLSTM,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?[]const *Tensor,
        c_0: ?[]const *Tensor,
    ) !StackedLSTMResult {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "GRUCell",

    pub const formula = "h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}))";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !GRUCell {
        const w_ih_r = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_r.deinit(allocator);
        const w_hh_r = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_r.deinit(allocator);

        const w_ih_z = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_z.deinit(allocator);
        const w_hh_z = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
        errdefer w_hh_z.deinit(allocator);

        const w_ih_h = try Linear.init(allocator, input_dim, hidden_dim, random_opt);
        errdefer w_ih_h.deinit(allocator);
        const w_hh_h = try Linear.init(allocator, hidden_dim, hidden_dim, random_opt);
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

    pub fn setName(self: *GRUCell, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.w_ih_r.setNameFormatted("{s}.w_ih_r", .{self.name.?});
        self.w_hh_r.setNameFormatted("{s}.w_hh_r", .{self.name.?});
        self.w_ih_z.setNameFormatted("{s}.w_ih_z", .{self.name.?});
        self.w_hh_z.setNameFormatted("{s}.w_hh_z", .{self.name.?});
        self.w_ih_h.setNameFormatted("{s}.w_ih_h", .{self.name.?});
        self.w_hh_h.setNameFormatted("{s}.w_hh_h", .{self.name.?});
    }

    pub fn setNameFormatted(self: *GRUCell, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("gru_cell");
        }
    }

    pub fn getName(self: *const GRUCell) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const GRUCell, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.w_ih_r.registerFormula(graph);
            try self.w_hh_r.registerFormula(graph);
            try self.w_ih_z.registerFormula(graph);
            try self.w_hh_z.registerFormula(graph);
            try self.w_ih_h.registerFormula(graph);
            try self.w_hh_h.registerFormula(graph);
        }
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
        graph: *autodiff.Graph,
        x: *Tensor,
        h_prev: *Tensor,
    ) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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

/// 单层 GRU 序列前向传播返回结果
pub const GRUResult = struct {
    outputs: []*Tensor,
    h_n: *Tensor,
};

/// 沿时间展开的单层 GRU 序列容器
pub const GRU = struct {
    cell: GRUCell,
    input_dim: usize,
    hidden_dim: usize,
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "GRU",

    pub const formula = "h_{1:T} = \\text{GRU}(x_{1:T}, h_0)";

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize, random_opt: anytype) !GRU {
        const cell = try GRUCell.init(allocator, input_dim, hidden_dim, random_opt);
        return GRU{
            .cell = cell,
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn setName(self: *GRU, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.cell.setNameFormatted("{s}.cell", .{self.name.?});
    }

    pub fn setNameFormatted(self: *GRU, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("gru");
        }
    }

    pub fn getName(self: *const GRU) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const GRU, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.cell.registerFormula(graph);
        }
    }

    pub fn deinit(self: GRU, allocator: std.mem.Allocator) void {
        self.cell.deinit(allocator);
    }

    pub fn zeroGrad(self: GRU) void {
        self.cell.zeroGrad();
    }

    pub fn forward(
        self: GRU,
        graph: *autodiff.Graph,
        inputs: []const *Tensor,
        h_0: ?*Tensor,
    ) !GRUResult {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);

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
