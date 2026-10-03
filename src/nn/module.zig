//! 模块协议与反射遍历 (类似 PyTorch `nn.Module` 的参数 / 子模块自动注册)。
//!
//! znn 中的模块就是普通 struct：字段中的 `*Tensor` / `?*Tensor` / `[]*Tensor` / `[N]*Tensor` 是该模块持有的张量
//! (`requires_grad = true` 为可训练参数，否则为缓冲区)，struct 字段及其切片 / 数组是子模块。
//! `walk` 在编译期展开字段，按字段路径 (如 `cell.w_ih_r.weight`、`layers.0.attn.c_attn.bias`) 访问所有张量与子模块，
//! 释放、清零梯度、收集参数、训练 / 推理模式切换、自定义初始化、序列化与命名均建立在这一个遍历之上。
//!
//! 库层只需声明数据字段、`name: ?[]const u8 = null`、`module_type` 与可选的 `pub const formula`，
//! 不再各自实现 `deinit` / `zeroGrad` / `setName`：
//! - `nameModules(&model, arena, root)` 按字段路径为子模块与张量命名 (如 `gpt.decoder.h.0.attn`)，
//!   名称字符串存放在调用方的 arena 中，模型按值移动后仍然有效；
//! - `enterModuleScope(graph, self)` 在 `forward` 开头进入模块作用域并登记 `formula`。
const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;
const freePersistentTensor = @import("core.zig").freePersistentTensor;

/// 字段路径的最大长度 (字节)
pub const max_path_len = 256;

/// 遍历过程中的字段路径缓冲，按 `a.b.0.c` 形式拼接
pub const Path = struct {
    buf: [max_path_len]u8 = undefined,
    len: usize = 0,

    pub fn slice(self: *const Path) []const u8 {
        return self.buf[0..self.len];
    }

    fn push(self: *Path, comptime fmt: []const u8, args: anytype) error{PathTooLong}!usize {
        const mark = self.len;
        const sep: []const u8 = if (self.len > 0) "." else "";
        const written = std.fmt.bufPrint(self.buf[self.len..], "{s}" ++ fmt, .{sep} ++ args) catch return error.PathTooLong;
        self.len += written.len;
        return mark;
    }

    fn pop(self: *Path, mark: usize) void {
        self.len = mark;
    }
};

/// 递归遍历模型 `model` (指向 struct 的指针) 中的全部张量与子模块。
/// `visitor` 为指向访问者 struct 的指针，可声明以下成员：
/// - `pub fn visitTensor(self, path: []const u8, t: *Tensor) !void` (必需)：每个非空张量字段调用一次；
/// - `pub fn enterModule(self, path: []const u8, module: anytype) !bool` (可选)：进入每个 struct (含根模型) 前调用，
///   返回 `false` 时跳过其字段；
/// - `pub fn visitOwnedSlice(self, slice: anytype) !void` (可选)：可变切片字段 (`[]f32`、`[]*Tensor`、`[]Struct`)
///   的元素遍历完毕后调用 (用于释放切片本身)；
/// - `pub const wants_path = true` (可选)：需要字段路径时声明；未声明时 `path` 恒为空串，遍历不做任何字符串拼接。
/// 编译期字段与零大小字段会被跳过。
pub fn walk(model: anytype, visitor: anytype) !void {
    const M = @TypeOf(model);
    if (comptime @typeInfo(M) != .pointer or @typeInfo(@typeInfo(M).pointer.child) != .@"struct") {
        @compileError("expected a pointer to a module struct, found " ++ @typeName(M));
    }
    var path = Path{};
    try walkStruct(model, visitor, &path);
}

fn wantsPath(comptime V: type) bool {
    return @hasDecl(V, "wants_path") and V.wants_path;
}

fn pushPath(comptime V: type, path: *Path, comptime fmt: []const u8, args: anytype) !usize {
    if (comptime wantsPath(V)) return path.push(fmt, args) else return 0;
}

fn popPath(comptime V: type, path: *Path, mark: usize) void {
    if (comptime wantsPath(V)) path.pop(mark);
}

fn walkStruct(model: anytype, visitor: anytype, path: *Path) !void {
    const T = @TypeOf(model.*);
    const info = @typeInfo(T);
    if (info != .@"struct") return;
    const V = @TypeOf(visitor.*);
    if (@hasDecl(V, "enterModule")) {
        if (!try visitor.enterModule(path.slice(), model)) return;
    }
    inline for (info.@"struct".fields) |field| {
        if (field.is_comptime or @sizeOf(field.type) == 0) continue;
        const mark = try pushPath(V, path, "{s}", .{field.name});
        defer popPath(V, path, mark);
        try walkField(&@field(model, field.name), visitor, path);
    }
}

fn walkField(field_ptr: anytype, visitor: anytype, path: *Path) !void {
    const F = @TypeOf(field_ptr.*);
    const V = @TypeOf(visitor.*);
    const info = @typeInfo(F);
    if (F == *Tensor) {
        try visitor.visitTensor(path.slice(), field_ptr.*);
    } else if (info == .optional and info.optional.child == *Tensor) {
        if (field_ptr.*) |t| try visitor.visitTensor(path.slice(), t);
    } else if (info == .pointer and info.pointer.size == .slice and !info.pointer.is_const) {
        const E = info.pointer.child;
        if (E == f32) {
            if (@hasDecl(V, "visitOwnedSlice")) try visitor.visitOwnedSlice(field_ptr.*);
        } else if (E == *Tensor or @typeInfo(E) == .@"struct") {
            for (field_ptr.*, 0..) |*item, idx| {
                const mark = try pushPath(V, path, "{d}", .{idx});
                defer popPath(V, path, mark);
                try walkElement(item, visitor, path);
            }
            if (@hasDecl(V, "visitOwnedSlice")) try visitor.visitOwnedSlice(field_ptr.*);
        }
    } else if (info == .@"struct") {
        try walkStruct(field_ptr, visitor, path);
    } else if (info == .array) {
        const E = info.array.child;
        if (E == *Tensor or @typeInfo(E) == .@"struct") {
            for (field_ptr, 0..) |*item, idx| {
                const mark = try pushPath(V, path, "{d}", .{idx});
                defer popPath(V, path, mark);
                try walkElement(item, visitor, path);
            }
        }
    }
}

fn walkElement(item_ptr: anytype, visitor: anytype, path: *Path) !void {
    if (@TypeOf(item_ptr.*) == *Tensor) {
        try visitor.visitTensor(path.slice(), item_ptr.*);
    } else {
        try walkStruct(item_ptr, visitor, path);
    }
}

// ============================================================================
// 构建在 walk 之上的模块操作
// ============================================================================

/// 释放模型持有的全部张量 (参数与缓冲区) 及可变切片字段
pub fn deinitModel(model: anytype, allocator: std.mem.Allocator) void {
    const Visitor = struct {
        allocator: std.mem.Allocator,
        pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
            freePersistentTensor(self.allocator, t);
        }
        pub fn visitOwnedSlice(self: *@This(), s: anytype) !void {
            self.allocator.free(s);
        }
    };
    var v = Visitor{ .allocator = allocator };
    walk(model, &v) catch |err| switch (err) {};
}

/// 将模型内全部张量的梯度清零 (等价于 PyTorch `module.zero_grad()`)
pub fn zeroGradModel(model: anytype) void {
    const Visitor = struct {
        pub fn visitTensor(_: *@This(), _: []const u8, t: *Tensor) !void {
            t.zeroGrad();
        }
    };
    var v = Visitor{};
    walk(model, &v) catch |err| switch (err) {};
}

/// 按字段顺序收集全部可训练参数 (等价于 PyTorch `module.parameters()`)，返回的切片由调用方释放
pub fn parameters(model: anytype, allocator: std.mem.Allocator) ![]*Tensor {
    const Visitor = struct {
        allocator: std.mem.Allocator,
        list: std.ArrayList(*Tensor) = .empty,
        pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
            if (t.requires_grad) try self.list.append(self.allocator, t);
        }
    };
    var v = Visitor{ .allocator = allocator };
    errdefer v.list.deinit(allocator);
    try walk(model, &v);
    return v.list.toOwnedSlice(allocator);
}

/// 带字段路径名称的参数
pub const NamedParameter = struct {
    name: []const u8,
    tensor: *Tensor,
};

/// `namedParameters` 的结果：名称字符串与列表由内部 arena 持有，调用 `deinit` 一并释放
pub const NamedParameterList = struct {
    arena: std.heap.ArenaAllocator,
    items: []NamedParameter,

    pub fn deinit(self: *NamedParameterList) void {
        self.arena.deinit();
    }
};

/// 按字段顺序收集全部可训练参数及其字段路径 (等价于 PyTorch `module.named_parameters()`)；
/// 名称与 `saveModel` 写入的键一致，例如 `cell.w_ih_r.weight`
pub fn namedParameters(model: anytype, allocator: std.mem.Allocator) !NamedParameterList {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const Visitor = struct {
        allocator: std.mem.Allocator,
        list: std.ArrayList(NamedParameter) = .empty,
        pub const wants_path = true;
        pub fn visitTensor(self: *@This(), path: []const u8, t: *Tensor) !void {
            if (!t.requires_grad) return;
            try self.list.append(self.allocator, .{ .name = try self.allocator.dupe(u8, path), .tensor = t });
        }
    };
    var v = Visitor{ .allocator = arena.allocator() };
    try walk(model, &v);
    const items = try v.list.toOwnedSlice(arena.allocator());
    return .{ .arena = arena, .items = items };
}

/// 可训练参数的元素总数
pub fn numParameters(model: anytype) usize {
    const Visitor = struct {
        count: usize = 0,
        pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
            if (t.requires_grad) self.count += t.data.len;
        }
    };
    var v = Visitor{};
    walk(model, &v) catch |err| switch (err) {};
    return v.count;
}

/// 冻结 (`requires_grad = false`) 或解冻 (`true`) 模型 (或其任意子模块) 内的全部参数，等价于 PyTorch `requires_grad_()`。
/// 冻结时释放梯度缓冲；解冻时用 `allocator` (须与创建参数时相同) 分配并清零梯度缓冲。
/// 只作用于参数张量，缓冲区 (`is_buffer`) 保持不变。
pub fn setRequiresGrad(model: anytype, allocator: std.mem.Allocator, requires_grad: bool) !void {
    const Visitor = struct {
        allocator: std.mem.Allocator,
        requires_grad: bool,
        pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
            if (t.is_buffer or t.requires_grad == self.requires_grad) return;
            if (self.requires_grad) {
                const grad = try self.allocator.alloc(f32, t.data.len);
                @memset(grad, 0.0);
                t.grad = grad;
            } else {
                if (t.grad.len > 0) self.allocator.free(t.grad);
                t.grad = &.{};
            }
            t.requires_grad = self.requires_grad;
        }
    };
    var v = Visitor{ .allocator = allocator, .requires_grad = requires_grad };
    try walk(model, &v);
}

/// 递归设置所有含 `training: bool` 字段的模块 (如 `BatchNorm2d`、`Dropout`) 的训练 / 推理状态
pub fn setTrainingModel(model: anytype, is_training: bool) void {
    const Visitor = struct {
        is_training: bool,
        pub fn visitTensor(_: *@This(), _: []const u8, _: *Tensor) !void {}
        pub fn enterModule(self: *@This(), _: []const u8, module: anytype) !bool {
            const M = @TypeOf(module.*);
            if (@hasField(M, "training") and @TypeOf(module.training) == bool) module.training = self.is_training;
            return true;
        }
    };
    var v = Visitor{ .is_training = is_training };
    walk(model, &v) catch |err| switch (err) {};
}

pub fn trainModel(model: anytype) void {
    setTrainingModel(model, true);
}

pub fn evalModel(model: anytype) void {
    setTrainingModel(model, false);
}

/// 递归调用模型中所有定义了 `customInit` 的模块 (由库外用户模块定义)，并将其可训练参数标记为 `is_custom_initialized`。
/// 支持 `fn(self: *Self) void` (确定性) 与 `fn(self: *Self, random: std.Random) void` 两种签名；
/// 定义了 `customInit` 的模块由自身负责其全部子模块，不再向下递归。
pub fn applyCustomInit(model: anytype, random: std.Random) void {
    const Visitor = struct {
        random: std.Random,
        pub fn visitTensor(_: *@This(), _: []const u8, _: *Tensor) !void {}
        pub fn enterModule(self: *@This(), _: []const u8, module: anytype) !bool {
            const M = @TypeOf(module.*);
            if (!@hasDecl(M, "customInit")) return true;
            const params = @typeInfo(@TypeOf(M.customInit)).@"fn".params;
            switch (params.len) {
                1 => module.customInit(),
                2 => module.customInit(self.random),
                else => @compileError(@typeName(M) ++ ".customInit must be fn(self: *Self) void or fn(self: *Self, random: std.Random) void"),
            }
            markCustomInitialized(module);
            return false;
        }
    };
    var v = Visitor{ .random = random };
    walk(model, &v) catch |err| switch (err) {};
}

/// 将模型内所有可训练参数标记为自定义初始化 (`Graph.initWeights` 不会覆盖)
pub fn markCustomInitialized(model: anytype) void {
    const Visitor = struct {
        pub fn visitTensor(_: *@This(), _: []const u8, t: *Tensor) !void {
            if (t.requires_grad) t.is_custom_initialized = true;
        }
    };
    var v = Visitor{};
    walk(model, &v) catch |err| switch (err) {};
}

/// 按字段路径为模型内全部子模块与张量命名 (名称与 PyTorch `named_modules()` / `named_parameters()` 一致)：
/// 含 `name: ?[]const u8` 字段的子模块得到 `root.<字段路径>` (如 `gpt.decoder.h.0.attn`)，张量得到 `root.<字段路径>`
/// (如 `gpt.decoder.h.0.attn.q_attn.weight`)；`root` 为空串时根模块不命名，子模块与张量直接使用字段路径。
/// 模块名决定前向计算时在计算图中进入的作用域，用于可视化导出与初始化报告。
/// 名称字符串由 `allocator` 分配且不单独释放，应传入生命周期不短于模型的 arena (`nn.Module` 自带)；
/// 模型须已位于最终地址 (名称写入模型字段，按值复制模型后应对新副本重新命名)。
pub fn nameModules(model: anytype, allocator: std.mem.Allocator, root: []const u8) !void {
    const Visitor = struct {
        allocator: std.mem.Allocator,
        root: []const u8,
        pub const wants_path = true;

        fn fullPath(self: *@This(), path: []const u8) !?[]const u8 {
            if (path.len == 0) return if (self.root.len == 0) null else try self.allocator.dupe(u8, self.root);
            if (self.root.len == 0) return try self.allocator.dupe(u8, path);
            return try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ self.root, path });
        }

        pub fn enterModule(self: *@This(), path: []const u8, module: anytype) !bool {
            const M = @TypeOf(module.*);
            if (comptime @hasField(M, "name") and @FieldType(M, "name") == ?[]const u8) {
                module.name = try self.fullPath(path);
            }
            return true;
        }

        pub fn visitTensor(self: *@This(), path: []const u8, t: *Tensor) !void {
            t.name = try self.fullPath(path);
        }
    };
    var v = Visitor{ .allocator = allocator, .root = root };
    try walk(model, &v);
}

/// 模块 `forward` 入口：已命名 (`module.name != null`) 时进入以模块名为路径的计算图作用域，登记 `module_type`，
/// 并在模块类型声明了 `pub const formula` 时登记其公式；未命名时返回空守卫。
/// 用法: `const scope = try enterModuleScope(graph, self); defer scope.exit();`
pub fn enterModuleScope(graph: *autodiff.Graph, module: anytype) !autodiff.Graph.ScopeGuard {
    const M = @TypeOf(module.*);
    const guard = try graph.enterModule(module.name, module.module_type);
    if (comptime @hasDecl(M, "formula")) {
        if (module.name) |n| try graph.setModuleFormula(n, M.formula);
    }
    return guard;
}
