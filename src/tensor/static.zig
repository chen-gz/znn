const std = @import("std");
const types_mod = @import("types.zig");
const core_mod = @import("core.zig");

fn normalizeDims(comptime dims: anytype) []const usize {
    const T = @TypeOf(dims);
    const info = @typeInfo(T);
    switch (info) {
        .pointer => |ptr| {
            if (ptr.size == .slice and ptr.child == usize) {
                return dims;
            }
            if (ptr.size == .one) {
                const child_info = @typeInfo(ptr.child);
                if (child_info == .array) {
                    const arr_len = child_info.array.len;
                    var result: [arr_len]usize = undefined;
                    for (0..arr_len) |i| {
                        result[i] = @intCast(dims[i]);
                    }
                    const final = result;
                    return &final;
                }
                if (child_info == .@"struct" and child_info.@"struct".is_tuple) {
                    const fields = child_info.@"struct".fields;
                    var result: [fields.len]usize = undefined;
                    inline for (fields, 0..) |f, i| {
                        result[i] = @intCast(@field(dims.*, f.name));
                    }
                    const final = result;
                    return &final;
                }
            }
            @compileError("StaticTensor dims must be a []const usize slice, array, or tuple of integers");
        },
        .array => |arr| {
            var result: [arr.len]usize = undefined;
            for (0..arr.len) |i| {
                result[i] = @intCast(dims[i]);
            }
            const final = result;
            return &final;
        },
        .@"struct" => |st| {
            if (!st.is_tuple) {
                @compileError("StaticTensor dims struct must be a tuple of integers");
            }
            const fields = st.fields;
            var result: [fields.len]usize = undefined;
            inline for (fields, 0..) |f, i| {
                result[i] = @intCast(@field(dims, f.name));
            }
            const final = result;
            return &final;
        },
        else => @compileError("StaticTensor dims must be a []const usize slice, array, or tuple of integers"),
    }
}

pub fn computeTotalElements(comptime dims: []const usize) usize {
    var count: usize = 1;
    for (dims) |d| count *= d;
    return count;
}

/// 编译期静态形状张量类型发生器 (Compile-Time Statically Shaped Tensor)
pub fn StaticTensor(comptime ElemT: type, comptime dims: anytype) type {
    const S = normalizeDims(dims);
    const total_elements = computeTotalElements(S);

    return struct {
        data: [total_elements]ElemT,

        pub const ElemType = ElemT;
        pub const shape: []const usize = S;
        pub const rank: usize = S.len;
        const Self = @This();

        /// 返回总元素数
        pub fn numel() usize {
            return total_elements;
        }

        /// 初始化张量（填充指定初值）
        pub fn init(initial_value: ElemT) Self {
            var self: Self = undefined;
            @memset(&self.data, initial_value);
            return self;
        }

        /// 全零静态张量
        pub fn zeros() Self {
            return init(types_mod.convertScalar(ElemT, usize, 0));
        }

        /// 全一静态张量
        pub fn ones() Self {
            return init(types_mod.convertScalar(ElemT, usize, 1));
        }

        /// 从已有数组切片初始化
        pub fn fromSlice(slice: []const ElemT) Self {
            std.debug.assert(slice.len == total_elements);
            var self: Self = undefined;
            @memcpy(&self.data, slice);
            return self;
        }

        /// 计算多维索引对应的扁平化下标
        pub fn flatIndex(indices: [S.len]usize) usize {
            var idx: usize = 0;
            var stride: usize = 1;
            var d: usize = S.len;
            while (d > 0) {
                d -= 1;
                std.debug.assert(indices[d] < S[d]);
                idx += indices[d] * stride;
                stride *= S[d];
            }
            return idx;
        }

        pub fn get(self: Self, indices: [S.len]usize) ElemT {
            return self.data[flatIndex(indices)];
        }

        pub fn set(self: *Self, indices: [S.len]usize, val: ElemT) void {
            self.data[flatIndex(indices)] = val;
        }

        /// 逐元素加法（编译期同形状保证）
        pub fn add(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (&out.data, self.data, other.data) |*dst, a, b| {
                dst.* = a + b;
            }
            return out;
        }

        /// 逐元素减法（编译期同形状保证）
        pub fn sub(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (&out.data, self.data, other.data) |*dst, a, b| {
                dst.* = a - b;
            }
            return out;
        }

        /// 逐元素乘法（编译期同形状保证）
        pub fn mul(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (&out.data, self.data, other.data) |*dst, a, b| {
                dst.* = a * b;
            }
            return out;
        }

        /// 标量乘法
        pub fn mulScalar(self: Self, scalar: ElemT) Self {
            var out: Self = undefined;
            for (&out.data, self.data) |*dst, a| {
                dst.* = a * scalar;
            }
            return out;
        }

        /// 编译期零开销形状重塑
        pub fn reshape(self: Self, comptime new_dims: anytype) StaticTensor(ElemT, new_dims) {
            const Target = StaticTensor(ElemT, new_dims);
            comptime {
                if (Target.numel() != total_elements) {
                    @compileError(std.fmt.comptimePrint(
                        "StaticTensor.reshape element count mismatch: {d} != {d}",
                        .{ total_elements, Target.numel() },
                    ));
                }
            }
            return Target{ .data = self.data };
        }

        /// 编译期 2D 矩阵转置
        pub fn transpose(self: Self) StaticTensor(ElemT, &.{ S[1], S[0] }) {
            comptime {
                if (S.len != 2) {
                    @compileError("StaticTensor.transpose requires a 2D tensor");
                }
            }
            const rows = S[0];
            const cols = S[1];
            var out: StaticTensor(ElemT, &.{ cols, rows }) = undefined;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    out.data[c * rows + r] = self.data[r * cols + c];
                }
            }
            return out;
        }

        /// 编译期静态安全矩阵乘法
        pub fn matmul(self: Self, other: anytype) StaticTensor(ElemT, &.{ S[0], @TypeOf(other).shape[1] }) {
            const OtherT = @TypeOf(other);
            const other_shape = OtherT.shape;
            comptime {
                if (OtherT.ElemType != ElemT) {
                    @compileError("StaticTensor.matmul requires matching element types");
                }
                if (S.len != 2 or other_shape.len != 2) {
                    @compileError("matmul currently requires 2D matrices");
                }
                if (S[1] != other_shape[0]) {
                    @compileError(std.fmt.comptimePrint(
                        "Matmul dimension mismatch: cannot multiply matrix [{d}, {d}] with [{d}, {d}]! (Inner dimensions {d} != {d})",
                        .{ S[0], S[1], other_shape[0], other_shape[1], S[1], other_shape[0] },
                    ));
                }
            }

            var result = StaticTensor(ElemT, &.{ S[0], other_shape[1] }).zeros();
            const M = S[0];
            const K = S[1];
            const N = other_shape[1];
            const zero_val = types_mod.convertScalar(ElemT, usize, 0);

            for (0..M) |i| {
                for (0..K) |p| {
                    const a_val = self.data[i * K + p];
                    if (a_val == zero_val) continue;
                    for (0..N) |j| {
                        result.data[i * N + j] += a_val * other.data[p * N + j];
                    }
                }
            }
            return result;
        }

        /// 导出为动态 GenericTensor(ElemT)
        pub fn toGeneric(self: *const Self, allocator: std.mem.Allocator) !*types_mod.GenericTensor(ElemT) {
            return types_mod.GenericTensor(ElemT).fromSlice(allocator, S, &self.data);
        }

        /// 导出为动态 Autograd Tensor (f32)
        pub fn toDynamic(self: *const Self, allocator: std.mem.Allocator) !*core_mod.Tensor {
            if (ElemT == f32) {
                return core_mod.array(allocator, S, &self.data);
            }
            const gen = try self.toGeneric(allocator);
            defer gen.deinit(allocator);
            return core_mod.Tensor.fromGeneric(ElemT, gen, allocator);
        }

        /// 格式化打印张量
        pub fn print(self: Self) void {
            if (S.len == 2) {
                const rows = S[0];
                const cols = S[1];
                std.debug.print("StaticTensor({s}) [{d}, {d}]:\n", .{ @typeName(ElemT), rows, cols });
                for (0..rows) |r| {
                    std.debug.print("  [", .{});
                    for (0..cols) |c| {
                        if (@typeInfo(ElemT) == .float) {
                            std.debug.print("{d:6.2}", .{self.data[r * cols + c]});
                        } else {
                            std.debug.print("{any}", .{self.data[r * cols + c]});
                        }
                        if (c + 1 < cols) std.debug.print(", ", .{});
                    }
                    std.debug.print("]\n", .{});
                }
            } else {
                std.debug.print("StaticTensor({s}) {any}: data len={d}\n", .{ @typeName(ElemT), S, self.data.len });
            }
        }
    };
}
