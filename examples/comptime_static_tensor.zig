const std = @import("std");

/// 编译期静态形状张量类型发生器 (Compile-Time Statically Shaped Tensor)
pub fn StaticTensor(comptime S: []const usize) type {
    return struct {
        data: [computeTotalElements(S)]f32,

        pub const shape = S;
        const Self = @This();

        /// 初始化张量（填充指定初值）
        pub fn init(initial_value: f32) Self {
            var self: Self = undefined;
            @memset(&self.data, initial_value);
            return self;
        }

        /// 从已有数组切片初始化
        pub fn fromSlice(slice: []const f32) Self {
            std.debug.assert(slice.len == computeTotalElements(S));
            var self: Self = undefined;
            @memcpy(&self.data, slice);
            return self;
        }

        /// 编译期静态安全矩阵乘法
        pub fn matmul(self: Self, other: anytype) StaticTensor(&.{ S[0], @TypeOf(other).shape[1] }) {
            const other_shape = @TypeOf(other).shape;
            // 1. 编译期强制检查维度合法性
            comptime {
                if (S.len != 2 or other_shape.len != 2) {
                    @compileError("matmul currently requires 2D matrices");
                }
                if (S[1] != other_shape[0]) {
                    @compileError(std.fmt.comptimePrint(
                        "❌ Matmul dimension mismatch: cannot multiply matrix [{d}, {d}] with [{d}, {d}]! (Inner dimensions {d} != {d})",
                        .{ S[0], S[1], other_shape[0], other_shape[1], S[1], other_shape[0] },
                    ));
                }
            }

            // 2. 运行时零额外开销的高性能纯矩阵乘法
            var result = StaticTensor(&.{ S[0], other_shape[1] }).init(0.0);
            const M = S[0];
            const K = S[1];
            const N = other_shape[1];

            for (0..M) |i| {
                for (0..K) |p| {
                    const a_val = self.data[i * K + p];
                    if (a_val == 0.0) continue;
                    for (0..N) |j| {
                        result.data[i * N + j] += a_val * other.data[p * N + j];
                    }
                }
            }
            return result;
        }

        /// 格式化打印张量
        pub fn print(self: Self) void {
            if (S.len == 2) {
                const rows = S[0];
                const cols = S[1];
                std.debug.print("StaticTensor [{d}, {d}]:\n", .{ rows, cols });
                for (0..rows) |r| {
                    std.debug.print("  [", .{});
                    for (0..cols) |c| {
                        std.debug.print("{d:6.2}", .{self.data[r * cols + c]});
                        if (c + 1 < cols) std.debug.print(", ", .{});
                    }
                    std.debug.print("]\n", .{});
                }
            } else {
                std.debug.print("StaticTensor {any}: data len={d}\n", .{ S, self.data.len });
            }
        }
    };
}

fn computeTotalElements(comptime dims: []const usize) usize {
    var count: usize = 1;
    for (dims) |d| count *= d;
    return count;
}

pub fn main() !void {
    std.debug.print("=== Zig Comptime Static Tensor Demo ===\n\n", .{});

    // 1. 初始化两个静态 2D 矩阵
    const A = StaticTensor(&.{ 2, 3 }).fromSlice(&.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });

    const B = StaticTensor(&.{ 3, 2 }).fromSlice(&.{
        7.0,  8.0,
        9.0,  1.0,
        2.0,  3.0,
    });

    std.debug.print("Matrix A:\n", .{});
    A.print();

    std.debug.print("\nMatrix B:\n", .{});
    B.print();

    // 2. 编译期验证并计算矩阵乘法 C = A @ B
    // A: [2, 3] @ B: [3, 2] -> C: [2, 2]
    const C = A.matmul(B);

    std.debug.print("\nResult Matrix C = A @ B (Shape [2, 2]):\n", .{});
    C.print();

    // 验证计算结果：
    // C[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    std.debug.assert(C.data[0] == 31.0);
    std.debug.assert(C.data[1] == 19.0);
    std.debug.assert(C.data[2] == 85.0);
    std.debug.assert(C.data[3] == 55.0);

    std.debug.print("\n✅ Verification passed! Matmul computed with zero runtime allocation.\n", .{});
}
