const std = @import("std");
const zig_ml = @import("zig_ml");
const StaticTensor = zig_ml.tensor.StaticTensor;

pub fn main() !void {
    std.debug.print("=== Zig Comptime Static Tensor Demo ===\n\n", .{});

    // 1. 初始化两个静态 2D 矩阵
    const A = StaticTensor(f32, &.{ 2, 3 }).fromSlice(&.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });

    const B = StaticTensor(f32, &.{ 3, 2 }).fromSlice(&.{
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0,
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

    std.debug.print("\nVerification passed! Matmul computed with zero runtime allocation.\n", .{});
}
