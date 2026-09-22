const std = @import("std");
const builtin = @import("builtin");
const zig_ml = @import("zig_ml");
const bench = zig_ml.bench;

fn printHelp() void {
    std.debug.print(
        \\Usage: benchmark [options] [filter]
        \\
        \\Options:
        \\  -h, --help               Show this help message
        \\  -f, --filter <pattern>   Run benchmarks matching pattern (case-insensitive substring)
        \\  -s, --suite <name>       Run specific suite (gemm, ops, activations, layers, models, optimizers, tokenizer, all)
        \\  -i, --iter <count>       Number of measurement iterations (default: 10)
        \\  -w, --warmup <count>     Number of warmup iterations (default: 3)
        \\  -q, --quiet              Only print the final summary table
        \\
        \\Examples:
        \\  zig build bench
        \\  zig build bench -- --filter gemm
        \\  zig build bench -- --suite models
        \\  zig build bench -- -i 20 -w 5
        \\  just bench --filter conv
        \\
    , .{});
}

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var config = bench.BenchmarkConfig{
        .warmup = 3,
        .iterations = 10,
        .filter = null,
        .suite = null,
        .quiet = false,
    };

    var iter = try init.minimal.args.iterateAllocator(allocator);
    defer iter.deinit();
    _ = iter.next(); // Skip executable path

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--filter") or std.mem.eql(u8, arg, "-f")) {
            config.filter = iter.next();
        } else if (std.mem.eql(u8, arg, "--suite") or std.mem.eql(u8, arg, "-s")) {
            config.suite = iter.next();
        } else if (std.mem.eql(u8, arg, "--iter") or std.mem.eql(u8, arg, "-i")) {
            if (iter.next()) |val| {
                config.iterations = std.fmt.parseInt(usize, val, 10) catch 10;
            }
        } else if (std.mem.eql(u8, arg, "--warmup") or std.mem.eql(u8, arg, "-w")) {
            if (iter.next()) |val| {
                config.warmup = std.fmt.parseInt(usize, val, 10) catch 3;
            }
        } else if (std.mem.eql(u8, arg, "--quiet") or std.mem.eql(u8, arg, "-q")) {
            config.quiet = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            config.filter = arg;
        }
    }

    const blas_backend = if (builtin.os.tag == .macos or builtin.os.tag == .ios)
        "macOS Accelerate Framework (AMX CBLAS)"
    else
        "Zig 8-way SIMD Fallback Engine";

    std.debug.print("======================================================================================================\n", .{});
    std.debug.print("                       znn (Zig Neural Network) - Performance Benchmark Suite                         \n", .{});
    std.debug.print("======================================================================================================\n", .{});
    std.debug.print("  Target Platform : {s} ({s})\n", .{ @tagName(builtin.os.tag), @tagName(builtin.cpu.arch) });
    std.debug.print("  Build Mode      : {s}\n", .{ @tagName(builtin.mode) });
    std.debug.print("  BLAS Backend    : {s}\n", .{ blas_backend });
    std.debug.print("  Config          : Warmup={d}, Iterations={d}", .{ config.warmup, config.iterations });
    if (config.suite) |s| {
        std.debug.print(", Suite='{s}'", .{s});
    }
    if (config.filter) |f| {
        std.debug.print(", Filter='{s}'", .{f});
    }
    std.debug.print("\n======================================================================================================\n\n", .{});

    var runner = bench.BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    if (!config.quiet) {
        std.debug.print("Running benchmarks...\n", .{});
    }

    const t_start = bench.getTimeNs();

    try bench.runAllBenchmarks(&runner, allocator);

    const t_end = bench.getTimeNs();
    const total_secs = @as(f64, @floatFromInt(t_end - t_start)) / 1_000_000_000.0;

    runner.printSummary();

    std.debug.print("Total benchmark execution time: {d:.2}s\n", .{total_secs});
}
