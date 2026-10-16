//! 包级分层依赖检查工具 (Layering Check)：`zig build check-layers`（同时作为 `zig build test` 的一部分运行）。
//!
//! 扫描 `src/` 下全部 `.zig` 文件中的 `@import("xxx.zig")`，按所在包（`src/<包>/...` 或门面文件 `src/<包>.zig`）
//! 归类，并要求跨包导入只能指向层级更低的包，从而保证包之间是单向依赖、不存在循环导入。
//! 同一包内的文件互相导入不受限制（例如 `Tensor` / `Graph` 的方法实现分布在同包的多个文件中）。
const std = @import("std");

/// 包的层级：数值越小越底层；跨包导入必须满足 `rank(被导入包) < rank(导入方包)`
const Layer = struct {
    name: []const u8,
    rank: u8,
};

const layers = [_]Layer{
    .{ .name = "cblas", .rank = 0 },
    .{ .name = "tensor", .rank = 1 },
    .{ .name = "dataset", .rank = 1 },
    .{ .name = "autodiff", .rank = 2 },
    .{ .name = "regression", .rank = 2 },
    .{ .name = "manifold", .rank = 2 },
    .{ .name = "nn", .rank = 3 },
    .{ .name = "cross_validation", .rank = 3 },
    .{ .name = "optim", .rank = 4 },
    .{ .name = "models", .rank = 4 },
    .{ .name = "engine", .rank = 5 },
    .{ .name = "bench", .rank = 6 },
    .{ .name = "root", .rank = 7 },
};

fn rankOf(package: []const u8) ?u8 {
    for (layers) |layer| {
        if (std.mem.eql(u8, layer.name, package)) return layer.rank;
    }
    return null;
}

/// 由相对 `src/` 的路径（`/` 分隔）推导所属包：子目录取首段目录名，顶层文件取去掉 `.zig` 的文件名；
/// 顶层测试汇总文件 `tests.zig` 归属 `root` 包
fn packageOf(rel_path: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, rel_path, '/')) |slash| return rel_path[0..slash];
    const stem = rel_path[0 .. rel_path.len - ".zig".len];
    if (std.mem.eql(u8, stem, "tests")) return "root";
    return stem;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 2) {
        std.debug.print("usage: check_layers <src-dir>\n", .{});
        std.process.exit(2);
    }

    var src_dir = try std.Io.Dir.cwd().openDir(io, args[1], .{ .iterate = true });
    defer src_dir.close(io);

    var walker = try src_dir.walk(arena);
    defer walker.deinit();

    var file_count: usize = 0;
    var violations: usize = 0;

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;
        file_count += 1;

        const rel = try arena.dupe(u8, entry.path);
        std.mem.replaceScalar(u8, rel, '\\', '/');
        const from_package = packageOf(rel);
        const from_rank = rankOf(from_package) orelse {
            std.debug.print("src/{s}: package `{s}` is not registered in tools/check_layers.zig\n", .{ rel, from_package });
            violations += 1;
            continue;
        };

        const source = try src_dir.readFileAlloc(io, entry.path, arena, .limited(16 << 20));
        const dir = std.fs.path.dirnamePosix(rel) orelse "";

        var line_no: usize = 0;
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            line_no += 1;
            if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " \t"), "//")) continue;

            var rest = line;
            while (std.mem.indexOf(u8, rest, "@import(\"")) |pos| {
                const after = rest[pos + "@import(\"".len ..];
                const end = std.mem.indexOfScalar(u8, after, '"') orelse break;
                const target = after[0..end];
                rest = after[end..];
                if (!std.mem.endsWith(u8, target, ".zig")) continue; // std / builtin / 具名模块

                const resolved = try std.fs.path.resolvePosix(arena, &.{ dir, target });
                if (std.mem.startsWith(u8, resolved, "..")) {
                    std.debug.print("src/{s}:{d}: import `{s}` escapes src/\n", .{ rel, line_no, target });
                    violations += 1;
                    continue;
                }

                const to_package = packageOf(resolved);
                if (std.mem.eql(u8, to_package, from_package)) continue;
                const to_rank = rankOf(to_package) orelse {
                    std.debug.print("src/{s}:{d}: package `{s}` is not registered in tools/check_layers.zig\n", .{ rel, line_no, to_package });
                    violations += 1;
                    continue;
                };
                if (to_rank >= from_rank) {
                    std.debug.print("src/{s}:{d}: `{s}` (layer {d}) must not import `{s}` (layer {d}) via \"{s}\"\n", .{
                        rel, line_no, from_package, from_rank, to_package, to_rank, target,
                    });
                    violations += 1;
                }
            }
        }
    }

    if (violations > 0) {
        std.debug.print("check-layers: {d} violation(s) in {d} files\n", .{ violations, file_count });
        std.process.exit(1);
    }
    std.debug.print("check-layers: {d} files, all package dependencies are one-way\n", .{file_count});
}
