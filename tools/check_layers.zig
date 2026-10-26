//! 包级分层依赖与架构隔离检查工具 (Layering & Architectural Isolation Check)
//!
//! 1. 扫描 `src/` 下全部 `.zig` 文件中的 `@import("xxx.zig")`，按所在包归类：
//!    跨包导入只能指向层级更低的包 (单向依赖，零循环导入)。
//! 2. 强制底层门面导入 (Public Facade Enforcement)：
//!    上层包 (Layer >= 1) 导入 Layer 0 (`backend`) 时，仅允许导入公共门面 `backend.zig`，
//!    严禁绕过门面直接导入私有内部文件 (如 `backend/cblas.zig`)。
//! 3. 严禁计算后端鸭子类型 (Duck-Typing Ban in backend)：
//!    `src/backend/` 下的公共函数严禁使用 `anytype` 接收张量类结构体；
//!    严禁在 `src/backend/` 中通过 `@hasField` / `@hasDecl` 探测张量内部字段。

const std = @import("std");

/// 包的层级：数值越小越底层；跨包导入必须满足 `rank(被导入包) < rank(导入方包)`
const Layer = struct {
    name: []const u8,
    rank: u8,
};

const layers = [_]Layer{
    .{ .name = "backend", .rank = 0 },
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

/// 必须通过顶层门面（例如 `backend.zig`）导入、禁止上层直接导入私有内部文件的底层包
fn requiresFacadeImport(package: []const u8) bool {
    return std.mem.eql(u8, package, "backend");
}

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r';
}

fn trimLeading(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len and isWhitespace(s[i])) : (i += 1) {}
    return s[i..];
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

    const tensor_introspect_props = [_][]const u8{
        "device_array",
        "data",
        "shape",
        "isContiguous",
        "numel",
    };

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

        const is_backend_pkg = std.mem.startsWith(u8, rel, "backend/") or std.mem.eql(u8, rel, "backend.zig");
        const is_mlx_c = std.mem.eql(u8, rel, "backend/mlx_c.zig");
        const is_tests_file = std.mem.endsWith(u8, rel, "tests.zig");
        const check_duck_typing = is_backend_pkg and !is_mlx_c and !is_tests_file;

        const source = try src_dir.readFileAlloc(io, entry.path, arena, .limited(16 << 20));
        const dir = std.fs.path.dirnamePosix(rel) orelse "";

        var line_no: usize = 0;
        var lines = std.mem.splitScalar(u8, source, '\n');

        var in_pub_fn = false;
        var fn_start_line: usize = 0;
        var fn_sig_buf: std.ArrayList(u8) = .empty;

        while (lines.next()) |raw_line| {
            line_no += 1;
            const line = trimLeading(raw_line);
            if (std.mem.startsWith(u8, line, "//")) continue;

            // -----------------------------------------------------------------
            // 1. @import 分层及门面检查 (Layering & Facade Enforcement)
            // -----------------------------------------------------------------
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

                // 层级单向依赖检查
                if (to_rank >= from_rank) {
                    std.debug.print("src/{s}:{d}: `{s}` (layer {d}) must not import `{s}` (layer {d}) via \"{s}\"\n", .{
                        rel, line_no, from_package, from_rank, to_package, to_rank, target,
                    });
                    violations += 1;
                    continue;
                }

                // 强制公共门面导入检查 (Higher layers may only import public facade)
                if (requiresFacadeImport(to_package)) {
                    const expected_facade = try std.fmt.allocPrint(arena, "{s}.zig", .{to_package});
                    if (!std.mem.eql(u8, resolved, expected_facade)) {
                        std.debug.print("src/{s}:{d}: `{s}` (layer {d}) must only import public facade `{s}`, not private internal `{s}` (via \"{s}\")\n", .{
                            rel, line_no, from_package, from_rank, expected_facade, resolved, target,
                        });
                        violations += 1;
                    }
                }
            }

            // -----------------------------------------------------------------
            // 2. 计算后端鸭子类型违规检查 (Duck-Typing Ban in backend)
            // -----------------------------------------------------------------
            if (check_duck_typing) {
                // 检查针对张量结构体的内省操作 (@hasField / @hasDecl)
                for (tensor_introspect_props) |prop| {
                    const has_field = std.mem.indexOf(u8, line, "@hasField(") != null;
                    const has_decl = std.mem.indexOf(u8, line, "@hasDecl(") != null;
                    if ((has_field or has_decl) and std.mem.indexOf(u8, line, prop) != null) {
                        std.debug.print("src/{s}:{d}: duck-typing introspection on \"{s}\" is banned in backend\n", .{
                            rel, line_no, prop,
                        });
                        violations += 1;
                    }
                }

                // 检查公共函数声明中的 anytype 逃逸
                if (std.mem.startsWith(u8, line, "pub fn ") or
                    std.mem.startsWith(u8, line, "pub inline fn ") or
                    std.mem.startsWith(u8, line, "pub noinline fn "))
                {
                    in_pub_fn = true;
                    fn_start_line = line_no;
                    fn_sig_buf.clearRetainingCapacity();
                    try fn_sig_buf.appendSlice(arena, line);
                } else if (in_pub_fn) {
                    try fn_sig_buf.append(arena, ' ');
                    try fn_sig_buf.appendSlice(arena, line);
                }

                if (in_pub_fn) {
                    if (std.mem.indexOfScalar(u8, line, ')') != null or
                        std.mem.indexOfScalar(u8, line, '{') != null or
                        std.mem.indexOfScalar(u8, line, ';') != null)
                    {
                        const sig = fn_sig_buf.items;
                        if (std.mem.indexOf(u8, sig, "anytype") != null) {
                            if (std.mem.indexOfScalar(u8, sig, '(')) |lparen| {
                                if (std.mem.lastIndexOfScalar(u8, sig, ')')) |rparen| {
                                    if (rparen > lparen) {
                                        const params = sig[lparen + 1 .. rparen];
                                        var p_iter = std.mem.splitScalar(u8, params, ',');
                                        while (p_iter.next()) |raw_p| {
                                            const p = trimLeading(raw_p);
                                            if (std.mem.indexOf(u8, p, "anytype") != null) {
                                                // 允许编译期泛型函数指针 (如 mlx.zig 中的 comptime func: anytype)
                                                if (std.mem.startsWith(u8, p, "comptime ")) continue;
                                                // 允许格式化输出 (device.zig 中的 writer: anytype)
                                                if (std.mem.startsWith(u8, p, "writer:")) continue;
                                                // 允许执行器通用闭包上下文 (executor.zig 中的 ctx: anytype)
                                                if (std.mem.eql(u8, rel, "backend/executor.zig") and std.mem.startsWith(u8, p, "ctx:")) continue;

                                                std.debug.print("src/{s}:{d}: banned duck-typed `anytype` parameter in public function `{s}`: `{s}`\n", .{
                                                    rel, fn_start_line, sig[0..@min(sig.len, 80)], p,
                                                });
                                                violations += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        in_pub_fn = false;
                        fn_sig_buf.clearRetainingCapacity();
                    }
                }
            }
        }
    }

    if (violations > 0) {
        std.debug.print("check-layers: {d} violation(s) in {d} files\n", .{ violations, file_count });
        std.process.exit(1);
    }
    std.debug.print("check-layers: {d} files, all package dependencies are one-way, facades enforced, zero duck-typing\n", .{file_count});
}
