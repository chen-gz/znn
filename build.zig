const std = @import("std");

const ExampleTarget = struct {
    name: []const u8,
    src: []const u8,
    run_step: []const u8,
    run_desc: []const u8,
    alias_step: ?[]const u8 = null,
    alias_desc: ?[]const u8 = null,
    forward_args: bool = false,
    test_name: ?[]const u8 = null,
};

const examples = [_]ExampleTarget{
    .{
        .name = "zig_ml",
        .src = "examples/fashion_mnist.zig",
        .run_step = "run",
        .run_desc = "Run the MLP Fashion MNIST app",
        .forward_args = true,
        .test_name = "exe_tests",
    },
    .{
        .name = "linear_regression",
        .src = "examples/linear_regression.zig",
        .run_step = "run-lr",
        .run_desc = "Run the linear regression app",
    },
    .{
        .name = "logistic_regression",
        .src = "examples/logistic_regression.zig",
        .run_step = "run-logr",
        .run_desc = "Run the logistic regression app",
    },
    .{
        .name = "ridge_regression",
        .src = "examples/ridge_regression.zig",
        .run_step = "run-ridge",
        .run_desc = "Run the ridge regression app",
    },
    .{
        .name = "cnn",
        .src = "examples/cnn.zig",
        .run_step = "run-cnn",
        .run_desc = "Run the CNN Fashion MNIST app",
        .test_name = "cnn_tests",
    },
    .{
        .name = "transformer_embedding",
        .src = "examples/transformer_embedding.zig",
        .run_step = "run-emb",
        .run_desc = "Run the Transformer Embedding example",
    },
    .{
        .name = "transformer_attention",
        .src = "examples/transformer_attention.zig",
        .run_step = "run-att",
        .run_desc = "Run the Transformer Attention example",
    },
    .{
        .name = "transformer_block",
        .src = "examples/transformer_block.zig",
        .run_step = "run-block",
        .run_desc = "Run the Transformer Block example",
    },
    .{
        .name = "transformer_gpt",
        .src = "examples/transformer_gpt.zig",
        .run_step = "run-gpt",
        .run_desc = "Run the Transformer GPT example",
    },
    .{
        .name = "export_model_report",
        .src = "examples/export_model_report.zig",
        .run_step = "run-report",
        .run_desc = "Run the Model Graph JSON export example",
    },
    .{
        .name = "export_book_models",
        .src = "examples/export_book_models.zig",
        .run_step = "run-book-models",
        .run_desc = "Export all book model architecture graphs to JSON",
    },
    .{
        .name = "llm_training",
        .src = "examples/llm_training.zig",
        .run_step = "run-llm",
        .run_desc = "Run the End-to-End LLM Pipeline example",
    },
    .{
        .name = "train_shakespeare",
        .src = "examples/train_shakespeare.zig",
        .run_step = "run-shakespeare",
        .run_desc = "Run the TinyShakespeare GPT training and generation example",
    },
    .{
        .name = "gan",
        .src = "examples/gan.zig",
        .run_step = "run-gan",
        .run_desc = "Run the Generative Adversarial Network (GAN) example",
    },
    .{
        .name = "regularized_regression",
        .src = "examples/regularized_regression.zig",
        .run_step = "run-reg",
        .run_desc = "Run the Regularized Regression (Ridge, Lasso, Elastic Net) example",
    },
    .{
        .name = "cross_validation",
        .src = "examples/cross_validation.zig",
        .run_step = "run-cv",
        .run_desc = "Run the 5-Fold Cross-Validation hyperparameter tuning example",
    },
    .{
        .name = "benchmark",
        .src = "examples/benchmark.zig",
        .run_step = "run-bench",
        .run_desc = "Run the znn performance benchmark suite",
        .alias_step = "bench",
        .alias_desc = "Run the znn performance benchmark suite (alias for run-bench)",
        .forward_args = true,
        .test_name = "bench_tests",
    },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zig_ml", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .link_libc = true,
    });
    if (target.result.os.tag == .macos) {
        mod.linkFramework("Accelerate", .{});
    }

    const test_step = b.step("test", "Run tests");

    const mod_tests = b.addTest(.{
        .root_module = mod,
        .name = "root_tests",
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const install_mod_tests = b.addInstallArtifact(mod_tests, .{});
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&install_mod_tests.step);

    for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(ex.src),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "zig_ml", .module = mod },
                },
            }),
        });
        if (target.result.os.tag == .macos) {
            exe.root_module.linkFramework("Accelerate", .{});
        }
        b.installArtifact(exe);

        const run_step = b.step(ex.run_step, ex.run_desc);
        const run_cmd = b.addRunArtifact(exe);
        run_step.dependOn(&run_cmd.step);
        run_cmd.step.dependOn(b.getInstallStep());
        if (ex.forward_args) {
            if (b.args) |args| {
                run_cmd.addArgs(args);
            }
        }

        if (ex.alias_step) |alias_name| {
            const alias = b.step(alias_name, ex.alias_desc orelse ex.run_desc);
            alias.dependOn(&run_cmd.step);
        }

        if (ex.test_name) |tname| {
            const exe_test = b.addTest(.{
                .root_module = exe.root_module,
                .name = tname,
            });
            const run_exe_test = b.addRunArtifact(exe_test);
            const install_exe_test = b.addInstallArtifact(exe_test, .{});
            test_step.dependOn(&run_exe_test.step);
            test_step.dependOn(&install_exe_test.step);
        }
    }

    // Code coverage step using kcov
    const coverage_step = b.step("coverage", "Generate and display test code coverage report using kcov");
    const coverage_cmd = b.addSystemCommand(&.{ "bash", "scripts/coverage.sh" });
    coverage_cmd.step.dependOn(test_step);
    if (b.args) |args| {
        coverage_cmd.addArgs(args);
    }
    coverage_step.dependOn(&coverage_cmd.step);

    // Dataset Download Step (Pure Zig: zig build download-dataset -- tinyshakespeare)
    const dataset_opt = b.option([]const u8, "dataset", "Dataset name to download (fashion_mnist, mnist, tinyshakespeare, wikitext2, tinystories, alpaca, all_llm)") orelse "fashion_mnist";

    const exe_download = b.addExecutable(.{
        .name = "download_data",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/download_data.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    if (target.result.os.tag == .macos) {
        exe_download.root_module.linkFramework("Accelerate", .{});
    }

    const download_cmd = b.addRunArtifact(exe_download);
    if (b.args) |args| {
        download_cmd.addArgs(args);
    } else {
        download_cmd.addArg(dataset_opt);
    }

    const download_step = b.step("download-dataset", "Download dataset in pure Zig (e.g. tinyshakespeare, wikitext2, tinystories, alpaca, fashion_mnist, mnist)");
    download_step.dependOn(&download_cmd.step);

    const download_data_step = b.step("download-data", "Alias for download-dataset");
    download_data_step.dependOn(&download_cmd.step);
}
