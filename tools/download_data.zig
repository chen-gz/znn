const std = @import("std");

fn downloadFile(
    client: *std.http.Client,
    io: std.Io,
    allocator: std.mem.Allocator,
    url: []const u8,
    target_path: []const u8,
    is_gzip: bool,
) !void {
    const cwd = std.Io.Dir.cwd();

    // Check if target already exists
    if (cwd.openFile(io, target_path, .{})) |f| {
        var existing = f;
        defer existing.close(io);
        const len = existing.length(io) catch 0;
        if (len > 0) {
            std.debug.print("  ✅ Already exists: {s} ({} bytes)\n", .{ target_path, len });
            return;
        }
    } else |_| {}

    std.debug.print("  📥 Downloading from {s} ...\n", .{url});

    if (is_gzip) {
        // 1. Download to temporary .gz buffer/file
        var temp_gz_path_buf: [256]u8 = undefined;
        const temp_gz_path = try std.fmt.bufPrint(&temp_gz_path_buf, "{s}.gz.tmp", .{target_path});

        var gz_file = try cwd.createFile(io, temp_gz_path, .{});
        var gz_write_buf: [8192]u8 = undefined;
        var gz_writer = gz_file.writer(io, &gz_write_buf);

        const fetch_res = try client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &gz_writer.interface,
        });
        try gz_writer.interface.flush();
        gz_file.close(io);

        if (fetch_res.status != .ok) {
            _ = cwd.deleteFile(io, temp_gz_path) catch {};
            std.debug.print("  ❌ Download failed with HTTP status: {}\n", .{fetch_res.status});
            return error.DownloadFailed;
        }

        // 2. Decompress .gz directly in pure Zig using std.compress.flate
        std.debug.print("  📦 Decompressing {s} -> {s} ...\n", .{ temp_gz_path, target_path });
        var read_gz = try cwd.openFile(io, temp_gz_path, .{});
        defer read_gz.close(io);
        defer _ = cwd.deleteFile(io, temp_gz_path) catch {};

        var file_buf: [8192]u8 = undefined;
        var file_reader = read_gz.reader(io, &file_buf);

        var window: [std.compress.flate.max_window_len]u8 = undefined;
        var decompressor = std.compress.flate.Decompress.init(&file_reader.interface, .gzip, &window);

        var out_file = try cwd.createFile(io, target_path, .{});
        defer out_file.close(io);

        var write_buf: [8192]u8 = undefined;
        var out_writer = out_file.writer(io, &write_buf);

        _ = try decompressor.reader.streamRemaining(&out_writer.interface);
        try out_writer.interface.flush();

        const final_len = try out_file.length(io);
        std.debug.print("  🎉 Extracted {s} ({} bytes)\n", .{ target_path, final_len });
    } else {
        var out_file = try cwd.createFile(io, target_path, .{});
        defer out_file.close(io);

        var write_buf: [8192]u8 = undefined;
        var out_writer = out_file.writer(io, &write_buf);

        const fetch_res = try client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &out_writer.interface,
        });
        try out_writer.interface.flush();

        if (fetch_res.status != .ok) {
            std.debug.print("  ❌ Download failed with HTTP status: {}\n", .{fetch_res.status});
            return error.DownloadFailed;
        }

        const final_len = try out_file.length(io);
        std.debug.print("  🎉 Saved {s} ({} bytes)\n", .{ target_path, final_len });
    }
    _ = allocator;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    // Ensure data/ directory exists
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, "data", .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    var iter = try init.minimal.args.iterateAllocator(allocator);
    defer iter.deinit();
    _ = iter.next(); // Skip executable path
    const dataset_name: []const u8 = iter.next() orelse "fashion_mnist";

    std.debug.print("=========================================================\n", .{});
    std.debug.print("   Pure Zig Dataset Downloader (znn dataset tool)        \n", .{});
    std.debug.print("=========================================================\n\n", .{});

    if (std.mem.eql(u8, dataset_name, "fashion_mnist") or std.mem.eql(u8, dataset_name, "fashion-mnist")) {
        std.debug.print("⬇️  Downloading Fashion MNIST dataset...\n", .{});
        const base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com";
        const files = [_][]const u8{
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        };
        for (files) |file| {
            var url_buf: [256]u8 = undefined;
            const url = try std.fmt.bufPrint(&url_buf, "{s}/{s}.gz", .{ base_url, file });
            var path_buf: [256]u8 = undefined;
            const path = try std.fmt.bufPrint(&path_buf, "data/{s}", .{file});
            try downloadFile(&client, io, allocator, url, path, true);
        }
        std.debug.print("✨ Fashion MNIST dataset is ready in data/!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "mnist")) {
        std.debug.print("⬇️  Downloading classic MNIST dataset...\n", .{});
        const base_url = "https://storage.googleapis.com/cvdf-datasets/mnist";
        const files = [_][]const u8{
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        };
        for (files) |file| {
            var url_buf: [256]u8 = undefined;
            const url = try std.fmt.bufPrint(&url_buf, "{s}/{s}.gz", .{ base_url, file });
            var path_buf: [256]u8 = undefined;
            const path = try std.fmt.bufPrint(&path_buf, "data/{s}", .{file});
            try downloadFile(&client, io, allocator, url, path, true);
        }
        std.debug.print("✨ MNIST dataset is ready in data/!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "tinyshakespeare") or std.mem.eql(u8, dataset_name, "shakespeare") or std.mem.eql(u8, dataset_name, "tiny_shakespeare")) {
        std.debug.print("⬇️  Downloading TinyShakespeare dataset (~1.1MB pure text)...\n", .{});
        const url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        try downloadFile(&client, io, allocator, url, "data/tinyshakespeare.txt", false);
        std.debug.print("✨ TinyShakespeare dataset is ready in data/tinyshakespeare.txt!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "wikitext2") or std.mem.eql(u8, dataset_name, "wikitext-2") or std.mem.eql(u8, dataset_name, "wikitext")) {
        std.debug.print("⬇️  Downloading WikiText-2 dataset (train/valid/test)...\n", .{});
        cwd.createDir(io, "data/wikitext-2", .default_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        const base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2";
        const files = [_][]const u8{ "train.txt", "valid.txt", "test.txt" };
        for (files) |f| {
            var url_buf: [256]u8 = undefined;
            const url = try std.fmt.bufPrint(&url_buf, "{s}/{s}", .{ base_url, f });
            var path_buf: [256]u8 = undefined;
            const path = try std.fmt.bufPrint(&path_buf, "data/wikitext-2/{s}", .{f});
            try downloadFile(&client, io, allocator, url, path, false);
        }
        std.debug.print("✨ WikiText-2 dataset is ready in data/wikitext-2/!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "tinystories") or std.mem.eql(u8, dataset_name, "tiny_stories")) {
        std.debug.print("⬇️  Downloading TinyStories (~19MB validation sample)...\n", .{});
        const url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt";
        try downloadFile(&client, io, allocator, url, "data/tinystories_valid.txt", false);
        std.debug.print("✨ TinyStories dataset is ready in data/tinystories_valid.txt!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "alpaca") or std.mem.eql(u8, dataset_name, "alpaca_data") or std.mem.eql(u8, dataset_name, "alpaca_cleaned")) {
        std.debug.print("⬇️  Downloading Stanford Alpaca SFT dataset (~22MB JSON)...\n", .{});
        const url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json";
        try downloadFile(&client, io, allocator, url, "data/alpaca_data.json", false);
        std.debug.print("✨ Stanford Alpaca dataset is ready in data/alpaca_data.json!\n", .{});

    } else if (std.mem.eql(u8, dataset_name, "all_llm") or std.mem.eql(u8, dataset_name, "llm")) {
        std.debug.print("⬇️  Downloading all 4 LLM datasets (tinyshakespeare, wikitext2, tinystories, alpaca)...\n", .{});
        try downloadFile(&client, io, allocator, "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "data/tinyshakespeare.txt", false);

        cwd.createDir(io, "data/wikitext-2", .default_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        try downloadFile(&client, io, allocator, "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt", "data/wikitext-2/train.txt", false);
        try downloadFile(&client, io, allocator, "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt", "data/wikitext-2/valid.txt", false);
        try downloadFile(&client, io, allocator, "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt", "data/wikitext-2/test.txt", false);

        try downloadFile(&client, io, allocator, "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt", "data/tinystories_valid.txt", false);
        try downloadFile(&client, io, allocator, "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json", "data/alpaca_data.json", false);
        std.debug.print("✨ All LLM datasets downloaded and ready in data/!\n", .{});

    } else {
        std.debug.print("❌ Unknown dataset: '{s}'\n", .{dataset_name});
        std.debug.print("Supported datasets:\n", .{});
        std.debug.print("  - Vision: fashion_mnist, mnist\n", .{});
        std.debug.print("  - LLM:    tinyshakespeare, wikitext2, tinystories, alpaca, all_llm\n", .{});
        return error.InvalidDatasetName;
    }
}
