const std = @import("std");
const dset = @import("../dataset.zig");
const ImageDataset = dset.ImageDataset;
const LabelDataset = dset.LabelDataset;
const loadImages = dset.loadImages;
const loadLabels = dset.loadLabels;
const Dataset = dset.Dataset;
const loadDataset = dset.loadDataset;
const loadTinyImageNetBin = dset.loadTinyImageNetBin;
const createSyntheticTinyImageNet = dset.createSyntheticTinyImageNet;
const DataLoader = dset.DataLoader;
const DataLoaderOptions = dset.DataLoaderOptions;
const TokenId = dset.TokenId;
const MergePair = dset.MergePair;
const MergeRule = dset.MergeRule;
const BPETokenizer = dset.BPETokenizer;
const BinaryMmapDataset = dset.BinaryMmapDataset;
const MultimodalPair = dset.MultimodalPair;
const MultimodalDataset = dset.MultimodalDataset;
const MultimodalDataLoaderOptions = dset.MultimodalDataLoaderOptions;
const MultimodalDataLoader = dset.MultimodalDataLoader;
const createSyntheticMultimodalDataset = dset.createSyntheticMultimodalDataset;

test "DataLoader basic functionality" {
    const allocator = std.testing.allocator;

    // Create a mock dataset
    var images_data = try allocator.alloc(f32, 12 * 2);
    defer allocator.free(images_data);
    for (0..24) |i| {
        images_data[i] = @as(f32, @floatFromInt(i));
    }

    var labels_data = try allocator.alloc(u8, 12);
    defer allocator.free(labels_data);
    for (0..12) |i| {
        labels_data[i] = @as(u8, @intCast(i));
    }

    const ds = Dataset{
        .images = .{
            .num_images = 12,
            .rows = 2,
            .cols = 1,
            .data = images_data,
        },
        .labels = .{
            .num_items = 12,
            .data = labels_data,
        },
    };

    var loader = try DataLoader.init(allocator, ds, 5, .{ .shuffle = false, .drop_last = false });
    defer loader.deinit(allocator);

    const x_buf = try allocator.alloc(f32, 5 * 2);
    defer allocator.free(x_buf);
    const y_buf = try allocator.alloc(u8, 5);
    defer allocator.free(y_buf);

    // First batch: 5 items
    const b1 = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, 5), b1);
    try std.testing.expectEqual(@as(f32, 0.0), x_buf[0]);
    try std.testing.expectEqual(@as(f32, 1.0), x_buf[1]);
    try std.testing.expectEqual(@as(u8, 0), y_buf[0]);
    try std.testing.expectEqual(@as(u8, 4), y_buf[4]);

    // Second batch: 5 items
    const b2 = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, 5), b2);
    try std.testing.expectEqual(@as(f32, 10.0), x_buf[0]);
    try std.testing.expectEqual(@as(u8, 5), y_buf[0]);

    // Third batch: 2 items (since drop_last = false)
    const b3 = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, 2), b3);
    try std.testing.expectEqual(@as(f32, 20.0), x_buf[0]);
    try std.testing.expectEqual(@as(u8, 10), y_buf[0]);

    // Fourth batch: null
    const b4 = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, null), b4);

    // Reset and try drop_last = true
    loader.drop_last = true;
    loader.reset();

    // First batch: 5 items
    const b1_dl = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, 5), b1_dl);

    // Second batch: 5 items
    const b2_dl = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, 5), b2_dl);

    // Third batch: null (remaining is 2, less than batch_size of 5, drop_last is true)
    const b3_dl = loader.nextInto(x_buf, y_buf);
    try std.testing.expectEqual(@as(?usize, null), b3_dl);
}

// ============================================================================
// 2. 字节级字节对编码分词器与内存映射数据集 (Byte-level Byte Pair Encoding, BPE & Memory-Mapped, MMap Dataset)
// ============================================================================



test "BPETokenizer encode, decode and merge" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // Add merge: "h" + "e" -> "he" (rank 0), "l" + "l" -> "ll" (rank 1), "he" + "ll" -> "hell" (rank 2), "hell" + "o" -> "hello" (rank 3)
    _ = try tokenizer.addMerge("h", "e", 0);
    _ = try tokenizer.addMerge("l", "l", 1);
    _ = try tokenizer.addMerge("he", "ll", 2);
    _ = try tokenizer.addMerge("hell", "o", 3);

    const encoded = try tokenizer.encode(allocator, "hello world hello");
    defer allocator.free(encoded);

    // "hello" should be compressed to a single token ID >= 256
    const hello_id = tokenizer.vocab.get("hello").?;
    try std.testing.expectEqual(hello_id, encoded[0]);

    const decoded = try tokenizer.decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("hello world hello", decoded);
}

test "BinaryMmapDataset batch slicing" {
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    const dataset = BinaryMmapDataset.fromSlice(&tokens, 3);

    const batch = dataset.getBatch(0, 2); // batch_size=2, seq_len=3 -> total 6 tokens
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 30, 40, 50, 60 }, batch.x);
    try std.testing.expectEqualSlices(u32, &.{ 20, 30, 40, 50, 60, 70 }, batch.y);
}

test "BPETokenizer unsupervised training from corpus" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    const corpus = "abababababab abababab";
    try tokenizer.train(allocator, corpus, 258); // 256 bytes + 2 merges ("ab", "abab")

    try std.testing.expect(tokenizer.vocab.contains("ab"));
    const encoded = try tokenizer.encode(allocator, corpus);
    defer allocator.free(encoded);

    const decoded = try tokenizer.decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(corpus, decoded);
}

test "BPETokenizer training breaks frequency ties by lexicographic pair order" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // 25 个相邻 Pair 都恰好出现 2 次：每一轮都是多方平局，胜者必须是 (a, b) 字典序最小者，
    // 依次为 "as" (97, 115)、"bn" (98, 110)、"cv" (99, 118)、"df" (100, 102)
    const keys = "qwertyuiopasdfghjklzxcvbnm";
    const corpus = keys ++ " " ++ keys;
    try tokenizer.train(allocator, corpus, 256 + 4);

    const expected = [_][2][]const u8{ .{ "a", "s" }, .{ "b", "n" }, .{ "c", "v" }, .{ "d", "f" } };
    try std.testing.expectEqual(expected.len, tokenizer.merges.count());
    for (expected, 0..) |pair, rank| {
        const key = MergePair{ .a = tokenizer.vocab.get(pair[0]).?, .b = tokenizer.vocab.get(pair[1]).? };
        const rule = tokenizer.merges.get(key) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(u32, @intCast(rank)), rule.rank);
        try std.testing.expectEqual(@as(TokenId, @intCast(256 + rank)), rule.id);
    }

    // 再训练一次得到完全相同的词表
    var again = try BPETokenizer.init(allocator);
    defer again.deinit();
    try again.train(allocator, corpus, 256 + 4);
    for (tokenizer.inv_vocab.items, again.inv_vocab.items) |a, b| {
        try std.testing.expectEqualStrings(a, b);
    }
}

test "BPETokenizer edge cases empty and multibyte UTF8" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // 1. Empty string encode & decode
    const empty_enc = try tokenizer.encode(allocator, "");
    defer allocator.free(empty_enc);
    try std.testing.expectEqual(@as(usize, 0), empty_enc.len);

    const empty_dec = try tokenizer.decode(allocator, empty_enc);
    defer allocator.free(empty_dec);
    try std.testing.expectEqualStrings("", empty_dec);

    // 2. Single ASCII char
    const single_enc = try tokenizer.encode(allocator, "Z");
    defer allocator.free(single_enc);
    try std.testing.expectEqual(@as(usize, 1), single_enc.len);
    try std.testing.expectEqual(@as(u32, 'Z'), single_enc[0]);

    const single_dec = try tokenizer.decode(allocator, single_enc);
    defer allocator.free(single_dec);
    try std.testing.expectEqualStrings("Z", single_dec);

    // 3. Multibyte UTF-8 characters without merges (fallback to byte tokens)
    const utf8_str = "深度学习🤖";
    const utf8_enc = try tokenizer.encode(allocator, utf8_str);
    defer allocator.free(utf8_enc);
    try std.testing.expectEqual(utf8_str.len, utf8_enc.len); // each byte is a byte token

    const utf8_dec = try tokenizer.decode(allocator, utf8_enc);
    defer allocator.free(utf8_dec);
    try std.testing.expectEqualStrings(utf8_str, utf8_dec);
}

test "BPETokenizer multi-byte UTF-8 chunk boundary handling" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // 构造一个长度远超 256 字节且无空格的连续多字节中文串，强制在 256 字节处分割
    const chinese_segment = "人工智能与神经网络系统深度学习框架矩阵计算"; // 21 个汉字 * 3 = 63 字节
    const long_chinese_arr = comptime blk: {
        var buf: [chinese_segment.len * 6]u8 = undefined;
        for (0..6) |i| @memcpy(buf[i * chinese_segment.len .. (i + 1) * chinese_segment.len], chinese_segment);
        break :blk buf;
    };
    const long_chinese = &long_chinese_arr;

    const encoded = try tokenizer.encode(allocator, long_chinese);
    defer allocator.free(encoded);

    const decoded = try tokenizer.decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(long_chinese, decoded);

    // 构造带有自然单词与空格的长文本，测试空白字符后向探测与无损还原
    const mixed_str = "ZNN framework tokenization across multi-byte UTF-8 boundaries. ";
    const mixed_corpus_arr = comptime blk: {
        var buf: [mixed_str.len * 8]u8 = undefined;
        for (0..8) |i| @memcpy(buf[i * mixed_str.len .. (i + 1) * mixed_str.len], mixed_str);
        break :blk buf;
    };
    const mixed_corpus = &mixed_corpus_arr;
    const mixed_encoded = try tokenizer.encode(allocator, mixed_corpus);
    defer allocator.free(mixed_encoded);

    const mixed_decoded = try tokenizer.decode(allocator, mixed_encoded);
    defer allocator.free(mixed_decoded);

    try std.testing.expectEqualStrings(mixed_corpus, mixed_decoded);
}

test "DataLoader drop_last, shuffle, and edge batch slicing" {
    const allocator = std.testing.allocator;

    var img_data = [_]f32{
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
    };
    var lbl_data = [_]u8{ 1, 2, 3, 4, 5 };
    const mock_ds = Dataset{
        .images = .{ .num_images = 5, .rows = 1, .cols = 2, .data = &img_data },
        .labels = .{ .num_items = 5, .data = &lbl_data },
    };

    // 1. drop_last = true with 5 samples and batch_size = 2 -> 2 full batches (4 items), 1 dropped
    var loader_drop = try DataLoader.init(allocator, mock_ds, 2, .{ .shuffle = false, .drop_last = true });
    defer loader_drop.deinit(allocator);

    var x_buf: [4]f32 = undefined;
    var y_buf: [2]u8 = undefined;

    try std.testing.expectEqual(@as(usize, 2), loader_drop.peekNextBatchSize());
    _ = loader_drop.nextInto(&x_buf, &y_buf);
    try std.testing.expectEqual(@as(usize, 2), loader_drop.peekNextBatchSize());
    _ = loader_drop.nextInto(&x_buf, &y_buf);
    // 5th element remaining, but drop_last is true, so peekNextBatchSize returns 0
    try std.testing.expectEqual(@as(usize, 0), loader_drop.peekNextBatchSize());

    // 2. drop_last = false -> 5th element should be yielded as a batch of size 1
    var loader_keep = try DataLoader.init(allocator, mock_ds, 2, .{ .shuffle = false, .drop_last = false });
    defer loader_keep.deinit(allocator);

    _ = loader_keep.nextInto(&x_buf, &y_buf);
    _ = loader_keep.nextInto(&x_buf, &y_buf);
    try std.testing.expectEqual(@as(usize, 1), loader_keep.peekNextBatchSize());
    const last_n = loader_keep.nextInto(&x_buf, &y_buf);
    try std.testing.expectEqual(@as(usize, 1), last_n.?);

    // 3. BinaryMmapDataset numBatches edge case
    const tokens = [_]u32{ 1, 2 };
    const small_ds = BinaryMmapDataset.fromSlice(&tokens, 5); // seq_len=5, tokens.len=2 < batch*seq_len
    try std.testing.expectEqual(@as(usize, 0), small_ds.numBatches(1));
}

test "loadImages" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const cwd = std.Io.Dir.cwd();
    const io = testing.io;

    const valid_file = "test_dataset_images_valid.bin";
    const invalid_file = "test_dataset_images_invalid.bin";

    // 1. Test Valid File
    {
        var file = try cwd.createFile(io, valid_file, .{});
        defer cwd.deleteFile(io, valid_file) catch {};

        var buf: [1024]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        // Write magic number: 0x00000803
        var temp: [4]u8 = undefined;
        std.mem.writeInt(u32, &temp, 0x00000803, .big);
        try writer.writeAll(&temp);

        // Write num_images: 2
        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        // Write rows: 2
        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        // Write cols: 2
        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        // Write 8 pixels (2 images * 2 rows * 2 cols)
        const pixels = [_]u8{ 0, 51, 102, 153, 204, 255, 127, 64 };
        try writer.writeAll(&pixels);

        try writer.flush();
        file.close(io);

        var ds = try loadImages(allocator, io, valid_file);
        defer ds.deinit(allocator);

        try testing.expectEqual(@as(u32, 2), ds.num_images);
        try testing.expectEqual(@as(u32, 2), ds.rows);
        try testing.expectEqual(@as(u32, 2), ds.cols);

        // Check normalized values
        try testing.expectApproxEqRel(@as(f32, 0.0), ds.data[0], 1e-4);
        try testing.expectApproxEqRel(@as(f32, 51.0 / 255.0), ds.data[1], 1e-4);
        try testing.expectApproxEqRel(@as(f32, 1.0), ds.data[5], 1e-4);
    }

    // 2. Test Invalid Magic Number
    {
        var file = try cwd.createFile(io, invalid_file, .{});
        defer cwd.deleteFile(io, invalid_file) catch {};

        var buf: [128]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        // Write wrong magic number: 0x00000801 (this is for labels, not images)
        var temp: [4]u8 = undefined;
        std.mem.writeInt(u32, &temp, 0x00000801, .big);
        try writer.writeAll(&temp);

        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        std.mem.writeInt(u32, &temp, 2, .big);
        try writer.writeAll(&temp);

        const pixels: [8]u8 = @splat(0);
        try writer.writeAll(&pixels);

        try writer.flush();
        file.close(io);

        try testing.expectError(error.InvalidMagicNumber, loadImages(allocator, io, invalid_file));
    }
}

test "DataLoader rejects mismatched datasets and shuffles empty datasets safely" {
    const allocator = std.testing.allocator;

    var img_data = [_]f32{ 1, 2, 3, 4 };
    var lbl_data = [_]u8{ 0, 1 };
    const mismatched = Dataset{
        .images = .{ .num_images = 2, .rows = 1, .cols = 2, .data = &img_data },
        .labels = .{ .num_items = 1, .data = lbl_data[0..1] },
    };
    try std.testing.expectError(error.DatasetSizeMismatch, DataLoader.init(allocator, mismatched, 1, .{}));

    const ok = Dataset{
        .images = .{ .num_images = 2, .rows = 1, .cols = 2, .data = &img_data },
        .labels = .{ .num_items = 2, .data = &lbl_data },
    };
    try std.testing.expectError(error.InvalidBatchSize, DataLoader.init(allocator, ok, 0, .{}));

    var empty_img = [_]f32{};
    var empty_lbl = [_]u8{};
    const empty = Dataset{
        .images = .{ .num_images = 0, .rows = 1, .cols = 2, .data = &empty_img },
        .labels = .{ .num_items = 0, .data = &empty_lbl },
    };
    var loader = try DataLoader.init(allocator, empty, 4, .{ .shuffle = true });
    defer loader.deinit(allocator);
    loader.reset();
    var x: [8]f32 = undefined;
    var y: [4]u8 = undefined;
    try std.testing.expectEqual(@as(?usize, null), loader.nextInto(&x, &y));
}

test "BPETokenizer train uses the merged id when the merged string already exists" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // "ab" 预先作为特殊 Token 注册，之后再注册一个 ID 更大的 Token；
    // 训练合并 ("a","b") 时必须沿用已有的 "ab" ID，而不是词表中最后一个 Token
    const ab_id = try tokenizer.addToken("ab");
    _ = try tokenizer.addToken("<|eos|>");
    try tokenizer.train(allocator, "abababababab abababab", 262);

    try std.testing.expectEqual(ab_id, tokenizer.merges.get(.{ .a = 'a', .b = 'b' }).?.id);
    try std.testing.expect(tokenizer.vocab.contains("abab"));
    try std.testing.expect(!tokenizer.vocab.contains("<|eos|><|eos|>"));

    const encoded = try tokenizer.encode(allocator, "abababab");
    defer allocator.free(encoded);
    try std.testing.expectEqual(@as(usize, 1), encoded.len);
}

fn initTokenizerWithMerges(allocator: std.mem.Allocator) !void {
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();
    _ = try tokenizer.addMerge("a", "b", 0);
    _ = try tokenizer.addToken("<|eos|>");
}

test "BPETokenizer init, addToken and addMerge do not leak on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, initTokenizerWithMerges, .{});
}

test "BinaryMmapDataset loadFromBytes reinterprets aligned bytes" {
    const tokens = [_]u32{ 7, 8, 9, 10 };
    const bytes: []align(@alignOf(u32)) const u8 = std.mem.sliceAsBytes(&tokens);
    const ds = BinaryMmapDataset.loadFromBytes(bytes, 2);
    try std.testing.expectEqualSlices(u32, &tokens, ds.tokens);
    try std.testing.expectEqual(@as(usize, 1), ds.numBatches(1));
    const batch = ds.getBatch(1, 1);
    try std.testing.expectEqualSlices(u32, &.{ 8, 9 }, batch.x);
    try std.testing.expectEqualSlices(u32, &.{ 9, 10 }, batch.y);
}

test "loadImages 4D multi-channel RGB format" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const cwd = std.Io.Dir.cwd();
    const io = testing.io;

    const test_file = "test_4d_images.bin";
    var file = try cwd.createFile(io, test_file, .{});
    defer cwd.deleteFile(io, test_file) catch {};

    var buf: [1024]u8 = undefined;
    var file_writer = file.writer(io, &buf);
    const writer = &file_writer.interface;

    // Magic: 0x00000804 (4D unsigned byte tensor: N, C, H, W)
    var temp: [4]u8 = undefined;
    std.mem.writeInt(u32, &temp, 0x00000804, .big);
    try writer.writeAll(&temp);

    // N = 2
    std.mem.writeInt(u32, &temp, 2, .big);
    try writer.writeAll(&temp);

    // C = 3 (RGB)
    std.mem.writeInt(u32, &temp, 3, .big);
    try writer.writeAll(&temp);

    // H = 2
    std.mem.writeInt(u32, &temp, 2, .big);
    try writer.writeAll(&temp);

    // W = 2
    std.mem.writeInt(u32, &temp, 2, .big);
    try writer.writeAll(&temp);

    // 2 * 3 * 2 * 2 = 24 pixels
    var pixels: [24]u8 = undefined;
    for (&pixels, 0..) |*p, i| {
        p.* = @as(u8, @intCast(i * 10));
    }
    try writer.writeAll(&pixels);
    try writer.flush();
    file.close(io);

    var ds = try loadImages(allocator, io, test_file);
    defer ds.deinit(allocator);

    try testing.expectEqual(@as(u32, 2), ds.num_images);
    try testing.expectEqual(@as(u32, 3), ds.channels);
    try testing.expectEqual(@as(u32, 2), ds.rows);
    try testing.expectEqual(@as(u32, 2), ds.cols);
    try testing.expectEqual(@as(usize, 12), ds.sampleSize());
    try testing.expectApproxEqRel(@as(f32, 0.0), ds.data[0], 1e-4);
    try testing.expectApproxEqRel(@as(f32, 10.0 / 255.0), ds.data[1], 1e-4);
}

test "createSyntheticTinyImageNet and loadTinyImageNetBin roundtrip" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const cwd = std.Io.Dir.cwd();
    const io = testing.io;

    var synth = try createSyntheticTinyImageNet(allocator, 4);
    defer synth.deinit(allocator);

    try testing.expectEqual(@as(u32, 4), synth.images.num_images);
    try testing.expectEqual(@as(u32, 3), synth.images.channels);
    try testing.expectEqual(@as(u32, 64), synth.images.rows);
    try testing.expectEqual(@as(u32, 64), synth.images.cols);
    try testing.expectEqual(@as(u32, 4), synth.labels.num_items);

    // Write binary Tiny ImageNet format file
    const test_bin_file = "test_tiny_imagenet_roundtrip.bin";
    var file = try cwd.createFile(io, test_bin_file, .{});
    defer cwd.deleteFile(io, test_bin_file) catch {};

    var buf: [65536]u8 = undefined;
    var file_writer = file.writer(io, &buf);
    const writer = &file_writer.interface;

    const sample_pixels = 3 * 64 * 64;
    var record_buf: [1 + sample_pixels]u8 = undefined;
    for (0..4) |i| {
        record_buf[0] = synth.labels.data[i];
        for (0..sample_pixels) |p_idx| {
            const float_val = synth.images.data[i * sample_pixels + p_idx];
            record_buf[1 + p_idx] = @as(u8, @intFromFloat(std.math.clamp(float_val * 255.0, 0.0, 255.0)));
        }
        try writer.writeAll(&record_buf);
    }
    try writer.flush();
    file.close(io);

    // Read back with loadTinyImageNetBin
    var loaded = try loadTinyImageNetBin(allocator, io, test_bin_file, null);
    defer loaded.deinit(allocator);

    try testing.expectEqual(@as(u32, 4), loaded.images.num_images);
    try testing.expectEqual(@as(u32, 3), loaded.images.channels);
    try testing.expectEqual(@as(u32, 64), loaded.images.rows);
    try testing.expectEqual(@as(u32, 64), loaded.images.cols);
    try testing.expectEqual(synth.labels.data[0], loaded.labels.data[0]);
    try testing.expectEqual(synth.labels.data[3], loaded.labels.data[3]);
}

test "MultimodalDataset and MultimodalDataLoader" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    var mm_ds = try createSyntheticMultimodalDataset(allocator, &tokenizer, 8);
    defer mm_ds.deinit();

    try testing.expectEqual(@as(usize, 8), mm_ds.count());
    const pair0 = mm_ds.get(0);
    try testing.expect(pair0.caption.len > 0);
    try testing.expect(pair0.tokens.len > 0);
    try testing.expectEqual(@as(usize, 3 * 64 * 64), pair0.image.len);

    // Multimodal DataLoader
    const batch_sz = 3;
    const max_seq = 32;
    var mm_loader = try MultimodalDataLoader.init(
        allocator,
        mm_ds,
        batch_sz,
        max_seq,
        .{ .shuffle = true, .drop_last = false, .pad_token_id = 0 },
    );
    defer mm_loader.deinit(allocator);

    const img_buf = try allocator.alloc(f32, batch_sz * 3 * 64 * 64);
    defer allocator.free(img_buf);
    const tokens_buf = try allocator.alloc(u32, batch_sz * max_seq);
    defer allocator.free(tokens_buf);
    const lengths_buf = try allocator.alloc(usize, batch_sz);
    defer allocator.free(lengths_buf);

    // Batch 1: 3 items
    const b1 = mm_loader.nextInto(img_buf, tokens_buf, lengths_buf);
    try testing.expectEqual(@as(?usize, 3), b1);
    try testing.expect(lengths_buf[0] > 0);

    // Batch 2: 3 items
    const b2 = mm_loader.nextInto(img_buf, tokens_buf, lengths_buf);
    try testing.expectEqual(@as(?usize, 3), b2);

    // Batch 3: 2 items (drop_last is false)
    const b3 = mm_loader.nextInto(img_buf, tokens_buf, lengths_buf);
    try testing.expectEqual(@as(?usize, 2), b3);

    // Batch 4: null
    const b4 = mm_loader.nextInto(img_buf, tokens_buf, lengths_buf);
    try testing.expectEqual(@as(?usize, null), b4);
}
