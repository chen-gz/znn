const std = @import("std");

pub const ImageDataset = struct {
    num_images: u32,
    rows: u32,
    cols: u32,
    data: []f32, // normalized between 0.0 and 1.0 (shape: num_images * rows * cols)

    pub fn deinit(self: *ImageDataset, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub const LabelDataset = struct {
    num_items: u32,
    data: []u8, // raw class labels 0-9 (shape: num_items)

    pub fn deinit(self: *LabelDataset, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub fn loadImages(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !ImageDataset {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    var file_buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &file_buf);
    const reader = &file_reader.interface;

    var temp_4: [4]u8 = undefined;

    try reader.readSliceAll(&temp_4);
    const magic = std.mem.readInt(u32, &temp_4, .big);
    if (magic != 0x00000803) {
        return error.InvalidMagicNumber;
    }

    try reader.readSliceAll(&temp_4);
    const num_images = std.mem.readInt(u32, &temp_4, .big);

    try reader.readSliceAll(&temp_4);
    const rows = std.mem.readInt(u32, &temp_4, .big);

    try reader.readSliceAll(&temp_4);
    const cols = std.mem.readInt(u32, &temp_4, .big);

    const num_pixels = @as(usize, num_images) * rows * cols;
    const data = try allocator.alloc(f32, num_pixels);
    errdefer allocator.free(data);

    // Read pixels in chunks
    var temp_chunk: [4096]u8 = undefined;
    var idx: usize = 0;
    while (idx < num_pixels) {
        const to_read = @min(temp_chunk.len, num_pixels - idx);
        try reader.readSliceAll(temp_chunk[0..to_read]);
        for (temp_chunk[0..to_read]) |pixel| {
            data[idx] = @as(f32, @floatFromInt(pixel)) / 255.0;
            idx += 1;
        }
    }

    return ImageDataset{
        .num_images = num_images,
        .rows = rows,
        .cols = cols,
        .data = data,
    };
}

pub fn loadLabels(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !LabelDataset {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    var file_buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &file_buf);
    const reader = &file_reader.interface;

    var temp_4: [4]u8 = undefined;

    try reader.readSliceAll(&temp_4);
    const magic = std.mem.readInt(u32, &temp_4, .big);
    if (magic != 0x00000801) {
        return error.InvalidMagicNumber;
    }

    try reader.readSliceAll(&temp_4);
    const num_items = std.mem.readInt(u32, &temp_4, .big);

    const data = try allocator.alloc(u8, num_items);
    errdefer allocator.free(data);

    try reader.readSliceAll(data);

    return LabelDataset{
        .num_items = num_items,
        .data = data,
    };
}

pub const Dataset = struct {
    images: ImageDataset,
    labels: LabelDataset,

    pub fn deinit(self: *Dataset, allocator: std.mem.Allocator) void {
        self.images.deinit(allocator);
        self.labels.deinit(allocator);
    }
};

pub fn loadDataset(allocator: std.mem.Allocator, io: std.Io, images_path: []const u8, labels_path: []const u8) !Dataset {
    var images = try loadImages(allocator, io, images_path);
    errdefer images.deinit(allocator);

    var labels = try loadLabels(allocator, io, labels_path);
    errdefer labels.deinit(allocator);

    return Dataset{
        .images = images,
        .labels = labels,
    };
}

pub const DataLoaderOptions = struct {
    shuffle: bool = false,
    seed: ?u64 = null,
    drop_last: bool = false,
};

pub const DataLoader = struct {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: []usize,
    current_index: usize,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, ds: Dataset, batch_size: usize, options: DataLoaderOptions) !DataLoader {
        const num_samples = ds.images.num_images;
        const indices = try allocator.alloc(usize, num_samples);
        for (0..num_samples) |i| {
            indices[i] = i;
        }
        var self = DataLoader{
            .dataset = ds,
            .batch_size = batch_size,
            .shuffle = options.shuffle,
            .drop_last = options.drop_last,
            .indices = indices,
            .current_index = 0,
            .prng = std.Random.DefaultPrng.init(options.seed orelse 1337),
        };
        if (self.shuffle) {
            self.shuffleIndices();
        }
        return self;
    }

    pub fn deinit(self: *DataLoader, allocator: std.mem.Allocator) void {
        allocator.free(self.indices);
    }

    pub fn shuffleIndices(self: *DataLoader) void {
        const random = self.prng.random();
        var i: usize = self.indices.len - 1;
        while (i > 0) : (i -= 1) {
            const j = random.intRangeLessThan(usize, 0, i + 1);
            const temp = self.indices[i];
            self.indices[i] = self.indices[j];
            self.indices[j] = temp;
        }
    }

    pub fn reset(self: *DataLoader) void {
        self.current_index = 0;
        if (self.shuffle) {
            self.shuffleIndices();
        }
    }

    pub fn peekNextBatchSize(self: DataLoader) usize {
        const num_samples = self.indices.len;
        if (self.current_index >= num_samples) {
            return 0;
        }
        const remaining = num_samples - self.current_index;
        if (self.drop_last and remaining < self.batch_size) {
            return 0;
        }
        return @min(self.batch_size, remaining);
    }

    pub fn nextInto(self: *DataLoader, x_dest: []f32, y_dest: []u8) ?usize {
        const num_samples = self.indices.len;
        if (self.current_index >= num_samples) {
            return null;
        }
        const remaining = num_samples - self.current_index;
        if (self.drop_last and remaining < self.batch_size) {
            return null;
        }
        const actual_batch_size = @min(self.batch_size, remaining);
        if (actual_batch_size == 0) {
            return null;
        }

        const input_dim = self.dataset.images.rows * self.dataset.images.cols;
        for (0..actual_batch_size) |j| {
            const idx = self.indices[self.current_index + j];
            @memcpy(
                x_dest[j * input_dim .. (j + 1) * input_dim],
                self.dataset.images.data[idx * input_dim .. (idx + 1) * input_dim]
            );
            y_dest[j] = self.dataset.labels.data[idx];
        }
        self.current_index += actual_batch_size;
        return actual_batch_size;
    }
};

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
// 2. Byte-level BPE 分词器与内存映射数据集 (Byte-level BPE & Mmap Dataset)
// ============================================================================

pub const TokenId = u32;
pub const MergePair = struct { a: TokenId, b: TokenId };

pub const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(TokenId),
    inv_vocab: std.ArrayList([]const u8),
    merges: std.AutoHashMap(MergePair, u32),

    pub fn init(allocator: std.mem.Allocator) !BPETokenizer {
        var vocab = std.StringHashMap(TokenId).init(allocator);
        errdefer vocab.deinit();
        var inv_vocab: std.ArrayList([]const u8) = .empty;
        errdefer inv_vocab.deinit(allocator);

        // 初始化 256 个基本单字节 Token (0x00 ~ 0xFF)
        for (0..256) |b| {
            const slice = try allocator.alloc(u8, 1);
            slice[0] = @as(u8, @intCast(b));
            try inv_vocab.append(allocator, slice);
            try vocab.put(slice, @as(TokenId, @intCast(b)));
        }

        return BPETokenizer{
            .allocator = allocator,
            .vocab = vocab,
            .inv_vocab = inv_vocab,
            .merges = std.AutoHashMap(MergePair, u32).init(allocator),
        };
    }

    pub fn deinit(self: *BPETokenizer) void {
        self.vocab.deinit();
        for (self.inv_vocab.items) |slice| {
            self.allocator.free(slice);
        }
        self.inv_vocab.deinit(self.allocator);
        self.merges.deinit();
    }

    /// 注册一个自定义/特殊 Token (如 <|bos|>, <|eos|>)
    pub fn addToken(self: *BPETokenizer, token_str: []const u8) !TokenId {
        if (self.vocab.get(token_str)) |id| return id;

        const owned_str = try self.allocator.dupe(u8, token_str);
        const new_id = @as(TokenId, @intCast(self.inv_vocab.items.len));
        try self.inv_vocab.append(self.allocator, owned_str);
        try self.vocab.put(owned_str, new_id);
        return new_id;
    }

    /// 添加一条合并规则: (str_a, str_b) -> 优先级 rank
    pub fn addMerge(self: *BPETokenizer, str_a: []const u8, str_b: []const u8, rank: u32) !void {
        const id_a = self.vocab.get(str_a) orelse try self.addToken(str_a);
        const id_b = self.vocab.get(str_b) orelse try self.addToken(str_b);

        var merged_str = try self.allocator.alloc(u8, str_a.len + str_b.len);
        defer self.allocator.free(merged_str);
        @memcpy(merged_str[0..str_a.len], str_a);
        @memcpy(merged_str[str_a.len..], str_b);

        _ = try self.addToken(merged_str);
        try self.merges.put(.{ .a = id_a, .b = id_b }, rank);
    }

    /// 将 UTF-8 文本编码为 Token ID 序列
    pub fn encode(self: *const BPETokenizer, allocator: std.mem.Allocator, text: []const u8) ![]TokenId {
        var tokens: std.ArrayList(TokenId) = .empty;
        defer tokens.deinit(allocator);

        // 1. 初始化为单字节 Token ID (0 ~ 255)
        for (text) |byte| {
            try tokens.append(allocator, @as(TokenId, byte));
        }

        // 2. 迭代根据 merges 规则进行连续最高优先级合并
        while (tokens.items.len >= 2) {
            var min_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;
            var best_pair: ?MergePair = null;

            for (0..tokens.items.len - 1) |i| {
                const pair = MergePair{ .a = tokens.items[i], .b = tokens.items[i + 1] };
                if (self.merges.get(pair)) |rank| {
                    if (rank < min_rank) {
                        min_rank = rank;
                        best_idx = i;
                        best_pair = pair;
                    }
                }
            }

            if (best_idx) |idx| {
                const pair = best_pair.?;
                const str_a = self.inv_vocab.items[pair.a];
                const str_b = self.inv_vocab.items[pair.b];
                var merged_str = try allocator.alloc(u8, str_a.len + str_b.len);
                defer allocator.free(merged_str);
                @memcpy(merged_str[0..str_a.len], str_a);
                @memcpy(merged_str[str_a.len..], str_b);

                if (self.vocab.get(merged_str)) |new_id| {
                    tokens.items[idx] = new_id;
                    _ = tokens.orderedRemove(idx + 1);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// 将 Token ID 序列还原为 UTF-8 字符串
    pub fn decode(self: *const BPETokenizer, allocator: std.mem.Allocator, tokens: []const TokenId) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);

        for (tokens) |tok| {
            if (tok < self.inv_vocab.items.len) {
                try buf.appendSlice(allocator, self.inv_vocab.items[tok]);
            }
        }

        return buf.toOwnedSlice(allocator);
    }
};

/// 零拷贝内存映射二进制数据集 (MMap Binary DataLoader)
pub const BinaryMmapDataset = struct {
    tokens: []const u32,
    seq_len: usize,

    pub fn fromSlice(tokens: []const u32, seq_len: usize) BinaryMmapDataset {
        return .{
            .tokens = tokens,
            .seq_len = seq_len,
        };
    }

    pub fn loadFromBytes(bytes: []align(std.mem.page_size) const u8, seq_len: usize) BinaryMmapDataset {
        const token_count = bytes.len / @sizeOf(u32);
        const tokens: [*]const u32 = @ptrCast(@alignCast(bytes.ptr));
        return .{
            .tokens = tokens[0..token_count],
            .seq_len = seq_len,
        };
    }

    pub fn close(self: *BinaryMmapDataset) void {
        _ = self;
    }

    /// 获取训练批次 (输入 X 与移位预测标签 Y)
    pub fn getBatch(self: *const BinaryMmapDataset, offset: usize, batch_size: usize) struct { x: []const u32, y: []const u32 } {
        const total_tokens = batch_size * self.seq_len;
        std.debug.assert(offset + total_tokens + 1 <= self.tokens.len);
        const x_slice = self.tokens[offset .. offset + total_tokens];
        const y_slice = self.tokens[offset + 1 .. offset + total_tokens + 1];
        return .{ .x = x_slice, .y = y_slice };
    }

    pub fn numBatches(self: *const BinaryMmapDataset, batch_size: usize) usize {
        const total_tokens = batch_size * self.seq_len;
        if (self.tokens.len <= total_tokens) return 0;
        return (self.tokens.len - 1) / total_tokens;
    }
};

test "BPETokenizer encode, decode and merge" {
    const allocator = std.testing.allocator;
    var tokenizer = try BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // Add merge: "h" + "e" -> "he" (rank 0), "l" + "l" -> "ll" (rank 1), "he" + "ll" -> "hell" (rank 2), "hell" + "o" -> "hello" (rank 3)
    try tokenizer.addMerge("h", "e", 0);
    try tokenizer.addMerge("l", "l", 1);
    try tokenizer.addMerge("he", "ll", 2);
    try tokenizer.addMerge("hell", "o", 3);

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
    var dataset = BinaryMmapDataset.fromSlice(&tokens, 3);
    defer dataset.close();

    const batch = dataset.getBatch(0, 2); // batch_size=2, seq_len=3 -> total 6 tokens
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 30, 40, 50, 60 }, batch.x);
    try std.testing.expectEqualSlices(u32, &.{ 20, 30, 40, 50, 60, 70 }, batch.y);
}


