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

/// 读取 IDX3 格式 (MNIST / Fashion-MNIST) 图像文件 (大端序头部：magic 0x00000803、图像数、行数、列数)。
/// 每个像素字节按固定常数缩放到 [0, 1]：`x = pixel / 255`。这是与数据无关的全局线性缩放，
/// 所有像素、所有样本以及训练集与测试集共用同一映射；不统计任何逐像素或全局的均值与方差，
/// 因此训练集与测试集之间不存在统计量泄漏。需要零均值、单位方差标准化的调用方应只在训练集上
/// 计算统计量，并把同一组统计量同时作用于训练集与测试集。
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

    // 头部尺寸来自文件，使用带溢出检查的乘法，防止损坏文件导致分配尺寸回绕
    const num_pixels = try std.math.mul(usize, try std.math.mul(usize, num_images, rows), cols);
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

    pub const default: DataLoaderOptions = .{};
    pub fn defaultOptions() DataLoaderOptions {
        return .{};
    }
};

pub const DataLoader = struct {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: []usize,
    current_index: usize,
    prng: std.Random.DefaultPrng,

    pub fn initDefault(allocator: std.mem.Allocator, ds: Dataset, batch_size: usize) !DataLoader {
        return init(allocator, ds, batch_size, DataLoaderOptions.default);
    }

    pub fn init(allocator: std.mem.Allocator, ds: Dataset, batch_size: usize, options: DataLoaderOptions) !DataLoader {
        if (batch_size == 0) return error.InvalidBatchSize;
        const num_samples = ds.images.num_images;
        // 图像与标签条数不一致时，按图像下标取标签会越界
        if (ds.labels.num_items != num_samples) return error.DatasetSizeMismatch;
        const indices = try allocator.alloc(usize, num_samples);
        for (indices, 0..) |*idx, i| {
            idx.* = i;
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
        // 空数据集或单样本无需打乱 (同时避免 len - 1 下溢)
        if (self.indices.len < 2) return;
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
            @memcpy(x_dest[j * input_dim .. (j + 1) * input_dim], self.dataset.images.data[idx * input_dim .. (idx + 1) * input_dim]);
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
// 2. 字节级字节对编码分词器与内存映射数据集 (Byte-level Byte Pair Encoding, BPE & Memory-Mapped, MMap Dataset)
// ============================================================================

pub const TokenId = u32;
pub const MergePair = struct { a: TokenId, b: TokenId };

/// 按 (a, b) 字典序比较两个 Pair，用于字节对编码 (Byte Pair Encoding, BPE) 训练时确定性地打破频次平局
fn pairLessThan(lhs: MergePair, rhs: MergePair) bool {
    return lhs.a < rhs.a or (lhs.a == rhs.a and lhs.b < rhs.b);
}

/// 一条合并规则：优先级 (rank 越小越先合并) 与合并结果的 Token ID
pub const MergeRule = struct { rank: u32, id: TokenId };

/// 字节级字节对编码分词器 (Byte Pair Encoding Tokenizer, BPETokenizer)
pub const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(TokenId),
    inv_vocab: std.ArrayList([]const u8),
    merges: std.AutoHashMap(MergePair, MergeRule),

    pub fn init(allocator: std.mem.Allocator) !BPETokenizer {
        var self = BPETokenizer{
            .allocator = allocator,
            .vocab = std.StringHashMap(TokenId).init(allocator),
            .inv_vocab = .empty,
            .merges = std.AutoHashMap(MergePair, MergeRule).init(allocator),
        };
        errdefer self.deinit();

        // 初始化 256 个基本单字节 Token (0x00 ~ 0xFF)
        for (0..256) |b| {
            const byte = [1]u8{@as(u8, @intCast(b))};
            _ = try self.addToken(&byte);
        }
        return self;
    }

    pub fn deinit(self: *BPETokenizer) void {
        self.vocab.deinit();
        for (self.inv_vocab.items) |slice| {
            self.allocator.free(slice);
        }
        self.inv_vocab.deinit(self.allocator);
        self.merges.deinit();
    }

    /// 注册一个自定义/特殊 Token (如 <|bos|>, <|eos|>)；已存在时直接返回原 ID
    pub fn addToken(self: *BPETokenizer, token_str: []const u8) !TokenId {
        if (self.vocab.get(token_str)) |id| return id;

        const owned_str = try self.allocator.dupe(u8, token_str);
        errdefer self.allocator.free(owned_str);
        const new_id = @as(TokenId, @intCast(self.inv_vocab.items.len));
        try self.inv_vocab.append(self.allocator, owned_str);
        errdefer _ = self.inv_vocab.pop();
        try self.vocab.put(owned_str, new_id);
        return new_id;
    }

    /// 添加一条合并规则: (str_a, str_b) -> 优先级 rank，返回合并结果的 Token ID
    /// (合并结果已在词表中时复用原 ID，例如 "a"+"bc" 与 "ab"+"c" 都得到 "abc")
    pub fn addMerge(self: *BPETokenizer, str_a: []const u8, str_b: []const u8, rank: u32) !TokenId {
        const id_a = try self.addToken(str_a);
        const id_b = try self.addToken(str_b);

        const merged_str = try self.allocator.alloc(u8, str_a.len + str_b.len);
        defer self.allocator.free(merged_str);
        @memcpy(merged_str[0..str_a.len], str_a);
        @memcpy(merged_str[str_a.len..], str_b);

        const merged_id = try self.addToken(merged_str);
        try self.merges.put(.{ .a = id_a, .b = id_b }, .{ .rank = rank, .id = merged_id });
        return merged_id;
    }

    /// 在 `tokens` 上原地反复应用优先级最高 (rank 最小) 的合并规则，返回合并后的长度
    fn mergeInPlace(self: *const BPETokenizer, tokens: []TokenId) usize {
        var len = tokens.len;
        while (len >= 2) {
            var best: ?MergeRule = null;
            var best_idx: usize = 0;
            for (0..len - 1) |i| {
                if (self.merges.get(.{ .a = tokens[i], .b = tokens[i + 1] })) |rule| {
                    if (best == null or rule.rank < best.?.rank) {
                        best = rule;
                        best_idx = i;
                    }
                }
            }
            const rule = best orelse break;
            tokens[best_idx] = rule.id;
            std.mem.copyForwards(TokenId, tokens[best_idx + 1 .. len - 1], tokens[best_idx + 2 .. len]);
            len -= 1;
        }
        return len;
    }

    /// 将 UTF-8 文本编码为 Token ID 序列
    pub fn encode(self: *const BPETokenizer, allocator: std.mem.Allocator, text: []const u8) ![]TokenId {
        var result: std.ArrayList(TokenId) = .empty;
        defer result.deinit(allocator);
        // 字节级编码的 Token 数不超过字节数，一次性预留容量
        try result.ensureTotalCapacityPrecise(allocator, text.len);

        const chunk_size: usize = 256;
        var start: usize = 0;
        while (start < text.len) {
            var end = @min(start + chunk_size, text.len);
            if (end < text.len) {
                // 优先在后半段寻找空白字符 (空格、换行、制表符) 进行分块，保持单词完整性
                var found_ws: bool = false;
                var scan_idx = end;
                const min_split = start + chunk_size / 2;
                while (scan_idx > min_split) {
                    scan_idx -= 1;
                    const c = text[scan_idx];
                    if (c == ' ' or c == '\n' or c == '\t' or c == '\r') {
                        end = scan_idx + 1;
                        found_ws = true;
                        break;
                    }
                }

                // 若未找到空白字符，检测并避免截断多字节 UTF-8 字符序列 (UTF-8 延续字节: 10xxxxxx)
                if (!found_ws) {
                    while (end > start and (text[end] & 0xC0) == 0x80) {
                        end -= 1;
                    }
                    if (end == start) {
                        end = start + 1;
                        while (end < text.len and (text[end] & 0xC0) == 0x80) {
                            end += 1;
                        }
                    }
                }
            }

            // 先把本块的字节 Token 追加到结果尾部，再在尾部原地合并，无需逐块分配临时缓冲
            const chunk_start = result.items.len;
            for (text[start..end]) |byte| {
                result.appendAssumeCapacity(@as(TokenId, byte));
            }
            const merged_len = self.mergeInPlace(result.items[chunk_start..]);
            result.shrinkRetainingCapacity(chunk_start + merged_len);
            start = end;
        }

        return result.toOwnedSlice(allocator);
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

    /// 从语料库无监督自动统计最高频 Pair 并训练字节对编码 (Byte Pair Encoding, BPE) 合并规则，直到达到目标词表大小 target_vocab_size
    pub fn train(self: *BPETokenizer, allocator: std.mem.Allocator, corpus: []const u8, target_vocab_size: usize) !void {
        if (target_vocab_size <= self.inv_vocab.items.len) return;

        const tokens_buf = try self.encode(allocator, corpus);
        defer allocator.free(tokens_buf);
        var tokens = tokens_buf;

        var rank: u32 = @as(u32, @intCast(self.merges.count()));

        var pair_counts = std.AutoHashMap(MergePair, usize).init(allocator);
        defer pair_counts.deinit();

        while (self.inv_vocab.items.len < target_vocab_size) {
            if (tokens.len < 2) break;

            pair_counts.clearRetainingCapacity();
            for (0..tokens.len - 1) |i| {
                const pair = MergePair{ .a = tokens[i], .b = tokens[i + 1] };
                const entry = try pair_counts.getOrPut(pair);
                if (entry.found_existing) {
                    entry.value_ptr.* += 1;
                } else {
                    entry.value_ptr.* = 1;
                }
            }

            // 频次最高者胜出；频次相同时取 (a, b) 字典序最小的 Pair，
            // 使训练结果与哈希表迭代顺序无关，在不同运行与平台上完全确定
            var best_pair: ?MergePair = null;
            var max_freq: usize = 0;
            var it = pair_counts.iterator();
            while (it.next()) |entry| {
                const freq = entry.value_ptr.*;
                const candidate = entry.key_ptr.*;
                const better = freq > max_freq or
                    (freq == max_freq and best_pair != null and pairLessThan(candidate, best_pair.?));
                if (better) {
                    max_freq = freq;
                    best_pair = candidate;
                }
            }

            if (best_pair == null or max_freq <= 1) break;

            const pair = best_pair.?;
            const str_a = self.inv_vocab.items[pair.a];
            const str_b = self.inv_vocab.items[pair.b];
            // 合并结果可能已在词表中 (特殊 Token 或其他拆分方式得到的同一字符串)，必须使用 addMerge 返回的 ID
            const new_id = try self.addMerge(str_a, str_b, rank);
            rank += 1;

            // 原地压缩：写指针始终不超过读指针
            var write: usize = 0;
            var read: usize = 0;
            while (read < tokens.len) {
                if (read + 1 < tokens.len and tokens[read] == pair.a and tokens[read + 1] == pair.b) {
                    tokens[write] = new_id;
                    read += 2;
                } else {
                    tokens[write] = tokens[read];
                    read += 1;
                }
                write += 1;
            }
            tokens = tokens[0..write];
        }
    }
};

/// 零拷贝二进制 Token 数据集 (Binary Token Dataset)：只借用外部 Token 缓冲 (例如内存映射 (Memory-Mapped, MMap) 文件)，不持有内存
pub const BinaryMmapDataset = struct {
    tokens: []const u32,
    seq_len: usize,

    pub fn fromSlice(tokens: []const u32, seq_len: usize) BinaryMmapDataset {
        return .{
            .tokens = tokens,
            .seq_len = seq_len,
        };
    }

    /// 把按 u32 对齐的原始字节 (例如内存映射文件的页对齐映射区) 解释为 Token 序列；末尾不足 4 字节的部分被忽略
    pub fn loadFromBytes(bytes: []align(@alignOf(u32)) const u8, seq_len: usize) BinaryMmapDataset {
        const token_count = bytes.len / @sizeOf(u32);
        const tokens: [*]const u32 = @ptrCast(bytes.ptr);
        return .{
            .tokens = tokens[0..token_count],
            .seq_len = seq_len,
        };
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
