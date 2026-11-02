const std = @import("std");

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

// ============================================================================
// 3. 多模态图文对数据集 (Multimodal Vision-Language Dataset)
// ============================================================================

