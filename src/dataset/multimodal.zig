const std = @import("std");
const vision = @import("vision.zig");
const ImageDataset = vision.ImageDataset;
const createSyntheticTinyImageNet = vision.createSyntheticTinyImageNet;
const text_mod = @import("text.zig");
const TokenId = text_mod.TokenId;
const BPETokenizer = text_mod.BPETokenizer;

pub const MultimodalPair = struct {
    image: []const f32, // [channels * rows * cols]
    caption: []const u8,
    tokens: []const u32,
};

pub const MultimodalDataset = struct {
    images: ImageDataset,
    captions: [][]const u8,
    tokens: [][]const u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MultimodalDataset) void {
        self.images.deinit(self.allocator);
        for (self.captions) |cap| {
            self.allocator.free(cap);
        }
        self.allocator.free(self.captions);
        for (self.tokens) |toks| {
            self.allocator.free(toks);
        }
        self.allocator.free(self.tokens);
    }

    pub fn count(self: MultimodalDataset) usize {
        return @as(usize, self.images.num_images);
    }

    pub fn get(self: MultimodalDataset, idx: usize) MultimodalPair {
        const sample_len = self.images.sampleSize();
        return .{
            .image = self.images.data[idx * sample_len .. (idx + 1) * sample_len],
            .caption = self.captions[idx],
            .tokens = self.tokens[idx],
        };
    }
};

pub const MultimodalDataLoaderOptions = struct {
    shuffle: bool = false,
    seed: ?u64 = null,
    drop_last: bool = false,
    pad_token_id: u32 = 0,

    pub const default: MultimodalDataLoaderOptions = .{};
    pub fn defaultOptions() MultimodalDataLoaderOptions {
        return .{};
    }
};

pub const MultimodalDataLoader = struct {
    dataset: MultimodalDataset,
    batch_size: usize,
    max_seq_len: usize,
    pad_token_id: u32,
    shuffle: bool,
    drop_last: bool,
    indices: []usize,
    current_index: usize,
    prng: std.Random.DefaultPrng,

    pub fn initDefault(
        allocator: std.mem.Allocator,
        ds: MultimodalDataset,
        batch_size: usize,
        max_seq_len: usize,
    ) !MultimodalDataLoader {
        return init(allocator, ds, batch_size, max_seq_len, MultimodalDataLoaderOptions.default);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        ds: MultimodalDataset,
        batch_size: usize,
        max_seq_len: usize,
        options: MultimodalDataLoaderOptions,
    ) !MultimodalDataLoader {
        if (batch_size == 0) return error.InvalidBatchSize;
        const num_samples = ds.count();
        if (ds.captions.len != num_samples or ds.tokens.len != num_samples) {
            return error.DatasetSizeMismatch;
        }
        const indices = try allocator.alloc(usize, num_samples);
        for (indices, 0..) |*idx, i| {
            idx.* = i;
        }
        var self = MultimodalDataLoader{
            .dataset = ds,
            .batch_size = batch_size,
            .max_seq_len = max_seq_len,
            .pad_token_id = options.pad_token_id,
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

    pub fn deinit(self: *MultimodalDataLoader, allocator: std.mem.Allocator) void {
        allocator.free(self.indices);
    }

    pub fn shuffleIndices(self: *MultimodalDataLoader) void {
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

    pub fn reset(self: *MultimodalDataLoader) void {
        self.current_index = 0;
        if (self.shuffle) {
            self.shuffleIndices();
        }
    }

    pub fn nextInto(
        self: *MultimodalDataLoader,
        img_dest: []f32,
        tokens_dest: []u32,
        lengths_dest: []usize,
    ) ?usize {
        const num_samples = self.indices.len;
        if (self.current_index >= num_samples) return null;
        const remaining = num_samples - self.current_index;
        if (self.drop_last and remaining < self.batch_size) return null;
        const actual_batch_size = @min(self.batch_size, remaining);
        if (actual_batch_size == 0) return null;

        const sample_sz = self.dataset.images.sampleSize();
        @memset(tokens_dest[0 .. actual_batch_size * self.max_seq_len], self.pad_token_id);

        for (0..actual_batch_size) |j| {
            const idx = self.indices[self.current_index + j];
            const pair = self.dataset.get(idx);
            @memcpy(img_dest[j * sample_sz .. (j + 1) * sample_sz], pair.image);

            const copy_len = @min(pair.tokens.len, self.max_seq_len);
            @memcpy(tokens_dest[j * self.max_seq_len .. j * self.max_seq_len + copy_len], pair.tokens[0..copy_len]);
            lengths_dest[j] = copy_len;
        }

        self.current_index += actual_batch_size;
        return actual_batch_size;
    }
};

/// 构造用于测试与基准评测的多模态图文对数据集。
/// 包含 3 通道高分辨率图像与其对应的自然语言描述文本，并通过给定的分词器完成离散 Token 编码。
pub fn createSyntheticMultimodalDataset(
    allocator: std.mem.Allocator,
    tokenizer: *const BPETokenizer,
    num_samples: usize,
) !MultimodalDataset {
    var img_ds = try createSyntheticTinyImageNet(allocator, num_samples);
    errdefer img_ds.deinit(allocator);

    const captions = try allocator.alloc([]const u8, num_samples);
    var captions_allocated: usize = 0;
    errdefer {
        for (captions[0..captions_allocated]) |c| allocator.free(c);
        allocator.free(captions);
    }

    const token_arrays = try allocator.alloc([]const u32, num_samples);
    var tokens_allocated: usize = 0;
    errdefer {
        for (token_arrays[0..tokens_allocated]) |t| allocator.free(t);
        allocator.free(token_arrays);
    }

    const objects = [_][]const u8{ "cat", "dog", "car", "airplane", "ship", "tree", "mountain", "flower" };
    const colors = [_][]const u8{ "golden", "crimson", "emerald", "sapphire", "violet", "silver" };
    const scenes = [_][]const u8{ "under sunlight", "in front of the camera", "against a blue sky", "in the forest", "at sunset" };

    var buf: [256]u8 = undefined;
    for (0..num_samples) |i| {
        const obj = objects[i % objects.len];
        const col = colors[(i / objects.len) % colors.len];
        const scn = scenes[(i / (objects.len * colors.len)) % scenes.len];

        const text = try std.fmt.bufPrint(&buf, "A high quality picture of a {s} {s} {s}.", .{ col, obj, scn });
        captions[i] = try allocator.dupe(u8, text);
        captions_allocated += 1;

        const enc = try tokenizer.encode(allocator, captions[i]);
        token_arrays[i] = enc;
        tokens_allocated += 1;
    }

    img_ds.labels.deinit(allocator);

    return MultimodalDataset{
        .images = img_ds.images,
        .captions = captions,
        .tokens = token_arrays,
        .allocator = allocator,
    };
}

