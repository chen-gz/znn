const std = @import("std");
const vision = @import("vision.zig");
const ImageDataset = vision.ImageDataset;
const LabelDataset = vision.LabelDataset;
const Dataset = vision.Dataset;

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

        const input_dim = self.dataset.images.sampleSize();
        for (0..actual_batch_size) |j| {
            const idx = self.indices[self.current_index + j];
            @memcpy(x_dest[j * input_dim .. (j + 1) * input_dim], self.dataset.images.data[idx * input_dim .. (idx + 1) * input_dim]);
            y_dest[j] = self.dataset.labels.data[idx];
        }
        self.current_index += actual_batch_size;
        return actual_batch_size;
    }
};

