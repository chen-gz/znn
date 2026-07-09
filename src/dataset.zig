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

pub fn loadImages(allocator: std.mem.Allocator, file_path: []const u8) !ImageDataset {
    const cwd = std.fs.cwd();
    var file = try cwd.openFile(file_path, .{});
    defer file.close();

    var file_buf: [65536]u8 = undefined;
    var file_reader = file.reader(&file_buf);
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

pub fn loadLabels(allocator: std.mem.Allocator, file_path: []const u8) !LabelDataset {
    const cwd = std.fs.cwd();
    var file = try cwd.openFile(file_path, .{});
    defer file.close();

    var file_buf: [65536]u8 = undefined;
    var file_reader = file.reader(&file_buf);
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

pub fn loadDataset(allocator: std.mem.Allocator, images_path: []const u8, labels_path: []const u8) !Dataset {
    var images = try loadImages(allocator, images_path);
    errdefer images.deinit(allocator);

    var labels = try loadLabels(allocator, labels_path);
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

