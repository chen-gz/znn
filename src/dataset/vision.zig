const std = @import("std");

pub const ImageDataset = struct {
    num_images: u32,
    channels: u32 = 1, // 通道数：1 (灰度图, 如 MNIST/Fashion-MNIST), 3 (彩色 RGB, 如 CIFAR/Tiny ImageNet)
    rows: u32,
    cols: u32,
    data: []f32, // normalized between 0.0 and 1.0 (shape: num_images * channels * rows * cols)

    pub fn sampleSize(self: ImageDataset) usize {
        return @as(usize, self.channels) * @as(usize, self.rows) * @as(usize, self.cols);
    }

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

    var channels: u32 = 1;
    var num_images: u32 = 0;
    var rows: u32 = 0;
    var cols: u32 = 0;

    if (magic == 0x00000803) {
        // 3D IDX format: [num_images, rows, cols], channels = 1
        try reader.readSliceAll(&temp_4);
        num_images = std.mem.readInt(u32, &temp_4, .big);

        try reader.readSliceAll(&temp_4);
        rows = std.mem.readInt(u32, &temp_4, .big);

        try reader.readSliceAll(&temp_4);
        cols = std.mem.readInt(u32, &temp_4, .big);
    } else if (magic == 0x00000804) {
        // 4D IDX format: [num_images, channels, rows, cols]
        try reader.readSliceAll(&temp_4);
        num_images = std.mem.readInt(u32, &temp_4, .big);

        try reader.readSliceAll(&temp_4);
        channels = std.mem.readInt(u32, &temp_4, .big);

        try reader.readSliceAll(&temp_4);
        rows = std.mem.readInt(u32, &temp_4, .big);

        try reader.readSliceAll(&temp_4);
        cols = std.mem.readInt(u32, &temp_4, .big);
    } else {
        return error.InvalidMagicNumber;
    }

    // 头部尺寸来自文件，使用带溢出检查的乘法，防止损坏文件导致分配尺寸回绕
    const num_pixels = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, num_images, channels), rows), cols);
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
        .channels = channels,
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

/// 读取 Tiny ImageNet 紧凑二进制格式 (每张样本: 1 字节类别标签 [0..199] + 3*64*64=12288 字节 RGB 像素，CHW 格式)。
/// 像素字节按固定线性缩放到 [0, 1]：x = pixel / 255.0。
pub fn loadTinyImageNetBin(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    max_samples: ?usize,
) !Dataset {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    const file_len = try file.length(io);
    const bytes_per_record: usize = 1 + 3 * 64 * 64; // 12289
    if (file_len < bytes_per_record) {
        return error.InvalidDatasetFormat;
    }

    const total_records = file_len / bytes_per_record;
    const num_samples = if (max_samples) |m| @min(m, total_records) else total_records;
    if (num_samples == 0) return error.EmptyDataset;

    const img_data = try allocator.alloc(f32, num_samples * 3 * 64 * 64);
    errdefer allocator.free(img_data);
    const lbl_data = try allocator.alloc(u8, num_samples);
    errdefer allocator.free(lbl_data);

    var file_buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &file_buf);
    const reader = &file_reader.interface;

    var record_buf: [12289]u8 = undefined;
    for (0..num_samples) |i| {
        try reader.readSliceAll(&record_buf);
        lbl_data[i] = record_buf[0];
        const pixel_bytes = record_buf[1..12289];
        const dest_slice = img_data[i * 12288 .. (i + 1) * 12288];
        for (pixel_bytes, 0..) |p, p_idx| {
            dest_slice[p_idx] = @as(f32, @floatFromInt(p)) / 255.0;
        }
    }

    return Dataset{
        .images = .{
            .num_images = @as(u32, @intCast(num_samples)),
            .channels = 3,
            .rows = 64,
            .cols = 64,
            .data = img_data,
        },
        .labels = .{
            .num_items = @as(u32, @intCast(num_samples)),
            .data = lbl_data,
        },
    };
}

/// 快速构造用于测试与基准评测的合成 Tiny ImageNet 数据集 (3 通道 64x64，200 个语义类别)。
/// 每个类别生成具有区分度的空间几何与色彩频谱图案，用于在无外部大文件下载时完整验证 3 通道高分辨率模型。
pub fn createSyntheticTinyImageNet(allocator: std.mem.Allocator, num_samples: usize) !Dataset {
    const num_pixels = num_samples * 3 * 64 * 64;
    const img_data = try allocator.alloc(f32, num_pixels);
    errdefer allocator.free(img_data);
    const lbl_data = try allocator.alloc(u8, num_samples);
    errdefer allocator.free(lbl_data);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..num_samples) |i| {
        const class_id: u8 = @as(u8, @intCast(i % 200));
        lbl_data[i] = class_id;

        const img_slice = img_data[i * 12288 .. (i + 1) * 12288];
        const class_f = @as(f32, @floatFromInt(class_id)) / 200.0;
        for (0..64) |r| {
            for (0..64) |c| {
                const pixel_idx = r * 64 + c;
                const r_val = @as(f32, @floatFromInt(r)) / 64.0;
                const c_val = @as(f32, @floatFromInt(c)) / 64.0;

                // R 通道: 类别基准强度 + 随机微扰
                img_slice[0 * 4096 + pixel_idx] = std.math.clamp(class_f * 0.7 + random.float(f32) * 0.3, 0.0, 1.0);
                // G 通道: 空间斜向渐变
                img_slice[1 * 4096 + pixel_idx] = (r_val + c_val) * 0.5;
                // B 通道: 径向距离场 (中心高亮)
                const dr = r_val - 0.5;
                const dc = c_val - 0.5;
                const dist = @sqrt(dr * dr + dc * dc);
                img_slice[2 * 4096 + pixel_idx] = std.math.clamp(1.0 - dist, 0.0, 1.0);
            }
        }
    }

    return Dataset{
        .images = .{
            .num_images = @as(u32, @intCast(num_samples)),
            .channels = 3,
            .rows = 64,
            .cols = 64,
            .data = img_data,
        },
        .labels = .{
            .num_items = @as(u32, @intCast(num_samples)),
            .data = lbl_data,
        },
    };
}

