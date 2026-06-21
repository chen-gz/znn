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

pub fn loadImages(io: std.Io, allocator: std.mem.Allocator, file_path: []const u8) !ImageDataset {
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

pub fn loadLabels(io: std.Io, allocator: std.mem.Allocator, file_path: []const u8) !LabelDataset {
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
