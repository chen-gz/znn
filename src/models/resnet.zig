//! ResNet (Residual Network) - 残差卷积神经网络架构
//!
//! 包含标准残差块 (ResidualBlock / BasicBlock) 与模块化 ResNet 网络定义，
//! 支持 1D/2D 图像输入、自适应平均池化、批量归一化 (BatchNorm2d) 与残差捷径连接。

const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Graph = autodiff.Graph;

/// 残差块 (Residual Block / BasicBlock)
///
/// 经典残差单元结构：
/// - 主路径：Conv2D(3x3) -> BatchNorm2d -> ReLU -> Conv2D(3x3) -> BatchNorm2d
/// - 捷径路径 (Shortcut)：当 stride != 1 或 in_channels != out_channels 时，使用 1x1 Conv2D + BatchNorm2d 降采样/升维匹配；
///   否则为恒等映射 (Identity)。
/// - 合并：ReLU(F(x) + Shortcut(x))
pub const ResidualBlock = struct {
    conv1: nn.Conv2D,
    bn1: nn.BatchNorm2d,
    conv2: nn.Conv2D,
    bn2: nn.BatchNorm2d,
    shortcut_conv: ?nn.Conv2D = null,
    shortcut_bn: ?nn.BatchNorm2d = null,
    stride: usize = 1,
    name: ?[]const u8 = null,
    module_type: []const u8 = "ResidualBlock",

    pub const Options = struct {
        stride: usize = 1,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        options: Options,
    ) !ResidualBlock {
        const stride = options.stride;
        var conv1 = try nn.Conv2D.init(allocator, in_channels, out_channels, 3, .{
            .stride = stride,
            .padding = 1,
        });
        errdefer nn.deinitModel(&conv1, allocator);

        var bn1 = try nn.BatchNorm2d.init(allocator, out_channels, .{});
        errdefer nn.deinitModel(&bn1, allocator);

        var conv2 = try nn.Conv2D.init(allocator, out_channels, out_channels, 3, .{
            .stride = 1,
            .padding = 1,
        });
        errdefer nn.deinitModel(&conv2, allocator);

        var bn2 = try nn.BatchNorm2d.init(allocator, out_channels, .{});
        errdefer nn.deinitModel(&bn2, allocator);

        var shortcut_conv: ?nn.Conv2D = null;
        var shortcut_bn: ?nn.BatchNorm2d = null;
        if (stride != 1 or in_channels != out_channels) {
            shortcut_conv = try nn.Conv2D.init(allocator, in_channels, out_channels, 1, .{
                .stride = stride,
                .padding = 0,
            });
            errdefer if (shortcut_conv) |*sc| nn.deinitModel(sc, allocator);

            shortcut_bn = try nn.BatchNorm2d.init(allocator, out_channels, .{});
            errdefer if (shortcut_bn) |*sbn| nn.deinitModel(sbn, allocator);
        }

        return .{
            .conv1 = conv1,
            .bn1 = bn1,
            .conv2 = conv2,
            .bn2 = bn2,
            .shortcut_conv = shortcut_conv,
            .shortcut_bn = shortcut_bn,
            .stride = stride,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !ResidualBlock {
        return init(allocator, channels, channels, .{});
    }

    pub fn forward(self: *const ResidualBlock, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        // 主残差路径 (Main residual path): Conv2D -> BatchNorm2d -> ReLU -> Conv2D -> BatchNorm2d
        const c1 = try self.conv1.forward(graph, x);
        const b1 = try self.bn1.forward(graph, c1);
        const a1 = try graph.relu(b1);

        const c2 = try self.conv2.forward(graph, a1);
        const b2 = try self.bn2.forward(graph, c2);

        // 捷径分支 (Shortcut / Identity path)
        const residual = if (self.shortcut_conv) |*sc| blk: {
            const sc_out = try sc.forward(graph, x);
            break :blk try self.shortcut_bn.?.forward(graph, sc_out);
        } else x;

        // 残差累加与激活: ReLU(F(x) + Shortcut(x))
        const added = try graph.add(b2, residual);
        return try graph.relu(added);
    }
};

/// ResNet 各阶段配置
pub const StageConfig = struct {
    channels: usize,
    num_blocks: usize,
    stride: usize = 1,
};

/// ResNet 超参数配置
pub const ResNetConfig = struct {
    in_channels: usize = 1,
    num_classes: usize = 10,
    base_channels: usize = 16,
    stages: []const StageConfig = &.{
        .{ .channels = 16, .num_blocks = 2, .stride = 1 },
        .{ .channels = 32, .num_blocks = 2, .stride = 2 },
    },

    pub const default: ResNetConfig = .{};

    pub fn defaultConfig() ResNetConfig {
        return .{};
    }

    /// 小型 ResNet (用于 Fashion-MNIST / CIFAR 等轻量级实验)
    pub fn resnet9(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 1, .stride = 1 },
                .{ .channels = 32, .num_blocks = 1, .stride = 2 },
                .{ .channels = 64, .num_blocks = 1, .stride = 2 },
            },
        };
    }

    /// 标准 ResNet-18 配置 (4 个阶段，每阶段 2 个残差块)
    pub fn resnet18(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 64,
            .stages = &.{
                .{ .channels = 64, .num_blocks = 2, .stride = 1 },
                .{ .channels = 128, .num_blocks = 2, .stride = 2 },
                .{ .channels = 256, .num_blocks = 2, .stride = 2 },
                .{ .channels = 512, .num_blocks = 2, .stride = 2 },
            },
        };
    }
};

/// ResNet 残差卷积神经网络主模型
pub const ResNet = struct {
    conv1: nn.Conv2D,
    bn1: nn.BatchNorm2d,
    blocks: nn.ModuleList(ResidualBlock),
    fc: nn.Linear,
    in_channels: usize,
    num_classes: usize,
    name: ?[]const u8 = null,
    module_type: []const u8 = "ResNet",

    pub fn init(allocator: std.mem.Allocator, config: ResNetConfig) !ResNet {
        // Stem 初步特征提取卷积: [N, in_channels, H, W] -> [N, base_channels, H, W]
        var conv1 = try nn.Conv2D.init(allocator, config.in_channels, config.base_channels, 3, .{
            .stride = 1,
            .padding = 1,
        });
        errdefer nn.deinitModel(&conv1, allocator);

        var bn1 = try nn.BatchNorm2d.init(allocator, config.base_channels, .{});
        errdefer nn.deinitModel(&bn1, allocator);

        // 构建残差阶段 (Residual Stages)
        var total_blocks: usize = 0;
        for (config.stages) |st| total_blocks += st.num_blocks;

        const raw_blocks = try allocator.alloc(ResidualBlock, total_blocks);
        errdefer allocator.free(raw_blocks);

        var cur_channels = config.base_channels;
        var b_idx: usize = 0;
        for (config.stages) |st| {
            for (0..st.num_blocks) |blk_i| {
                const stride = if (blk_i == 0) st.stride else 1;
                raw_blocks[b_idx] = try ResidualBlock.init(allocator, cur_channels, st.channels, .{ .stride = stride });
                cur_channels = st.channels;
                b_idx += 1;
            }
        }

        const blocks = nn.ModuleList(ResidualBlock){
            .items = raw_blocks,
            .allocator = allocator,
        };

        // 最终分类全连接层 (Linear Classifier): [N, last_channels] -> [N, num_classes]
        const last_channels = if (config.stages.len > 0) config.stages[config.stages.len - 1].channels else config.base_channels;
        var fc = try nn.Linear.init(allocator, last_channels, config.num_classes);
        errdefer nn.deinitModel(&fc, allocator);

        return .{
            .conv1 = conv1,
            .bn1 = bn1,
            .blocks = blocks,
            .fc = fc,
            .in_channels = config.in_channels,
            .num_classes = config.num_classes,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator) !ResNet {
        return init(allocator, ResNetConfig.defaultConfig());
    }

    pub fn forward(self: *const ResNet, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const batch_size = x.shape.dims[0];

        // 允许输入为展平形式 [N, 784] 或四维形式 [N, C, H, W]
        const x_4d = if (x.shape.len == 2) blk: {
            const total_elements = x.shape.dims[1];
            const spatial = @as(usize, @intFromFloat(@sqrt(@as(f32, @floatFromInt(total_elements / self.in_channels)))));
            break :blk try graph.reshape(x, &.{ batch_size, self.in_channels, spatial, spatial });
        } else x;

        // Stem: Conv2D -> BatchNorm2d -> ReLU
        const c1 = try self.conv1.forward(graph, x_4d);
        const b1 = try self.bn1.forward(graph, c1);
        var h = try graph.relu(b1);

        // 残差块序列 (Residual Blocks)
        for (self.blocks.items) |*block| {
            h = try block.forward(graph, h);
        }

        // 全局自适应平均池化 (Global Adaptive Average Pooling): [N, C, H, W] -> [N, C, 1, 1]
        const pooled = try graph.adaptiveAvgPool2d(h, .{ 1, 1 });

        // 展平为二维矩阵: [N, C, 1, 1] -> [N, C]
        const flat_dim = pooled.shape.dims[1];
        const flat = try graph.reshape(pooled, &.{ batch_size, flat_dim });

        // 全连接分类头: [N, C] -> [N, num_classes]
        return try self.fc.forward(graph, flat);
    }
};

/// 具备标准生命周期、梯度与参数初始化管理的 ResNet 模块
pub const DefaultResNet = nn.Module(ResNet);
