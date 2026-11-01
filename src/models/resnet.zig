//! ResNet (Deep Residual Learning for Image Recognition, He et al. 2015)
//!
//! 完整实现原始论文中的两大核心残差结构与各大经典规格：
//!
//! 1. 残差块结构：
//!    - `BasicBlock` (标准两层 3x3 残差块，用于 ResNet-18/34 及 CIFAR-10 系列)
//!    - `Bottleneck` (三层 1x1 -> 3x3 -> 1x1 瓶颈残差块，扩展系数 expansion=4，用于 ResNet-50/101/152)
//!
//! 2. 原始论文标准规格配置：
//!    - ImageNet 系列 (Table 1):
//!      - `ResNet-18`  (BasicBlock: [2, 2, 2, 2], channels: 64->128->256->512)
//!      - `ResNet-34`  (BasicBlock: [3, 4, 6, 3], channels: 64->128->256->512)
//!      - `ResNet-50`  (Bottleneck: [3, 4, 6, 3], channels: 64->128->256->512, out: 256->512->1024->2048)
//!      - `ResNet-101` (Bottleneck: [3, 4, 23, 3])
//!      - `ResNet-152` (Bottleneck: [3, 8, 36, 3])
//!    - CIFAR-10 系列 (Section 4.2):
//!      - `ResNet-20`  (BasicBlock: [3, 3, 3], channels: 16->32->64)
//!      - `ResNet-32`  (BasicBlock: [5, 5, 5])
//!      - `ResNet-44`  (BasicBlock: [7, 7, 7])
//!      - `ResNet-56`  (BasicBlock: [9, 9, 9])
//!      - `ResNet-110` (BasicBlock: [18, 18, 18])
//!      - `ResNet-1202`(BasicBlock: [200, 200, 200])
//!    - 极速演示系列:
//!      - `ResNet-9`   (BasicBlock: [1, 1, 1], channels: 16->32->64)

const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Graph = autodiff.Graph;

// ============================================================================
// 1. 核心残差单元: BasicBlock 与 Bottleneck
// ============================================================================

/// 基础两层残差块 (BasicBlock, He et al. 2015 §3.3)
///
/// 结构：
/// - Conv2D(3x3) -> BatchNorm2d -> ReLU -> Conv2D(3x3) -> BatchNorm2d
/// - 捷径 (Shortcut): 当 stride != 1 或 in_channels != out_channels 时，使用 1x1 卷积 + BatchNorm2d 升维/降采样；
///   否则为纯恒等映射 (Identity)。
/// - 合并: ReLU(F(x) + Shortcut(x))
pub const BasicBlock = struct {
    conv1: nn.Conv2D,
    bn1: nn.BatchNorm2d,
    conv2: nn.Conv2D,
    bn2: nn.BatchNorm2d,
    shortcut_conv: ?nn.Conv2D = null,
    shortcut_bn: ?nn.BatchNorm2d = null,
    stride: usize = 1,
    name: ?[]const u8 = null,
    module_type: []const u8 = "BasicBlock",

    pub const expansion: usize = 1;

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
        mid_channels: usize,
        options: Options,
    ) !BasicBlock {
        const stride = options.stride;
        const out_channels = mid_channels * expansion;

        var conv1 = try nn.Conv2D.init(allocator, in_channels, mid_channels, 3, .{
            .stride = stride,
            .padding = 1,
        });
        errdefer nn.deinitModel(&conv1, allocator);

        var bn1 = try nn.BatchNorm2d.init(allocator, mid_channels, .{});
        errdefer nn.deinitModel(&bn1, allocator);

        var conv2 = try nn.Conv2D.init(allocator, mid_channels, out_channels, 3, .{
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

    pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !BasicBlock {
        return init(allocator, channels, channels, .{});
    }

    pub fn forward(self: *const BasicBlock, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const c1 = try self.conv1.forward(graph, x);
        const b1 = try self.bn1.forward(graph, c1);
        const a1 = try graph.relu(b1);

        const c2 = try self.conv2.forward(graph, a1);
        const b2 = try self.bn2.forward(graph, c2);

        const residual = if (self.shortcut_conv) |*sc| blk: {
            const sc_out = try sc.forward(graph, x);
            break :blk try self.shortcut_bn.?.forward(graph, sc_out);
        } else x;

        const added = try graph.add(b2, residual);
        return try graph.relu(added);
    }
};

/// 兼容别名：ResidualBlock 等价于 BasicBlock
pub const ResidualBlock = BasicBlock;

/// 瓶颈三层残差块 (Bottleneck, He et al. 2015 §3.3)
///
/// 专为深层网络设计的计算效率瓶颈块：
/// - 1x1 Conv(in_channels -> mid_channels): 降维压缩
/// - 3x3 Conv(mid_channels -> mid_channels): 空间特征提取 (可带 stride 降采样)
/// - 1x1 Conv(mid_channels -> mid_channels * 4): 升维还原 (expansion = 4)
/// - 捷径 (Shortcut): 当 stride != 1 或 in_channels != mid_channels * 4 时，使用 1x1 卷积 + BatchNorm2d 升维；
///   否则为恒等映射。
/// - 合并: ReLU(F(x) + Shortcut(x))
pub const Bottleneck = struct {
    conv1: nn.Conv2D,
    bn1: nn.BatchNorm2d,
    conv2: nn.Conv2D,
    bn2: nn.BatchNorm2d,
    conv3: nn.Conv2D,
    bn3: nn.BatchNorm2d,
    shortcut_conv: ?nn.Conv2D = null,
    shortcut_bn: ?nn.BatchNorm2d = null,
    stride: usize = 1,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Bottleneck",

    pub const expansion: usize = 4;

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
        mid_channels: usize,
        options: Options,
    ) !Bottleneck {
        const stride = options.stride;
        const out_channels = mid_channels * expansion;

        // 1. 1x1 降维
        var conv1 = try nn.Conv2D.init(allocator, in_channels, mid_channels, 1, .{
            .stride = 1,
            .padding = 0,
        });
        errdefer nn.deinitModel(&conv1, allocator);

        var bn1 = try nn.BatchNorm2d.init(allocator, mid_channels, .{});
        errdefer nn.deinitModel(&bn1, allocator);

        // 2. 3x3 空间卷积 (可带 stride 降采样)
        var conv2 = try nn.Conv2D.init(allocator, mid_channels, mid_channels, 3, .{
            .stride = stride,
            .padding = 1,
        });
        errdefer nn.deinitModel(&conv2, allocator);

        var bn2 = try nn.BatchNorm2d.init(allocator, mid_channels, .{});
        errdefer nn.deinitModel(&bn2, allocator);

        // 3. 1x1 升维 (恢复为 4x mid_channels)
        var conv3 = try nn.Conv2D.init(allocator, mid_channels, out_channels, 1, .{
            .stride = 1,
            .padding = 0,
        });
        errdefer nn.deinitModel(&conv3, allocator);

        var bn3 = try nn.BatchNorm2d.init(allocator, out_channels, .{});
        errdefer nn.deinitModel(&bn3, allocator);

        // 4. 捷径分支 (Shortcut)
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
            .conv3 = conv3,
            .bn3 = bn3,
            .shortcut_conv = shortcut_conv,
            .shortcut_bn = shortcut_bn,
            .stride = stride,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !Bottleneck {
        return init(allocator, channels * expansion, channels, .{});
    }

    pub fn forward(self: *const Bottleneck, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        // 1x1 降维
        const c1 = try self.conv1.forward(graph, x);
        const b1 = try self.bn1.forward(graph, c1);
        const a1 = try graph.relu(b1);

        // 3x3 空间卷积
        const c2 = try self.conv2.forward(graph, a1);
        const b2 = try self.bn2.forward(graph, c2);
        const a2 = try graph.relu(b2);

        // 1x1 升维
        const c3 = try self.conv3.forward(graph, a2);
        const b3 = try self.bn3.forward(graph, c3);

        // 捷径分支
        const residual = if (self.shortcut_conv) |*sc| blk: {
            const sc_out = try sc.forward(graph, x);
            break :blk try self.shortcut_bn.?.forward(graph, sc_out);
        } else x;

        // 残差累加与最终激活
        const added = try graph.add(b3, residual);
        return try graph.relu(added);
    }
};

// ============================================================================
// 2. 网络阶段与超参数配置
// ============================================================================

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

    // ------------------------------------------------------------------------
    // ImageNet 系列规格 (He et al. 2015, Table 1)
    // ------------------------------------------------------------------------

    /// ResNet-18 (ImageNet): 4 阶段，[2, 2, 2, 2] 块
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

    /// ResNet-34 (ImageNet): 4 阶段，[3, 4, 6, 3] 块
    pub fn resnet34(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 64,
            .stages = &.{
                .{ .channels = 64, .num_blocks = 3, .stride = 1 },
                .{ .channels = 128, .num_blocks = 4, .stride = 2 },
                .{ .channels = 256, .num_blocks = 6, .stride = 2 },
                .{ .channels = 512, .num_blocks = 3, .stride = 2 },
            },
        };
    }

    /// ResNet-50 (ImageNet, 采用 Bottleneck): 4 阶段，[3, 4, 6, 3] 块
    pub fn resnet50(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 64,
            .stages = &.{
                .{ .channels = 64, .num_blocks = 3, .stride = 1 },
                .{ .channels = 128, .num_blocks = 4, .stride = 2 },
                .{ .channels = 256, .num_blocks = 6, .stride = 2 },
                .{ .channels = 512, .num_blocks = 3, .stride = 2 },
            },
        };
    }

    /// ResNet-101 (ImageNet, 采用 Bottleneck): 4 阶段，[3, 4, 23, 3] 块
    pub fn resnet101(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 64,
            .stages = &.{
                .{ .channels = 64, .num_blocks = 3, .stride = 1 },
                .{ .channels = 128, .num_blocks = 4, .stride = 2 },
                .{ .channels = 256, .num_blocks = 23, .stride = 2 },
                .{ .channels = 512, .num_blocks = 3, .stride = 2 },
            },
        };
    }

    /// ResNet-152 (ImageNet, 采用 Bottleneck): 4 阶段，[3, 8, 36, 3] 块
    pub fn resnet152(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 64,
            .stages = &.{
                .{ .channels = 64, .num_blocks = 3, .stride = 1 },
                .{ .channels = 128, .num_blocks = 8, .stride = 2 },
                .{ .channels = 256, .num_blocks = 36, .stride = 2 },
                .{ .channels = 512, .num_blocks = 3, .stride = 2 },
            },
        };
    }

    // ------------------------------------------------------------------------
    // CIFAR-10 系列规格 (He et al. 2015 §4.2, 3 个阶段，通道 16->32->64)
    // ------------------------------------------------------------------------

    /// ResNet-20 (CIFAR-10, n=3): 3 阶段各 3 块 ([3, 3, 3])
    pub fn resnet20(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 3, .stride = 1 },
                .{ .channels = 32, .num_blocks = 3, .stride = 2 },
                .{ .channels = 64, .num_blocks = 3, .stride = 2 },
            },
        };
    }

    /// ResNet-32 (CIFAR-10, n=5): 3 阶段各 5 块 ([5, 5, 5])
    pub fn resnet32(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 5, .stride = 1 },
                .{ .channels = 32, .num_blocks = 5, .stride = 2 },
                .{ .channels = 64, .num_blocks = 5, .stride = 2 },
            },
        };
    }

    /// ResNet-44 (CIFAR-10, n=7): 3 阶段各 7 块 ([7, 7, 7])
    pub fn resnet44(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 7, .stride = 1 },
                .{ .channels = 32, .num_blocks = 7, .stride = 2 },
                .{ .channels = 64, .num_blocks = 7, .stride = 2 },
            },
        };
    }

    /// ResNet-56 (CIFAR-10, n=9): 3 阶段各 9 块 ([9, 9, 9])
    pub fn resnet56(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 9, .stride = 1 },
                .{ .channels = 32, .num_blocks = 9, .stride = 2 },
                .{ .channels = 64, .num_blocks = 9, .stride = 2 },
            },
        };
    }

    /// ResNet-110 (CIFAR-10, n=18): 3 阶段各 18 块 ([18, 18, 18])
    pub fn resnet110(in_channels: usize, num_classes: usize) ResNetConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .base_channels = 16,
            .stages = &.{
                .{ .channels = 16, .num_blocks = 18, .stride = 1 },
                .{ .channels = 32, .num_blocks = 18, .stride = 2 },
                .{ .channels = 64, .num_blocks = 18, .stride = 2 },
            },
        };
    }

    // ------------------------------------------------------------------------
    // 小型与演示规格
    // ------------------------------------------------------------------------

    /// 小型极速演示 ResNet-9 (3 个阶段各 1 块，共 9 层，适合 CPU 秒级基准评测)
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
};

// ============================================================================
// 3. 通用泛型 ResNet 模型结构定义
// ============================================================================

/// 泛型 ResNet 模型容器，支持 BasicBlock 与 Bottleneck 自由组装
pub fn GenericResNet(comptime Block: type) type {
    return struct {
        const Self = @This();
        pub const BlockType = Block;

        conv1: nn.Conv2D,
        bn1: nn.BatchNorm2d,
        blocks: nn.ModuleList(Block),
        fc: nn.Linear,
        in_channels: usize,
        num_classes: usize,
        name: ?[]const u8 = null,
        module_type: []const u8 = "ResNet",

        pub fn init(allocator: std.mem.Allocator, config: ResNetConfig) !Self {
            // Stem 初步特征提取卷积: [N, in_channels, H, W] -> [N, base_channels, H, W]
            var conv1 = try nn.Conv2D.init(allocator, config.in_channels, config.base_channels, 3, .{
                .stride = 1,
                .padding = 1,
            });
            errdefer nn.deinitModel(&conv1, allocator);

            var bn1 = try nn.BatchNorm2d.init(allocator, config.base_channels, .{});
            errdefer nn.deinitModel(&bn1, allocator);

            // 统计各阶段总块数
            var total_blocks: usize = 0;
            for (config.stages) |st| total_blocks += st.num_blocks;

            const raw_blocks = try allocator.alloc(Block, total_blocks);
            errdefer allocator.free(raw_blocks);

            var cur_channels = config.base_channels;
            var b_idx: usize = 0;
            for (config.stages) |st| {
                for (0..st.num_blocks) |blk_i| {
                    const stride = if (blk_i == 0) st.stride else 1;
                    raw_blocks[b_idx] = try Block.init(allocator, cur_channels, st.channels, .{ .stride = stride });
                    cur_channels = st.channels * Block.expansion;
                    b_idx += 1;
                }
            }

            const blocks = nn.ModuleList(Block){
                .items = raw_blocks,
                .allocator = allocator,
            };

            // 最终全连接分类头: [N, last_channels] -> [N, num_classes]
            const last_channels = if (config.stages.len > 0)
                config.stages[config.stages.len - 1].channels * Block.expansion
            else
                config.base_channels;
            var fc = try nn.Linear.init(allocator, last_channels, config.num_classes);
            errdefer nn.deinitModel(&fc, allocator);

            return Self{
                .conv1 = conv1,
                .bn1 = bn1,
                .blocks = blocks,
                .fc = fc,
                .in_channels = config.in_channels,
                .num_classes = config.num_classes,
            };
        }

        pub fn initDefault(allocator: std.mem.Allocator) !Self {
            return init(allocator, ResNetConfig.defaultConfig());
        }

        pub fn forward(self: *const Self, graph: *Graph, x: *Tensor) !*Tensor {
            const module_scope = try nn.enterModuleScope(graph, self);
            defer module_scope.exit();

            const batch_size = x.shape.dims[0];

            // 自动支持二维展平输入 [N, C*H*W] 与四维图像输入 [N, C, H, W]
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

            // 展平为二维分类特征: [N, C, 1, 1] -> [N, C]
            const flat_dim = pooled.shape.dims[1];
            const flat = try graph.reshape(pooled, &.{ batch_size, flat_dim });

            // 全连接分类头: [N, C] -> [N, num_classes]
            return try self.fc.forward(graph, flat);
        }
    };
}

/// 标准 BasicBlock ResNet (如 ResNet-18, ResNet-34, ResNet-20/32/56)
pub const ResNet = GenericResNet(BasicBlock);

/// 标准 Bottleneck ResNet (如 ResNet-50, ResNet-101, ResNet-152)
pub const ResNetBottleneck = GenericResNet(Bottleneck);

/// 具备标准生命周期、梯度与参数初始化管理的 ResNet 模块
pub const DefaultResNet = nn.Module(ResNet);
pub const DefaultResNetBottleneck = nn.Module(ResNetBottleneck);

// ============================================================================
// 4. 开箱即用的论文标准构造入口
// ============================================================================

/// 构造 ResNet-18 模型 (BasicBlock: [2, 2, 2, 2])
pub fn resnet18(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet18(in_channels, num_classes));
}

/// 构造 ResNet-34 模型 (BasicBlock: [3, 4, 6, 3])
pub fn resnet34(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet34(in_channels, num_classes));
}

/// 构造 ResNet-50 模型 (Bottleneck: [3, 4, 6, 3], expansion=4)
pub fn resnet50(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNetBottleneck {
    return ResNetBottleneck.init(allocator, ResNetConfig.resnet50(in_channels, num_classes));
}

/// 构造 ResNet-101 模型 (Bottleneck: [3, 4, 23, 3], expansion=4)
pub fn resnet101(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNetBottleneck {
    return ResNetBottleneck.init(allocator, ResNetConfig.resnet101(in_channels, num_classes));
}

/// 构造 ResNet-152 模型 (Bottleneck: [3, 8, 36, 3], expansion=4)
pub fn resnet152(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNetBottleneck {
    return ResNetBottleneck.init(allocator, ResNetConfig.resnet152(in_channels, num_classes));
}

/// 构造 CIFAR-10 ResNet-20 模型 (BasicBlock: [3, 3, 3])
pub fn resnet20(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet20(in_channels, num_classes));
}

/// 构造 CIFAR-10 ResNet-32 模型 (BasicBlock: [5, 5, 5])
pub fn resnet32(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet32(in_channels, num_classes));
}

/// 构造 CIFAR-10 ResNet-56 模型 (BasicBlock: [9, 9, 9])
pub fn resnet56(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet56(in_channels, num_classes));
}

/// 构造 CIFAR-10 ResNet-110 模型 (BasicBlock: [18, 18, 18])
pub fn resnet110(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ResNet {
    return ResNet.init(allocator, ResNetConfig.resnet110(in_channels, num_classes));
}
