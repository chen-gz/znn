//! ConvNeXt: A ConvNet for the 2020s (Liu et al. CVPR 2022)
//!
//! 现代化纯卷积视觉模型架构，吸收了 Vision Transformer (ViT / Swin) 的设计精髓：
//! 1. 深度可分离卷积 (Depthwise Separable Convolution, 7x7 DWConv)；
//! 2. 倒瓶颈结构 (Inverted Bottleneck, 4x 隐藏维度扩展)；
//! 3. 极简归一化与激活设计 (单层 LayerNorm + 单层 GELU)；
//! 4. 可学习层级缩放 (LayerScale) 与恒等残差连接；
//! 5. 独立下采样层 (LayerNorm2d + 2x2 Conv) 与分阶段层级设计。

const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Graph = autodiff.Graph;
const Shape = tensor.Shape;

// ============================================================================
// 1. ConvNeXt 基础残差块与下采样层
// ============================================================================

/// ConvNeXt 基础残差块 (Liu et al. 2022 §3)
///
/// 结构：
/// - 7x7 深度卷积 (Depthwise Conv2D, groups = dim, padding = 3)
/// - 维度转置为 Channels-Last [N, H, W, C]
/// - 层归一化 (LayerNorm, 沿末维 C 归一化)
/// - 点卷积 1x1 升维 (Linear: dim -> 4 * dim)
/// - 高斯误差线性单元 (GELU) 激活函数
/// - 点卷积 1x1 降维还原 (Linear: 4 * dim -> dim)
/// - 可选层级缩放 (LayerScale, 逐通道乘以 gamma 向量)
/// - 维度转置回 Channels-First [N, C, H, W]
/// - 恒等残差累加 (Residual Add: input + F(input))
pub const ConvNeXtBlock = struct {
    dwconv: nn.Conv2D,
    norm: nn.LayerNorm,
    pwconv1: nn.Linear,
    pwconv2: nn.Linear,
    gamma: ?*Tensor = null,
    dim: usize,
    name: ?[]const u8 = null,
    module_type: []const u8 = "ConvNeXtBlock",

    pub const Options = struct {
        layer_scale_init_value: f32 = 1e-6,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub fn defaultOptions() Options {
        return Options.defaultOptions();
    }

    pub fn init(allocator: std.mem.Allocator, dim: usize, options: Options) !ConvNeXtBlock {
        // 1. 7x7 深度卷积 (groups = dim)
        var dwconv = try nn.Conv2D.init(allocator, dim, dim, 7, .{
            .stride = 1,
            .padding = 3,
            .groups = dim,
        });
        errdefer nn.deinitModel(&dwconv, allocator);

        // 2. 沿通道维度的层归一化 (LayerNorm)
        var norm = try nn.LayerNorm.init(allocator, dim, 1e-6);
        errdefer nn.deinitModel(&norm, allocator);

        // 3. 点卷积扩展 (dim -> 4 * dim)
        var pwconv1 = try nn.Linear.init(allocator, dim, 4 * dim);
        errdefer nn.deinitModel(&pwconv1, allocator);

        // 4. 点卷积还原 (4 * dim -> dim)
        var pwconv2 = try nn.Linear.init(allocator, 4 * dim, dim);
        errdefer nn.deinitModel(&pwconv2, allocator);

        // 5. LayerScale 可学习通道缩放向量 [dim]
        var gamma: ?*Tensor = null;
        if (options.layer_scale_init_value > 0.0) {
            const g = try nn.createVectorTensor(allocator, dim, options.layer_scale_init_value, true);
            gamma = g;
        }

        return ConvNeXtBlock{
            .dwconv = dwconv,
            .norm = norm,
            .pwconv1 = pwconv1,
            .pwconv2 = pwconv2,
            .gamma = gamma,
            .dim = dim,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, dim: usize) !ConvNeXtBlock {
        return init(allocator, dim, .{});
    }

    pub fn forward(self: *const ConvNeXtBlock, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const residual = x;

        // 1. 7x7 深度可分离卷积
        const dw = try self.dwconv.forward(graph, x);

        // 2. 维度转置为 Channels-Last: [N, C, H, W] -> [N, H, C, W] -> [N, H, W, C]
        const p1 = try graph.transpose(dw, 1, 2);
        const p2 = try graph.transpose(p1, 2, 3);

        // 3. 层归一化
        const normed = try self.norm.forward(graph, p2);

        // 4. 1x1 点卷积升维 (dim -> 4 * dim)
        const pw1 = try self.pwconv1.forward(graph, normed);

        // 5. GELU 激活函数
        const act = try graph.gelu(pw1, .none);

        // 6. 1x1 点卷积降维还原 (4 * dim -> dim)
        var pw2 = try self.pwconv2.forward(graph, act);

        // 7. LayerScale 通道缩放
        if (self.gamma) |g| {
            pw2 = try graph.mul(pw2, g);
        }

        // 8. 维度转置回 Channels-First: [N, H, W, C] -> [N, H, C, W] -> [N, C, H, W]
        const r1 = try graph.transpose(pw2, 2, 3);
        const r2 = try graph.transpose(r1, 1, 2);

        // 9. 恒等残差相加
        return try graph.add(residual, r2);
    }
};

/// 阶段间下采样层 (Downsample Layer: LayerNorm2d + 2x2 Conv stride=2)
pub const DownsampleLayer = struct {
    norm: nn.LayerNorm2d,
    conv: nn.Conv2D,
    name: ?[]const u8 = null,
    module_type: []const u8 = "DownsampleLayer",

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize) !DownsampleLayer {
        var norm = try nn.LayerNorm2d.init(allocator, in_channels, .{});
        errdefer nn.deinitModel(&norm, allocator);

        var conv = try nn.Conv2D.init(allocator, in_channels, out_channels, 2, .{
            .stride = 2,
            .padding = 0,
        });
        errdefer nn.deinitModel(&conv, allocator);

        return DownsampleLayer{
            .norm = norm,
            .conv = conv,
        };
    }

    pub fn forward(self: *const DownsampleLayer, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const normed = try self.norm.forward(graph, x);
        return try self.conv.forward(graph, normed);
    }
};

// ============================================================================
// 2. 超参数配置
// ============================================================================

/// ConvNeXt 规格配置
pub const ConvNeXtConfig = struct {
    in_channels: usize = 3,
    num_classes: usize = 1000,
    depths: []const usize = &.{ 3, 3, 9, 3 },
    dims: []const usize = &.{ 96, 192, 384, 768 },
    stem_kernel_size: usize = 4,
    stem_stride: usize = 4,
    stem_padding: usize = 0,
    layer_scale_init_value: f32 = 1e-6,

    pub const default: ConvNeXtConfig = .{};

    pub fn defaultConfig() ConvNeXtConfig {
        return .{};
    }

    // ------------------------------------------------------------------------
    // ImageNet 系列规格 (Liu et al. 2022 Table 1)
    // ------------------------------------------------------------------------

    /// ConvNeXt-Tiny: [3, 3, 9, 3], 通道 [96, 192, 384, 768]
    pub fn convnextTiny(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 3, 3, 9, 3 },
            .dims = &.{ 96, 192, 384, 768 },
            .stem_kernel_size = 4,
            .stem_stride = 4,
            .stem_padding = 0,
        };
    }

    /// ConvNeXt-Small: [3, 3, 27, 3], 通道 [96, 192, 384, 768]
    pub fn convnextSmall(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 3, 3, 27, 3 },
            .dims = &.{ 96, 192, 384, 768 },
            .stem_kernel_size = 4,
            .stem_stride = 4,
            .stem_padding = 0,
        };
    }

    /// ConvNeXt-Base: [3, 3, 27, 3], 通道 [128, 256, 512, 1024]
    pub fn convnextBase(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 3, 3, 27, 3 },
            .dims = &.{ 128, 256, 512, 1024 },
            .stem_kernel_size = 4,
            .stem_stride = 4,
            .stem_padding = 0,
        };
    }

    /// ConvNeXt-Large: [3, 3, 27, 3], 通道 [192, 384, 768, 1536]
    pub fn convnextLarge(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 3, 3, 27, 3 },
            .dims = &.{ 192, 384, 768, 1536 },
            .stem_kernel_size = 4,
            .stem_stride = 4,
            .stem_padding = 0,
        };
    }

    // ------------------------------------------------------------------------
    // CIFAR / 小型端侧与基准评测规格
    // ------------------------------------------------------------------------

    /// ConvNeXt-Femto (端侧轻量): [2, 2, 6, 2], 通道 [48, 96, 192, 384]
    pub fn convnextFemto(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 2, 2, 6, 2 },
            .dims = &.{ 48, 96, 192, 384 },
            .stem_kernel_size = 2,
            .stem_stride = 2,
            .stem_padding = 0,
        };
    }

    /// ConvNeXt-Atto (极速端侧): [1, 1, 3, 1], 通道 [24, 48, 96, 192]
    pub fn convnextAtto(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 1, 1, 3, 1 },
            .dims = &.{ 24, 48, 96, 192 },
            .stem_kernel_size = 2,
            .stem_stride = 2,
            .stem_padding = 0,
        };
    }

    /// ConvNeXt-Pico (秒级测试与 CPU 基准): [1, 1, 1, 1], 通道 [16, 32, 64, 128]
    pub fn convnextPico(in_channels: usize, num_classes: usize) ConvNeXtConfig {
        return .{
            .in_channels = in_channels,
            .num_classes = num_classes,
            .depths = &.{ 1, 1, 1, 1 },
            .dims = &.{ 16, 32, 64, 128 },
            .stem_kernel_size = 2,
            .stem_stride = 2,
            .stem_padding = 0,
        };
    }
};

// ============================================================================
// 3. ConvNeXt 完整模型结构
// ============================================================================

/// ConvNeXt 完整视觉骨干与分类模型
pub const ConvNeXt = struct {
    stem_conv: nn.Conv2D,
    stem_norm: nn.LayerNorm2d,
    blocks: nn.ModuleList(ConvNeXtBlock),
    downsamples: nn.ModuleList(DownsampleLayer),
    stage_ranges: [][2]usize,
    norm: nn.LayerNorm,
    head: nn.Linear,
    in_channels: usize,
    num_classes: usize,
    allocator: std.mem.Allocator,
    name: ?[]const u8 = null,
    module_type: []const u8 = "ConvNeXt",

    pub fn init(allocator: std.mem.Allocator, config: ConvNeXtConfig) !ConvNeXt {
        if (config.depths.len != config.dims.len or config.depths.len == 0) {
            return error.ShapeMismatch;
        }

        // 1. Stem: Patchify (初步特征提取与下采样)
        var stem_conv = try nn.Conv2D.init(
            allocator,
            config.in_channels,
            config.dims[0],
            config.stem_kernel_size,
            .{
                .stride = config.stem_stride,
                .padding = config.stem_padding,
            },
        );
        errdefer nn.deinitModel(&stem_conv, allocator);

        var stem_norm = try nn.LayerNorm2d.init(allocator, config.dims[0], .{});
        errdefer nn.deinitModel(&stem_norm, allocator);

        // 2. 统计所有 stage 的总 block 数量与分配
        var total_blocks: usize = 0;
        for (config.depths) |d| total_blocks += d;

        const raw_blocks = try allocator.alloc(ConvNeXtBlock, total_blocks);
        errdefer allocator.free(raw_blocks);

        const stage_ranges = try allocator.alloc([2]usize, config.depths.len);
        errdefer allocator.free(stage_ranges);

        var b_idx: usize = 0;
        for (config.depths, config.dims, 0..) |depth, dim, stage_i| {
            const start_idx = b_idx;
            for (0..depth) |_| {
                raw_blocks[b_idx] = try ConvNeXtBlock.init(allocator, dim, .{
                    .layer_scale_init_value = config.layer_scale_init_value,
                });
                b_idx += 1;
            }
            stage_ranges[stage_i] = .{ start_idx, b_idx };
        }

        const blocks = nn.ModuleList(ConvNeXtBlock){
            .items = raw_blocks,
            .allocator = allocator,
        };

        // 3. 阶段间下采样层 (num_stages - 1 个)
        const num_downsamples = config.depths.len - 1;
        const raw_downsamples = try allocator.alloc(DownsampleLayer, num_downsamples);
        errdefer allocator.free(raw_downsamples);

        for (0..num_downsamples) |i| {
            raw_downsamples[i] = try DownsampleLayer.init(allocator, config.dims[i], config.dims[i + 1]);
        }

        const downsamples = nn.ModuleList(DownsampleLayer){
            .items = raw_downsamples,
            .allocator = allocator,
        };

        // 4. 分类头 (Head): AdaptiveAvgPool2d -> LayerNorm -> Linear
        const last_dim = config.dims[config.dims.len - 1];
        var norm = try nn.LayerNorm.init(allocator, last_dim, 1e-6);
        errdefer nn.deinitModel(&norm, allocator);

        var head = try nn.Linear.init(allocator, last_dim, config.num_classes);
        errdefer nn.deinitModel(&head, allocator);

        return ConvNeXt{
            .stem_conv = stem_conv,
            .stem_norm = stem_norm,
            .blocks = blocks,
            .downsamples = downsamples,
            .stage_ranges = stage_ranges,
            .norm = norm,
            .head = head,
            .in_channels = config.in_channels,
            .num_classes = config.num_classes,
            .allocator = allocator,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator) !ConvNeXt {
        return init(allocator, ConvNeXtConfig.defaultConfig());
    }

    pub fn deinit(self: *ConvNeXt) void {
        self.allocator.free(self.stage_ranges);
        nn.deinitModel(&self.head, self.allocator);
        nn.deinitModel(&self.norm, self.allocator);
        self.downsamples.deinit(self.allocator);
        self.blocks.deinit(self.allocator);
        nn.deinitModel(&self.stem_norm, self.allocator);
        nn.deinitModel(&self.stem_conv, self.allocator);
    }

    pub fn forward(self: *const ConvNeXt, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const batch_size = x.shape.dims[0];

        // 自动适配 2D 展平输入 [N, C*H*W] 与 4D 图像输入 [N, C, H, W]
        const x_4d = if (x.shape.len == 2) blk: {
            const total_elements = x.shape.dims[1];
            const spatial = @as(usize, @intFromFloat(@sqrt(@as(f32, @floatFromInt(total_elements / self.in_channels)))));
            break :blk try graph.reshape(x, &.{ batch_size, self.in_channels, spatial, spatial });
        } else x;

        // Stem: Patchify
        const c_stem = try self.stem_conv.forward(graph, x_4d);
        var h = try self.stem_norm.forward(graph, c_stem);

        // Stages
        for (0..self.stage_ranges.len) |stage_i| {
            if (stage_i > 0) {
                h = try self.downsamples.items[stage_i - 1].forward(graph, h);
            }
            const range = self.stage_ranges[stage_i];
            for (range[0]..range[1]) |blk_i| {
                h = try self.blocks.items[blk_i].forward(graph, h);
            }
        }

        // Head: 全局自适应平均池化 -> 展平 -> LayerNorm -> Linear 分类
        const pooled = try graph.adaptiveAvgPool2d(h, .{ 1, 1 });
        const flat_dim = pooled.shape.dims[1];
        const flat = try graph.reshape(pooled, &.{ batch_size, flat_dim });
        const normed = try self.norm.forward(graph, flat);
        return try self.head.forward(graph, normed);
    }
};

/// 具备标准生命周期、梯度与参数初始化管理的 ConvNeXt 模块
pub const DefaultConvNeXt = nn.Module(ConvNeXt);

// ============================================================================
// 4. 开箱即用的论文标准构造入口
// ============================================================================

/// 构造 ConvNeXt-Tiny 模型 (Liu et al. 2022 Table 1)
pub fn convnext_tiny(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextTiny(in_channels, num_classes));
}

/// 构造 ConvNeXt-Small 模型
pub fn convnext_small(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextSmall(in_channels, num_classes));
}

/// 构造 ConvNeXt-Base 模型
pub fn convnext_base(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextBase(in_channels, num_classes));
}

/// 构造 ConvNeXt-Large 模型
pub fn convnext_large(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextLarge(in_channels, num_classes));
}

/// 构造 ConvNeXt-Femto 模型 (端侧轻量)
pub fn convnext_femto(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextFemto(in_channels, num_classes));
}

/// 构造 ConvNeXt-Atto 模型 (极速端侧)
pub fn convnext_atto(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextAtto(in_channels, num_classes));
}

/// 构造 ConvNeXt-Pico 模型 (秒级测试与 CPU 基准)
pub fn convnext_pico(allocator: std.mem.Allocator, in_channels: usize, num_classes: usize) !ConvNeXt {
    return ConvNeXt.init(allocator, ConvNeXtConfig.convnextPico(in_channels, num_classes));
}
