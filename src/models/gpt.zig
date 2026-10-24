//! 生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型架构定义
//!
//! 包含标准 GPT 模型配置 (`GPTConfig`)、堆叠变换器解码器 (`TransformerDecoder`) 与通用模型生成器 (`GPT`)。

const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Embedding = nn.Embedding;
const TransformerBlock = nn.TransformerBlock;
const RMSNorm = nn.RMSNorm;
const Linear = nn.Linear;
const enterModuleScope = nn.enterModuleScope;
const deinitModel = nn.deinitModel;

/// 堆叠多层变换器块 (Transformer Block) 的解码器主干网络 (Transformer Decoder)
pub fn TransformerDecoder(comptime n_layer: usize) type {
    return struct {
        const Self = @This();

        h: [n_layer]TransformerBlock, // 堆叠的变换器块 (Transformer Block) 数组
        ln_f: RMSNorm, // 骨架最末端用于规范化的均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 层

        name: ?[]const u8 = null,
        module_type: []const u8 = "TransformerDecoder",

        /// 初始化整个解码器组件
        pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize) !Self {
            var h: [n_layer]TransformerBlock = undefined;
            var i: usize = 0;
            errdefer {
                for (0..i) |j| {
                    deinitModel(&h[j], allocator);
                }
            }
            // 循环初始化每一层变换器块 (TransformerBlock)
            while (i < n_layer) : (i += 1) {
                h[i] = try TransformerBlock.init(allocator, n_embd, n_head);
            }

            // 初始化最后的均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 层
            const ln_f = try RMSNorm.init(allocator, n_embd, 1e-5);

            return Self{
                .h = h,
                .ln_f = ln_f,
            };
        }

        /// 模块标准数学变换公式
        pub const formula = "x_L = \\text{DecoderStack}(x_0) = (\\text{Block}_L \\circ \\dots \\circ \\text{Block}_1)(x_0)";

        /// 解码器主干网络的前向传播流程
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();
            var current_x = x;
            // 依次贯穿每一层变换器块 (Transformer Block)
            for (self.h) |layer| {
                current_x = try layer.forward(graph, current_x);
            }

            // 执行最后一层均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 映射输出
            return try self.ln_f.forward(graph, current_x);
        }
    };
}

/// 生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型配置结构体
pub const GPTConfig = struct {
    vocab_size: usize = 50257, // 词表大小 (Vocabulary Size, vocab_size)，决定输入和输出层的映射维度
    block_size: usize = 1024, // 最大上下文长度/时间步长度 (Context Length / Block Size)
    n_embd: usize = 768, // 隐藏特征嵌入维度 (Embedding Dimension, n_embd)
    n_head: usize = 12, // 多头注意力头数 (Multi-Head Attention Heads, n_head)
    n_layer: usize = 12, // 变换器块 (Transformer Block) 堆叠的层数 (Number of Decoder Layers, n_layer)

    pub const default: GPTConfig = .{};
    pub fn defaultConfig() GPTConfig {
        return .{};
    }
};

/// 泛型生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型定义函数
pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        token_embedding: Embedding, // 词元嵌入层 (Token Embedding, wte)
        position_embedding: Embedding, // 位置嵌入层 (Position Embedding, wpe)
        decoder: TransformerDecoder(config.n_layer), // 堆叠的变换器解码器 (Transformer Decoder) 层与最终归一化层
        lm_head: Linear, // 最终输出概率的语言模型线性分类投影头 (Language Model Head, lm_head)
        name: ?[]const u8 = null,
        module_type: []const u8 = "GPT",

        const Self = @This();

        /// 初始化默认配置的生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型实例
        pub fn initDefault(allocator: std.mem.Allocator) !Self {
            return init(allocator);
        }

        /// 初始化生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型中的所有网络层权重
        pub fn init(allocator: std.mem.Allocator) !Self {
            // 初始化词元 (Token) 嵌入矩阵 [vocab_size, n_embd]
            const token_embedding = try Embedding.init(allocator, config.vocab_size, config.n_embd);
            errdefer deinitModel(&token_embedding, allocator);

            // 初始化位置 (Position) 嵌入矩阵 [block_size, n_embd]
            const position_embedding = try Embedding.init(allocator, config.block_size, config.n_embd);
            errdefer deinitModel(&position_embedding, allocator);

            // 初始化变换器解码器 (Transformer Decoder) 主干网络
            const decoder = try TransformerDecoder(config.n_layer).init(allocator, config.n_embd, config.n_head);
            errdefer deinitModel(&decoder, allocator);

            // 初始化输出映射语言模型分类头 (Language Model Head, lm_head) [n_embd, vocab_size]
            const lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size);

            return Self{
                .token_embedding = token_embedding,
                .position_embedding = position_embedding,
                .decoder = decoder,
                .lm_head = lm_head,
            };
        }

        /// 模块标准数学变换公式
        pub const formula = "\\text{logits} = \\text{GPT}(\\text{TokenIDs}; \\theta) \\rightarrow [B, T, V]";

        /// 前向推理传播流程
        /// 输入 x 为包含词元标识符 (Token Identifier, Token ID) 的二维张量 (2-Dimensional Tensor, 2D，支持 `*Tensor` 或 `*GenericTensor(IntT)`)，形状为 [B, T]
        /// 输出为未归一化的预测对数几率 (Logits)，形状为三维 (3-Dimensional, 3D): [B, T, vocab_size]
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            // 1. 获取词元 (Token) 嵌入向量: [B, T] -> [B, T, n_embd]
            const tok_emb = try self.token_embedding.forward(graph, x);

            // 2. 生成对应的时间/位置索引 [0, 1, 2, ... T-1]，并将其转换为二维 (2-Dimensional, 2D) 位置张量 (Tensor) [B, T]
            const pos_node = try graph.tensorND(&.{ B, T }, false);
            for (0..B) |b| {
                for (0..T) |t| {
                    pos_node.data[b * T + t] = @as(f32, @floatFromInt(t));
                }
            }
            // 位置索引由输入形状在模型内部生成，属于生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 自身的常量缓冲区，而非模型输入
            pos_node.is_buffer = true;
            if (self.name) |mod_name| {
                pos_node.setNameFormatted("{s}.pos_indices", .{mod_name});
            }

            // 3. 获取对应的可学习 (Learned) 位置嵌入向量: [B, T] -> [B, T, n_embd]
            const pos_emb = try self.position_embedding.forward(graph, pos_node);

            // 4. 将词元 (Token) 嵌入和位置嵌入进行求和融合，作为初始隐藏输入: h = tok_emb + pos_emb
            const h_x = try graph.add(tok_emb, pos_emb);
            if (self.name) |mod_name| {
                h_x.setNameFormatted("{s}.embeddings_sum", .{mod_name});
            }

            // 5. 将混合后的输入送进层叠的变换器解码器 (Transformer Decoder) 主干网络中依次计算
            // 输出形状保持为: [B, T, n_embd]
            const decoder_out = try self.decoder.forward(graph, h_x);

            // 6. 将输出展平为二维 (2-Dimensional, 2D)，以便进行最终分类头的全连接投影计算: [B, T, n_embd] -> [B*T, n_embd]
            const ln_x_2d = try graph.reshape(decoder_out, &.{ B * T, config.n_embd });

            // 7. 进行投影以获得词表空间未归一化的分类对数几率 (Logits): [B*T, n_embd] -> [B*T, vocab_size]
            const logits_2d = try self.lm_head.forward(graph, ln_x_2d);

            // 8. 将形状重塑还原成三维 (3-Dimensional, 3D) 形式返回: [B, T, vocab_size]
            return try graph.reshape(logits_2d, &.{ B, T, config.vocab_size });
        }
    };
}

/// 采用默认生成式预训练变换器配置 (GPTConfig) 的标准生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型类型别名
pub const DefaultGPT = GPT(GPTConfig.default);
