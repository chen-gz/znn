const std = @import("std");
const autodiff = @import("autodiff.zig");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
pub const Device = tensor.Device;

// ============================================================================
// 领域驱动子模块 (Domain-driven Submodules)
// ============================================================================

pub const core = @import("nn/core.zig");
pub const activations = @import("nn/activations.zig");
pub const normalization = @import("nn/normalization.zig");
pub const recurrent = @import("nn/recurrent.zig");
pub const attention = @import("nn/attention.zig");
pub const llm = @import("nn/llm.zig");
pub const transformer = @import("nn/transformer.zig");
pub const serialization = @import("nn/serialization.zig");
pub const visualization = @import("nn/visualization.zig");
pub const graph_ir = @import("nn/graph_ir.zig").graph_ir;
pub const generateJson = graph_ir.generateJson;
pub const exportJson = graph_ir.exportJson;
pub const NodeKind = visualization.NodeKind;
pub const NodeStatus = visualization.NodeStatus;
pub const FlowNodeKind = visualization.FlowNodeKind;
pub const EdgeKind = visualization.EdgeKind;
pub const NodeData = visualization.NodeData;

// ============================================================================
// 门面层导出 (Facade Re-exports) - 确保 100% 向上兼容
// ============================================================================

// 1. 核心与容器 (Core & Containers)
pub const init_mod = core.init_mod;
pub const Nonlinearity = core.Nonlinearity;
pub const calculateGain = core.calculateGain;
pub const InitMethod = core.InitMethod;
pub const InitOptions = core.InitOptions;
pub const normalRandom = core.normalRandom;
pub const initWeights = core.initWeights;
pub const createPersistentTensor = core.createPersistentTensor;
pub const freePersistentTensor = core.freePersistentTensor;
pub const Linear = core.Linear;
pub const Q4Linear = core.Q4Linear;
pub const ConvOptions = tensor.ConvOptions;
pub const Conv1D = core.Conv1D;
pub const Conv2D = core.Conv2D;
pub const ConvTranspose1D = core.ConvTranspose1D;
pub const ConvTranspose2D = core.ConvTranspose2D;
pub const Module = core.Module;
pub const deinitModel = core.deinitModel;
pub const zeroGradModel = core.zeroGradModel;
pub const releaseModuleGrads = core.releaseModuleGrads;
pub const createLayerScalar = core.createLayerScalar;
pub const initModel = core.initModel;
pub const initModelWithSample = core.initModelWithSample;
pub const callForward = core.callForward;
pub const ForwardResult = core.ForwardResult;
pub const setTrainingModel = core.setTrainingModel;
pub const trainModel = core.trainModel;
pub const evalModel = core.evalModel;
pub const walk = core.walk;
pub const nameModules = core.nameModules;
pub const enterModuleScope = core.enterModuleScope;
pub const parameters = core.parameters;
pub const NamedParameter = core.NamedParameter;
pub const NamedParameterList = core.NamedParameterList;
pub const namedParameters = core.namedParameters;
pub const numParameters = core.numParameters;
pub const setRequiresGrad = core.setRequiresGrad;
pub const ParameterInitReport = core.ParameterInitReport;
pub const inspectParameterInit = core.inspectParameterInit;
pub const warnIfParametersUninitialized = core.warnIfParametersUninitialized;
pub const Sequential = core.Sequential;
pub const sequential = core.sequential;

// 2. 激活函数 (Activations)
pub const ReLU = activations.ReLU;
pub const GELU = activations.GELU;
pub const Sigmoid = activations.Sigmoid;
pub const Tanh = activations.Tanh;
pub const LeakyReLU = activations.LeakyReLU;
pub const SiLU = activations.SiLU;
pub const Swish = activations.Swish;

// 3. 归一化与池化 (Normalization & Pooling)
pub const PoolOptions = tensor.PoolOptions;
pub const RMSNorm = normalization.RMSNorm;
pub const LayerNorm = normalization.LayerNorm;
pub const BatchNorm1d = normalization.BatchNorm1d;
pub const BatchNorm2d = normalization.BatchNorm2d;
pub const GroupNorm = normalization.GroupNorm;
pub const Dropout = normalization.Dropout;
pub const MaxPool1D = normalization.MaxPool1D;
pub const MaxPool2D = normalization.MaxPool2D;
pub const AvgPool1D = normalization.AvgPool1D;
pub const AvgPool2D = normalization.AvgPool2D;
pub const AdaptiveAvgPool1D = normalization.AdaptiveAvgPool1D;
pub const AdaptiveAvgPool2D = normalization.AdaptiveAvgPool2D;

// 4. 循环神经网络 (Recurrent Neural Networks, RNN)
pub const RecurrentResult = recurrent.RecurrentResult;
pub const Recurrent = recurrent.Recurrent;
pub const RNNCell = recurrent.RNNCell;
pub const RNN = recurrent.RNN;
pub const RNNResult = recurrent.RNNResult;
pub const LSTMState = recurrent.LSTMState;
pub const LSTMCell = recurrent.LSTMCell;
pub const LSTM = recurrent.LSTM;
pub const LSTMResult = recurrent.LSTMResult;
pub const StackedLSTM = recurrent.StackedLSTM;
pub const StackedLSTMResult = recurrent.StackedLSTMResult;
pub const GRUCell = recurrent.GRUCell;
pub const GRU = recurrent.GRU;
pub const GRUResult = recurrent.GRUResult;

// 5. Transformer、注意力与生成对齐 (Transformer, Attention & Alignment)
pub const Embedding = transformer.Embedding;
pub const LayerKVCache = transformer.LayerKVCache;
pub const KVCache = transformer.KVCache;
pub const ScaledDotProductAttention = transformer.ScaledDotProductAttention;
pub const splitHeads = transformer.splitHeads;
pub const mergeHeads = transformer.mergeHeads;
pub const GQAAttention = transformer.GQAAttention;
pub const MLP = transformer.MLP;
pub const GatedActivation = transformer.GatedActivation;
pub const GatedMLP = transformer.GatedMLP;
pub const SwiGLU = transformer.SwiGLU;
pub const MoELayer = transformer.MoELayer;
pub const CausalSelfAttention = transformer.CausalSelfAttention;
pub const RopeOptions = tensor.RopeOptions;
pub const MLACache = transformer.MLACache;
pub const MLALayer = transformer.MLALayer;
pub const TransformerBlock = transformer.TransformerBlock;
pub const LoRALinear = transformer.LoRALinear;
pub const maskedCrossEntropyLoss = transformer.maskedCrossEntropyLoss;
pub const maskedCrossEntropyLossGraph = transformer.maskedCrossEntropyLossGraph;
pub const dpoLoss = transformer.dpoLoss;
pub const dpoLossGraph = transformer.dpoLossGraph;
pub const computeGroupAdvantages = transformer.computeGroupAdvantages;
pub const computeGRPOLoss = transformer.computeGRPOLoss;
pub const grpoLoss = transformer.grpoLoss;
pub const grpoLossGraph = transformer.grpoLossGraph;
pub const promptLookup = llm.promptLookup;
pub const SpeculativeDraftEngine = llm.SpeculativeDraftEngine;

// 硬件加速 Transformer 组合块 (MLX Transformer Blocks)
pub const transformer_blocks = @import("nn/transformer_blocks.zig");
pub const Rope = transformer_blocks.Rope;
pub const projectHeads = transformer_blocks.projectHeads;
pub const gatedMlp = transformer_blocks.gatedMlp;
pub const residualScale = transformer_blocks.residualScale;
pub const finalHidden = transformer_blocks.finalHidden;
pub const TokenExclusionMask = transformer_blocks.TokenExclusionMask;
pub const softcapLogits = transformer_blocks.softcapLogits;
pub const penalizeDevice = transformer_blocks.penalizeDevice;
pub const DeviceCandidates = transformer_blocks.DeviceCandidates;
pub const HostCandidates = transformer_blocks.HostCandidates;
pub const deviceCandidates = transformer_blocks.deviceCandidates;
pub const selectToken = transformer_blocks.selectToken;
pub const selectTokenFromHidden = transformer_blocks.selectTokenFromHidden;

// 6. 权重序列化 (Serialization - Safetensors)
pub const saveModel = serialization.saveModel;
pub const loadModel = serialization.loadModel;
pub const TensorMeta = serialization.TensorMeta;
pub const SafetensorsArchive = serialization.SafetensorsArchive;

// ============================================================================

test {
    std.testing.refAllDecls(@This());
    _ = @import("nn/tests.zig");
}
