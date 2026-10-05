const std = @import("std");
const autodiff = @import("autodiff.zig");
const tensor = @import("tensor.zig");
const engine = @import("engine.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

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
pub const graph_ir = visualization.graph_ir;
pub const generateJson = visualization.generateJson;
pub const exportJson = visualization.exportJson;
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
pub const ConvOptions = tensor.ConvOptions;
pub const Conv1D = core.Conv1D;
pub const Conv2D = core.Conv2D;
pub const ConvTranspose1D = core.ConvTranspose1D;
pub const ConvTranspose2D = core.ConvTranspose2D;
pub const Module = core.Module;
pub const deinitModel = core.deinitModel;
pub const zeroGradModel = core.zeroGradModel;
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
pub const RMSNorm = normalization.RMSNorm;
pub const LayerNorm = normalization.LayerNorm;
pub const BatchNorm2d = normalization.BatchNorm2d;
pub const Dropout = normalization.Dropout;
pub const AvgPool2D = normalization.AvgPool2D;

// 4. 循环神经网络 (Recurrent Neural Networks, RNN)
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
pub const KVCache = transformer.KVCache;
pub const ScaledDotProductAttention = transformer.ScaledDotProductAttention;
pub const MLP = transformer.MLP;
pub const SwiGLU = transformer.SwiGLU;
pub const MoELayer = transformer.MoELayer;
pub const CausalSelfAttention = transformer.CausalSelfAttention;
pub const applyRope1D = transformer.applyRope1D;
pub const MLACache = transformer.MLACache;
pub const MLALayer = transformer.MLALayer;
pub const TransformerBlock = transformer.TransformerBlock;
pub const TransformerDecoder = transformer.TransformerDecoder;
pub const GPTConfig = transformer.GPTConfig;
pub const GPT = transformer.GPT;
pub const DefaultGPT = transformer.DefaultGPT;
pub const LoRALinear = transformer.LoRALinear;
pub const maskedCrossEntropyLoss = transformer.maskedCrossEntropyLoss;
pub const maskedCrossEntropyLossGraph = transformer.maskedCrossEntropyLossGraph;
pub const dpoLoss = transformer.dpoLoss;
pub const dpoLossGraph = transformer.dpoLossGraph;
pub const computeGroupAdvantages = transformer.computeGroupAdvantages;
pub const computeGRPOLoss = transformer.computeGRPOLoss;
pub const grpoLoss = transformer.grpoLoss;
pub const grpoLossGraph = transformer.grpoLossGraph;
pub const sampleTopP = transformer.sampleTopP;
pub const sampleTopK = transformer.sampleTopK;

// 6. 权重序列化 (Serialization - Safetensors)
pub const saveModel = serialization.saveModel;
pub const loadModel = serialization.loadModel;

// 7. 引擎执行与评估 (Training & Evaluation Engine)
pub const trainClassificationStep = engine.trainClassificationStep;
pub const evalClassificationStep = engine.evalClassificationStep;
pub const trainClassificationEpoch = engine.trainClassificationEpoch;
pub const evaluateClassification = engine.evaluateClassification;
pub const computeAccuracy = engine.computeAccuracy;
pub const ClassificationStepResult = engine.ClassificationStepResult;
pub const ClassificationEpochResult = engine.ClassificationEpochResult;
pub const trainStep = engine.trainStep;
pub const evalStep = engine.evalStep;
pub const trainEpoch = engine.trainEpoch;
pub const evaluate = engine.evaluate;
pub const StepResult = engine.StepResult;
pub const EpochResult = engine.EpochResult;

// ============================================================================

test {
    std.testing.refAllDecls(@This());
    _ = @import("nn/tests.zig");
}
