const std = @import("std");

pub const shape = @import("tensor/shape.zig");
pub const types = @import("tensor/types.zig");
pub const core = @import("tensor/core.zig");
pub const ops = @import("tensor/ops.zig");

// --- Re-export shape and stride utilities ---
pub const Shape = shape.Shape;
pub const computeContiguousStrides = shape.computeContiguousStrides;
pub const transposeShape = shape.transposeShape;
pub const broadcastShapes = shape.broadcastShapes;
pub const computeBroadcastStrides = shape.computeBroadcastStrides;
pub const broadcastBinaryOpRaw = shape.broadcastBinaryOpRaw;

// --- Re-export data types and generic tensors ---
pub const DType = types.DType;
pub const bf16 = types.bf16;
pub const SliceRange = types.SliceRange;
pub const convertScalar = types.convertScalar;
pub const GenericTensor = types.GenericTensor;
pub const FloatTensor = types.FloatTensor;
pub const DoubleTensor = types.DoubleTensor;
pub const IntTensor = types.IntTensor;
pub const LongTensor = types.LongTensor;
pub const BoolTensor = types.BoolTensor;
pub const BFloat16Tensor = types.BFloat16Tensor;
pub const UsizeTensor = types.UsizeTensor;
pub const TensorOf = types.TensorOf;

// --- Re-export core Tensor ---
pub const Tensor = core.Tensor;

// --- Re-export operations & factory functions ---
pub const array = ops.array;
pub const zeros = ops.zeros;
pub const ones = ops.ones;
pub const full = ops.full;
pub const arange = ops.arange;
pub const linspace = ops.linspace;
pub const eye = ops.eye;
pub const identity = ops.identity;
pub const manualSeed = ops.manualSeed;
pub const rand = ops.rand;
pub const free = ops.free;
pub const concat = ops.concat;
pub const stack = ops.stack;
pub const split = ops.split;
pub const repeat = ops.repeat;
pub const tile = ops.tile;
pub const where = ops.where;
pub const sum = ops.sum;
pub const mean = ops.mean;
pub const variance = ops.variance;
pub const stdDev = ops.stdDev;
pub const squeeze = ops.squeeze;
pub const unsqueeze = ops.unsqueeze;
pub const slice = ops.slice;
pub const contiguous = ops.contiguous;
pub const clip = ops.clip;
pub const sort = ops.sort;
pub const argsort = ops.argsort;
pub const nonzero = ops.nonzero;
pub const sqrt = ops.sqrt;
pub const exp = ops.exp;
pub const log = ops.log;
pub const abs = ops.abs;
pub const solveLinearSystem = ops.solveLinearSystem;
pub const solveRidgeAnalytical = ops.solveRidgeAnalytical;
pub const svd = ops.svd;
pub const qr = ops.qr;
pub const eig = ops.eig;
pub const applyRoPE = ops.applyRoPE;
pub const applyRoPETensor = ops.applyRoPETensor;

test {
    _ = @import("tensor/tests.zig");
}
