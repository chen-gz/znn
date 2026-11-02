//! Advanced indexing, scatter/gather operations and strided view indexing.
//!
//! Provides foundational infrastructure for Milestone 11 (PyTorch / NumPy tensor indexing alignment).
//! Functions to be implemented in Milestone 11:
//! - `gather`: multidimensional element gathering along specified axis
//! - `scatter`: in-place or destination-driven value scattering
//! - `scatterAdd`: reduction scattering with addition
//! - `indexSelect`: slicing/indexing by 1D index tensor
//! - `maskedSelect`: 1D dynamic gathering based on boolean condition masks

const std = @import("std");
const types = @import("types.zig");
const Tensor = types.Tensor;
const Shape = types.Shape;

// Structural placeholders for Milestone 11
pub const IndexingError = error{
    IndexOutOfBounds,
    InvalidIndexDimension,
    IndexShapeMismatch,
    ScatterDimensionMismatch,
};
