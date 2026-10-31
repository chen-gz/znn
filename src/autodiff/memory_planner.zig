//! 静态计算图内存规划器与缓冲区复用 (Static Memory Planning & Buffer Reuse)
const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const backend = @import("../backend.zig");
const types = @import("types.zig");
const Op = types.Op;

/// 描述张量在拓扑执行序列中的活跃时间区间
pub const LivenessInterval = struct {
    tensor_id: usize,
    tensor: *Tensor,
    start_op: usize, // 生成该张量的算子索引 (对于外部输入/参数为 0)
    end_op: usize, // 最后读取该张量的算子索引 (已包含反向引用扩展)
    byte_size: usize, // 64 字节对齐后的物理内存需求
    is_persistent: bool, // 是否为模型参数/权重或终端图输出 (不可复用)
    is_saved_for_backward: bool, // 是否被反向求导算子引用
    cluster_id: ?usize = null, // 视图张量所属的存储别名等价簇编号

    pub fn overlaps(self: LivenessInterval, other: LivenessInterval) bool {
        if (self.is_persistent or other.is_persistent) return true;
        return !(self.end_op < other.start_op or other.end_op < self.start_op);
    }
};

/// 静态物理缓冲区内部的张量子切片分配信息
pub const BufferAssignment = struct {
    tensor: *Tensor,
    offset_bytes: usize, // 在 shared_arena 中的起始字节偏移量
    size_bytes: usize, // 实际物理尺寸 (已向上对齐至 64 字节)
    alignment: usize = 64, // 强制 64 字节对齐
};

/// 整个静态计算图的显存复用规划方案
pub const MemoryPlan = struct {
    allocator: std.mem.Allocator,
    total_arena_bytes: usize, // 共享物理缓冲区必须分配的峰值字节数 (max(offset + size))
    baseline_bytes: usize, // 若不复用，所有中间张量总字节数代数和
    assignments: []BufferAssignment,
    intervals: []LivenessInterval,

    pub fn deinit(self: *MemoryPlan) void {
        self.allocator.free(self.assignments);
        self.allocator.free(self.intervals);
    }

    pub fn savingsRatio(self: MemoryPlan) f32 {
        if (self.baseline_bytes == 0) return 0.0;
        if (self.total_arena_bytes >= self.baseline_bytes) return 0.0;
        const saved = self.baseline_bytes - self.total_arena_bytes;
        return @as(f32, @floatFromInt(saved)) / @as(f32, @floatFromInt(self.baseline_bytes));
    }
};

pub const MemoryPlanner = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MemoryPlanner {
        return .{ .allocator = allocator };
    }

    pub fn buildPlan(
        self: *MemoryPlanner,
        ops: []const *Op,
        tensors: []const *Tensor,
        graph_outputs: []const *Tensor,
    ) !MemoryPlan {
        const intervals = try self.allocator.alloc(LivenessInterval, tensors.len);
        errdefer self.allocator.free(intervals);

        var baseline_bytes: usize = 0;

        // 1. 分析每个张量的活跃生命周期
        for (tensors, 0..) |t, i| {
            var is_persistent = false;
            // 终端图输出不可被后续算子覆盖
            for (graph_outputs) |out| {
                if (out == t) {
                    is_persistent = true;
                    break;
                }
            }
            // 叶子节点与参数、静态缓冲区张量不可复用
            if (t.creator == null or t.is_buffer or (t.requires_grad and t.creator == null)) {
                is_persistent = true;
            }

            var start_op: usize = 0;
            var found_creator = false;
            for (ops, 0..) |op, op_idx| {
                for (op.outputs) |out| {
                    if (out == t) {
                        start_op = op_idx;
                        found_creator = true;
                        break;
                    }
                }
                if (found_creator) break;
            }
            if (!found_creator) {
                is_persistent = true;
            }

            var end_op: usize = start_op;
            for (ops, 0..) |op, op_idx| {
                for (op.inputs) |inp| {
                    if (inp == t) {
                        if (op_idx > end_op) end_op = op_idx;
                    }
                }
            }

            const raw_bytes = t.shape.numel() * @sizeOf(f32);
            const aligned_bytes = std.mem.alignForward(usize, raw_bytes, 64);
            if (!is_persistent) {
                baseline_bytes += aligned_bytes;
            }

            intervals[i] = .{
                .tensor_id = i,
                .tensor = t,
                .start_op = start_op,
                .end_op = end_op,
                .byte_size = aligned_bytes,
                .is_persistent = is_persistent,
                .is_saved_for_backward = false,
                .cluster_id = null,
            };
        }

        // 2. 收集参与内存复用的非持久中间张量
        var non_persistent_count: usize = 0;
        for (intervals) |inter| {
            if (!inter.is_persistent and inter.byte_size > 0) {
                non_persistent_count += 1;
            }
        }

        const assignments = try self.allocator.alloc(BufferAssignment, non_persistent_count);
        errdefer self.allocator.free(assignments);

        if (non_persistent_count == 0) {
            return MemoryPlan{
                .allocator = self.allocator,
                .total_arena_bytes = 0,
                .baseline_bytes = baseline_bytes,
                .assignments = assignments,
                .intervals = intervals,
            };
        }

        // 3. 采用 First-Fit Decreasing (FFD) 贪心区间着色算法分配偏移
        const Item = struct {
            interval_idx: usize,
            byte_size: usize,
        };
        const items = try self.allocator.alloc(Item, non_persistent_count);
        defer self.allocator.free(items);

        var idx: usize = 0;
        for (intervals, 0..) |inter, i| {
            if (!inter.is_persistent and inter.byte_size > 0) {
                items[idx] = .{ .interval_idx = i, .byte_size = inter.byte_size };
                idx += 1;
            }
        }

        // 按 byte_size 降序排序
        std.mem.sort(Item, items, {}, struct {
            fn lessThan(_: void, a: Item, b: Item) bool {
                return a.byte_size > b.byte_size;
            }
        }.lessThan);

        var total_arena_bytes: usize = 0;

        for (items, 0..) |item, cur_assign_idx| {
            const cur_inter = intervals[item.interval_idx];

            // 候选偏移列表：0 以及所有已分配活跃区间的末尾
            var assigned_offset: usize = 0;
            var offset_found = false;

            // 尝试递增搜索可用 64 字节对齐偏移
            var test_offset: usize = 0;
            while (!offset_found) {
                var collision = false;
                const test_start = test_offset;
                const test_end = test_start + cur_inter.byte_size;

                for (0..cur_assign_idx) |prev_idx| {
                    const prev_assign = assignments[prev_idx];
                    // 找到对应的 interval
                    var prev_inter: ?LivenessInterval = null;
                    for (intervals) |inv| {
                        if (inv.tensor == prev_assign.tensor) {
                            prev_inter = inv;
                            break;
                        }
                    }

                    if (prev_inter) |pi| {
                        if (cur_inter.overlaps(pi)) {
                            const prev_start = prev_assign.offset_bytes;
                            const prev_end = prev_start + prev_assign.size_bytes;
                            // 判断 [test_start, test_end) 与 [prev_start, prev_end) 是否冲突
                            if (!(test_end <= prev_start or test_start >= prev_end)) {
                                collision = true;
                                test_offset = std.mem.alignForward(usize, prev_end, 64);
                                break;
                            }
                        }
                    }
                }

                if (!collision) {
                    assigned_offset = test_start;
                    offset_found = true;
                }
            }

            assignments[cur_assign_idx] = .{
                .tensor = cur_inter.tensor,
                .offset_bytes = assigned_offset,
                .size_bytes = cur_inter.byte_size,
                .alignment = 64,
            };

            const end_byte = assigned_offset + cur_inter.byte_size;
            if (end_byte > total_arena_bytes) {
                total_arena_bytes = end_byte;
            }
        }

        total_arena_bytes = std.mem.alignForward(usize, total_arena_bytes, 64);

        return MemoryPlan{
            .allocator = self.allocator,
            .total_arena_bytes = total_arena_bytes,
            .baseline_bytes = baseline_bytes,
            .assignments = assignments,
            .intervals = intervals,
        };
    }
};
