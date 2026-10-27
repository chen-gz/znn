//! 统一计算设备架构 (Compute Unified Device Architecture, CUDA) 硬件加速后端
//!
//! 基于 NVIDIA 显卡驱动的底层驱动应用程序接口 (Driver Application Programming Interface, Driver API)
//! 与并行线程执行 (Parallel Thread Execution, PTX) 即时编译器 (Just-In-Time Compiler, JIT)，
//! 将通用矩阵乘法 (General Matrix Multiply, GEMM)、均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)、
//! 旋转位置编码 (Rotary Position Embedding, RoPE) 与注意力计算卸载至图形处理器 (Graphics Processing Unit, GPU)。
//! 零外部 CUDA 编译套件工具链依赖，系统只要安装 NVIDIA 显示驱动即可直接运行。

const std = @import("std");
const builtin = @import("builtin");
const activation = @import("activation.zig");
const GeluApproximation = activation.GeluApproximation;
const kernels = @import("cuda_kernels.zig");

pub const CUdevice = c_int;
pub const CUcontext = *anyopaque;
pub const CUmodule = *anyopaque;
pub const CUfunction = *anyopaque;
pub const CUstream = *anyopaque;
pub const CUdeviceptr = u64;

pub const CuResult = enum(c_int) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    deinitialized = 4,
    no_device = 100,
    invalid_device = 101,
    invalid_context = 201,
    invalid_handle = 400,
    not_found = 500,
    not_ready = 600,
    no_binary_for_gpu = 218,
    _,
};

pub const CudaError = error{
    CudaNotSupported,
    CudaInitializationFailed,
    CudaOperationFailed,
    OutOfMemory,
    InvalidShape,
    ShapeMismatch,
    KernelNotFound,
    DeviceNotFound,
    OperationNotSupported,
    EmptyTensor,
    InvalidArray,
    ReferenceCountOverflow,
};

/// 驱动 API 函数指针表
const DriverApi = struct {
    cuInit: *const fn (flags: c_uint) callconv(.c) CuResult,
    cuDeviceGet: *const fn (device: *CUdevice, ordinal: c_int) callconv(.c) CuResult,
    cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) CuResult,
    cuDeviceGetName: *const fn (name: [*]u8, len: c_int, dev: CUdevice) callconv(.c) CuResult,
    cuDeviceTotalMem: *const fn (bytes: *usize, dev: CUdevice) callconv(.c) CuResult,
    cuDevicePrimaryCtxRetain: *const fn (pctx: *?CUcontext, dev: CUdevice) callconv(.c) CuResult,
    cuDevicePrimaryCtxRelease: *const fn (dev: CUdevice) callconv(.c) CuResult,
    cuCtxCreate: *const fn (pctx: *?CUcontext, flags: c_uint, dev: CUdevice) callconv(.c) CuResult,
    cuCtxDestroy: *const fn (ctx: ?CUcontext) callconv(.c) CuResult,
    cuCtxSetCurrent: *const fn (ctx: ?CUcontext) callconv(.c) CuResult,
    cuCtxSynchronize: *const fn () callconv(.c) CuResult,
    cuMemAlloc: *const fn (dptr: *CUdeviceptr, bytesize: usize) callconv(.c) CuResult,
    cuMemFree: *const fn (dptr: CUdeviceptr) callconv(.c) CuResult,
    cuMemcpyHtoD: *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, ByteCount: usize) callconv(.c) CuResult,
    cuMemcpyDtoH: *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, ByteCount: usize) callconv(.c) CuResult,
    cuMemcpyDtoD: *const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) callconv(.c) CuResult,
    cuMemsetD8: *const fn (dstDevice: CUdeviceptr, uc: u8, N: usize) callconv(.c) CuResult,
    cuMemsetD32: *const fn (dstDevice: CUdeviceptr, ui: u32, N: usize) callconv(.c) CuResult,
    cuStreamCreate: *const fn (phStream: *?CUstream, Flags: c_uint) callconv(.c) CuResult,
    cuStreamDestroy: *const fn (hStream: ?CUstream) callconv(.c) CuResult,
    cuStreamSynchronize: *const fn (hStream: ?CUstream) callconv(.c) CuResult,
    cuModuleLoadData: *const fn (module: *?CUmodule, image: ?*const anyopaque) callconv(.c) CuResult,
    cuModuleUnload: *const fn (hmod: ?CUmodule) callconv(.c) CuResult,
    cuModuleGetFunction: *const fn (hfunc: *?CUfunction, hmod: ?CUmodule, name: [*:0]const u8) callconv(.c) CuResult,
    cuLaunchKernel: *const fn (
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: ?CUstream,
        kernelParams: ?[*]const ?*anyopaque,
        extra: ?[*]const ?*anyopaque,
    ) callconv(.c) CuResult,
};

/// 全局驱动状态单例
var global_driver: ?DriverApi = null;
var global_lib: ?std.DynLib = null;
var global_ctx: ?CUcontext = null;
var global_module: ?CUmodule = null;
var global_device: CUdevice = 0;
var probe_attempted: bool = false;
var is_cuda_available: bool = false;

// 缓存内核函数指针
var fn_sgemm: ?CUfunction = null;
var fn_rms_norm: ?CUfunction = null;
var fn_add: ?CUfunction = null;
var fn_multiply: ?CUfunction = null;
var fn_mul_scalar: ?CUfunction = null;
var fn_gelu: ?CUfunction = null;
var fn_softmax: ?CUfunction = null;
var fn_rope: ?CUfunction = null;
var fn_kv_write_step: ?CUfunction = null;
var fn_cached_attention: ?CUfunction = null;
var fn_embedding: ?CUfunction = null;
var fn_softcap: ?CUfunction = null;

const lib_search_paths = [_][]const u8{
    "libcuda.so.1",
    "libcuda.so",
    "/run/opengl-driver/lib/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so.1",
    "nvcuda.dll",
    "libcuda.dylib",
};

fn initDriver() bool {
    if (probe_attempted) return is_cuda_available;
    probe_attempted = true;

    for (lib_search_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            global_lib = lib;
            break;
        } else |_| {}
    }

    const lib = &(global_lib orelse return false);

    const cuInit = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuInit), "cuInit") orelse return false;
    const cuDeviceGet = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDeviceGet), "cuDeviceGet") orelse return false;
    const cuDeviceGetCount = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDeviceGetCount), "cuDeviceGetCount") orelse return false;
    const cuDeviceGetName = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDeviceGetName), "cuDeviceGetName") orelse return false;
    const cuDeviceTotalMem = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDeviceTotalMem), "cuDeviceTotalMem_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDeviceTotalMem), "cuDeviceTotalMem") orelse return false);
    const cuDevicePrimaryCtxRetain = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDevicePrimaryCtxRetain), "cuDevicePrimaryCtxRetain") orelse return false;
    const cuDevicePrimaryCtxRelease = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuDevicePrimaryCtxRelease), "cuDevicePrimaryCtxRelease") orelse return false;
    const cuCtxCreate = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxCreate), "cuCtxCreate_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxCreate), "cuCtxCreate") orelse return false);
    const cuCtxDestroy = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxDestroy), "cuCtxDestroy_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxDestroy), "cuCtxDestroy") orelse return false);
    const cuCtxSetCurrent = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxSetCurrent), "cuCtxSetCurrent") orelse return false;
    const cuCtxSynchronize = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuCtxSynchronize), "cuCtxSynchronize") orelse return false;
    const cuMemAlloc = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemAlloc), "cuMemAlloc_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemAlloc), "cuMemAlloc") orelse return false);
    const cuMemFree = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemFree), "cuMemFree_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemFree), "cuMemFree") orelse return false);
    const cuMemcpyHtoD = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyHtoD), "cuMemcpyHtoD_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyHtoD), "cuMemcpyHtoD") orelse return false);
    const cuMemcpyDtoH = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyDtoH), "cuMemcpyDtoH_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyDtoH), "cuMemcpyDtoH") orelse return false);
    const cuMemcpyDtoD = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyDtoD), "cuMemcpyDtoD_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemcpyDtoD), "cuMemcpyDtoD") orelse return false);
    const cuMemsetD8 = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemsetD8), "cuMemsetD8_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemsetD8), "cuMemsetD8") orelse return false);
    const cuMemsetD32 = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemsetD32), "cuMemsetD32_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuMemsetD32), "cuMemsetD32") orelse return false);
    const cuStreamCreate = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuStreamCreate), "cuStreamCreate") orelse return false;
    const cuStreamDestroy = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuStreamDestroy), "cuStreamDestroy_v2") orelse
        (lib.lookup(@TypeOf(@as(DriverApi, undefined).cuStreamDestroy), "cuStreamDestroy") orelse return false);
    const cuStreamSynchronize = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuStreamSynchronize), "cuStreamSynchronize") orelse return false;
    const cuModuleLoadData = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuModuleLoadData), "cuModuleLoadData") orelse return false;
    const cuModuleUnload = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuModuleUnload), "cuModuleUnload") orelse return false;
    const cuModuleGetFunction = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuModuleGetFunction), "cuModuleGetFunction") orelse return false;
    const cuLaunchKernel = lib.lookup(@TypeOf(@as(DriverApi, undefined).cuLaunchKernel), "cuLaunchKernel") orelse return false;

    if (cuInit(0) != .success) return false;

    var dev_count: c_int = 0;
    if (cuDeviceGetCount(&dev_count) != .success or dev_count <= 0) return false;

    var dev: CUdevice = 0;
    if (cuDeviceGet(&dev, 0) != .success) return false;
    global_device = dev;

    global_driver = .{
        .cuInit = cuInit,
        .cuDeviceGet = cuDeviceGet,
        .cuDeviceGetCount = cuDeviceGetCount,
        .cuDeviceGetName = cuDeviceGetName,
        .cuDeviceTotalMem = cuDeviceTotalMem,
        .cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain,
        .cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease,
        .cuCtxCreate = cuCtxCreate,
        .cuCtxDestroy = cuCtxDestroy,
        .cuCtxSetCurrent = cuCtxSetCurrent,
        .cuCtxSynchronize = cuCtxSynchronize,
        .cuMemAlloc = cuMemAlloc,
        .cuMemFree = cuMemFree,
        .cuMemcpyHtoD = cuMemcpyHtoD,
        .cuMemcpyDtoH = cuMemcpyDtoH,
        .cuMemcpyDtoD = cuMemcpyDtoD,
        .cuMemsetD8 = cuMemsetD8,
        .cuMemsetD32 = cuMemsetD32,
        .cuStreamCreate = cuStreamCreate,
        .cuStreamDestroy = cuStreamDestroy,
        .cuStreamSynchronize = cuStreamSynchronize,
        .cuModuleLoadData = cuModuleLoadData,
        .cuModuleUnload = cuModuleUnload,
        .cuModuleGetFunction = cuModuleGetFunction,
        .cuLaunchKernel = cuLaunchKernel,
    };

    is_cuda_available = true;
    return true;
}

/// 检查当前系统是否支持统一计算设备架构 (Compute Unified Device Architecture, CUDA)
pub fn isAvailable() bool {
    return initDriver();
}

/// 获取全局驱动函数表
fn getApi() CudaError!*const DriverApi {
    if (!isAvailable()) return error.CudaNotSupported;
    return &(global_driver.?);
}

/// 确保统一计算设备架构上下文与算子内核模块已初始化
pub fn ensureContext() CudaError!void {
    const api = try getApi();
    if (global_ctx != null and global_module != null) return;

    if (global_ctx == null) {
        var ctx: ?CUcontext = null;
        if (api.cuDevicePrimaryCtxRetain(&ctx, global_device) != .success) {
            if (api.cuCtxCreate(&ctx, 0, global_device) != .success) {
                return error.CudaInitializationFailed;
            }
        }
        _ = api.cuCtxSetCurrent(ctx);
        global_ctx = ctx;
    }

    if (global_module == null) {
        var mod: ?CUmodule = null;
        if (api.cuModuleLoadData(&mod, kernels.KERNELS_PTX.ptr) != .success) {
            return error.CudaInitializationFailed;
        }
        global_module = mod;

        _ = api.cuModuleGetFunction(&fn_sgemm, mod, "sgemm_kernel");
        _ = api.cuModuleGetFunction(&fn_rms_norm, mod, "rms_norm_kernel");
        _ = api.cuModuleGetFunction(&fn_add, mod, "add_kernel");
        _ = api.cuModuleGetFunction(&fn_multiply, mod, "multiply_kernel");
        _ = api.cuModuleGetFunction(&fn_mul_scalar, mod, "mul_scalar_kernel");
        _ = api.cuModuleGetFunction(&fn_gelu, mod, "gelu_kernel");
        _ = api.cuModuleGetFunction(&fn_softmax, mod, "softmax_kernel");
        _ = api.cuModuleGetFunction(&fn_rope, mod, "rope_kernel");
        _ = api.cuModuleGetFunction(&fn_kv_write_step, mod, "kv_write_step_kernel");
        _ = api.cuModuleGetFunction(&fn_cached_attention, mod, "cached_attention_kernel");
        _ = api.cuModuleGetFunction(&fn_embedding, mod, "embedding_kernel");
        _ = api.cuModuleGetFunction(&fn_softcap, mod, "softcap_kernel");
    }
}

/// 查询系统可用图形处理器数量
pub fn getDeviceCount() u32 {
    const api = getApi() catch return 0;
    var count: c_int = 0;
    if (api.cuDeviceGetCount(&count) != .success or count < 0) return 0;
    return @intCast(count);
}

pub fn deviceCount() u32 {
    return getDeviceCount();
}

/// 获取首选图形处理器的名称
pub fn getDeviceName(buf: []u8) []const u8 {
    const api = getApi() catch return "CPU";
    if (api.cuDeviceGetName(buf.ptr, @intCast(buf.len), global_device) != .success) {
        return "Unknown CUDA Device";
    }
    return std.mem.sliceTo(buf, 0);
}

/// 获取首选图形处理器的显存容量 (字节)
pub fn getDeviceMemory() struct { total: usize } {
    const api = getApi() catch return .{ .total = 0 };
    var total: usize = 0;
    _ = api.cuDeviceTotalMem(&total, global_device);
    return .{ .total = total };
}

/// 统一计算设备架构执行流 (Stream)
pub const Stream = struct {
    handle: ?CUstream = null,

    pub fn create() CudaError!Stream {
        const api = try getApi();
        try ensureContext();
        var s: ?CUstream = null;
        if (api.cuStreamCreate(&s, 0) != .success) return error.CudaOperationFailed;
        return .{ .handle = s };
    }

    pub fn destroy(self: *Stream) void {
        const api = getApi() catch return;
        if (self.handle) |s| {
            _ = api.cuStreamDestroy(s);
            self.handle = null;
        }
    }

    pub fn synchronize(self: Stream) CudaError!void {
        const api = try getApi();
        if (self.handle) |s| {
            if (api.cuStreamSynchronize(s) != .success) return error.CudaOperationFailed;
        } else {
            if (api.cuCtxSynchronize() != .success) return error.CudaOperationFailed;
        }
    }
};

/// 显存分配 (Device Memory Allocation)
pub fn alloc(bytes: usize) CudaError!CUdeviceptr {
    const api = try getApi();
    try ensureContext();
    var dptr: CUdeviceptr = 0;
    if (api.cuMemAlloc(&dptr, bytes) != .success) return error.OutOfMemory;
    return dptr;
}

/// 显存释放 (Device Memory Free)
pub fn free(dptr: CUdeviceptr) void {
    const api = getApi() catch return;
    if (dptr != 0) {
        _ = api.cuMemFree(dptr);
    }
}

/// 主机内存到显存拷贝 (Host to Device Copy)
pub fn copyHostToDevice(dst: CUdeviceptr, src: *const anyopaque, bytes: usize) CudaError!void {
    const api = try getApi();
    if (api.cuMemcpyHtoD(dst, src, bytes) != .success) return error.CudaOperationFailed;
}

/// 显存到主机内存拷贝 (Device to Host Copy)
pub fn copyDeviceToHost(dst: *anyopaque, src: CUdeviceptr, bytes: usize) CudaError!void {
    const api = try getApi();
    if (api.cuMemcpyDtoH(dst, src, bytes) != .success) return error.CudaOperationFailed;
}

/// 显存内部设备间拷贝 (Device to Device Copy)
pub fn copyDeviceToDevice(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) CudaError!void {
    const api = try getApi();
    if (api.cuMemcpyDtoD(dst, src, bytes) != .success) return error.CudaOperationFailed;
}

/// 显存清零 (Zero Out Device Memory)
pub fn setZero(dst: CUdeviceptr, bytes: usize) CudaError!void {
    const api = try getApi();
    if (bytes % 4 == 0) {
        if (api.cuMemsetD32(dst, 0, bytes / 4) != .success) return error.CudaOperationFailed;
    } else {
        if (api.cuMemsetD8(dst, 0, bytes) != .success) return error.CudaOperationFailed;
    }
}

/// 数据类型枚举
pub const DType = enum(u8) {
    float32,
    bfloat16,
};

const ArrayOwner = struct {
    ptr: CUdeviceptr,
    allocator: std.mem.Allocator,
    references: std.atomic.Value(usize) = std.atomic.Value(usize).init(1),

    // Ownership transfers only on success; the caller cleans up the device
    // allocation if allocating this control record fails.
    fn create(allocator: std.mem.Allocator, ptr: CUdeviceptr) CudaError!*ArrayOwner {
        const owner = allocator.create(ArrayOwner) catch return error.OutOfMemory;
        owner.* = .{ .ptr = ptr, .allocator = allocator };
        return owner;
    }
};

fn checkedElements(dims: []const usize) CudaError!usize {
    if (dims.len > 8) return error.InvalidShape;
    for (dims) |d| if (d == 0) return 0;
    var total: usize = 1;
    for (dims) |d| total = std.math.mul(usize, total, d) catch return error.InvalidShape;
    return total;
}

/// 统一计算设备架构设备端张量数组 (CUDA Device Array)
pub const Array = struct {
    // Plain descriptor copies are borrowed. retain()/reshape() acquire an owned
    // reference; each owned descriptor must be deinitialized once. Atomic counts
    // protect allocation lifetime, not concurrent mutation of a descriptor/data.
    ptr: CUdeviceptr,
    owner: ?*ArrayOwner,
    shape: [8]usize = @splat(0),
    ndim: usize = 0,
    len: usize = 0,
    dtype: DType = .float32,

    /// 创建全零初始化的显存数组
    pub fn zeros(dims: []const usize) CudaError!Array {
        const total_len = try checkedElements(dims);
        const bytes = std.math.mul(usize, total_len, @sizeOf(f32)) catch return error.InvalidShape;
        const dptr = try alloc(bytes);
        errdefer free(dptr);
        try setZero(dptr, bytes);

        var shape_arr: [8]usize = @splat(0);
        @memcpy(shape_arr[0..dims.len], dims);

        return .{
            .ptr = dptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, dptr),
            .shape = shape_arr,
            .ndim = dims.len,
            .len = total_len,
            .dtype = .float32,
        };
    }

    /// 从切片初始化设备数组
    pub fn fromSlice(slice: []const f32, dims: []const usize) CudaError!Array {
        const total_len = try checkedElements(dims);
        if (total_len != slice.len) return error.ShapeMismatch;

        const bytes = std.math.mul(usize, slice.len, @sizeOf(f32)) catch return error.InvalidShape;
        const dptr = try alloc(bytes);
        errdefer free(dptr);

        try copyHostToDevice(dptr, slice.ptr, bytes);

        var shape_arr: [8]usize = @splat(0);
        @memcpy(shape_arr[0..dims.len], dims);

        return .{
            .ptr = dptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, dptr),
            .shape = shape_arr,
            .ndim = dims.len,
            .len = slice.len,
            .dtype = .float32,
        };
    }

    /// 拷贝到主机切片
    pub fn toHostSlice(self: Array, dest: []f32) CudaError!void {
        if (self.len != dest.len) return error.ShapeMismatch;
        try copyDeviceToHost(dest.ptr, self.ptr, self.len * @sizeOf(f32));
    }

    pub fn copyToSlice(self: Array, dest: []f32) CudaError!void {
        return self.toHostSlice(dest);
    }

    /// 克隆设备数组
    pub fn clone(self: Array) CudaError!Array {
        const bytes = self.len * @sizeOf(f32);
        const new_ptr = try alloc(bytes);
        errdefer free(new_ptr);
        try copyDeviceToDevice(new_ptr, self.ptr, bytes);
        return .{
            .ptr = new_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, new_ptr),
            .shape = self.shape,
            .ndim = self.ndim,
            .len = self.len,
            .dtype = self.dtype,
        };
    }

    /// Acquire another owned reference without copying device data.
    pub fn retain(self: Array) CudaError!Array {
        const owner = self.owner orelse return error.InvalidArray;
        var refs = owner.references.load(.monotonic);
        while (true) {
            if (refs == std.math.maxInt(usize)) return error.ReferenceCountOverflow;
            if (owner.references.cmpxchgWeak(refs, refs + 1, .monotonic, .monotonic)) |actual| {
                refs = actual;
            } else break;
        }
        return self;
    }

    /// Reshape into an independently owned view of the same device allocation.
    pub fn reshape(self: Array, new_dims: []const usize) CudaError!Array {
        if (try checkedElements(new_dims) != self.len) return error.ShapeMismatch;
        var view = try self.retain();
        view.shape = @splat(0);
        @memcpy(view.shape[0..new_dims.len], new_dims);
        view.ndim = new_dims.len;
        return view;
    }

    /// Release the allocation only when its last owned descriptor is released.
    pub fn deinit(self: *Array) void {
        if (self.owner) |owner| {
            if (owner.references.fetchSub(1, .acq_rel) == 1) {
                const allocator = owner.allocator;
                if (owner.ptr != 0) free(owner.ptr);
                allocator.destroy(owner);
            }
        }
        self.owner = null;
        self.ptr = 0;
        self.len = 0;
        self.ndim = 0;
        self.shape = @splat(0);
    }

    /// 通用矩阵乘法 (General Matrix Multiply, GEMM / Batched GEMM, BMM): C = A * B
    pub fn matmul(a: Array, b: Array, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_sgemm orelse return error.KernelNotFound;

        if (a.ndim < 2 or b.ndim < 2) return error.InvalidShape;
        const M = a.shape[a.ndim - 2];
        const K1 = a.shape[a.ndim - 1];
        const K2 = b.shape[b.ndim - 2];
        const N = b.shape[b.ndim - 1];
        if (K1 != K2) return error.ShapeMismatch;

        const batch_a = a.len / (M * K1);
        const batch_b = b.len / (K2 * N);
        if (batch_b != 1 and batch_b != batch_a) return error.ShapeMismatch;

        const out_len = batch_a * M * N;
        const c_ptr = try alloc(out_len * @sizeOf(f32));
        errdefer free(c_ptr);

        var m_u32: u32 = @intCast(M);
        var n_u32: u32 = @intCast(N);
        var k_u32: u32 = @intCast(K1);
        var str_a: u32 = @intCast(M * K1);
        var str_b: u32 = if (batch_b == 1) 0 else @intCast(K2 * N);
        var str_c: u32 = @intCast(M * N);
        var d_a = a.ptr;
        var d_b = b.ptr;
        var d_c = c_ptr;

        var params = [_]?*anyopaque{
            &d_a,
            &d_b,
            &d_c,
            &m_u32,
            &n_u32,
            &k_u32,
            &str_a,
            &str_b,
            &str_c,
        };

        const block_dim: c_uint = 16;
        const grid_x: c_uint = @intCast((N + block_dim - 1) / block_dim);
        const grid_y: c_uint = @intCast((M + block_dim - 1) / block_dim);
        const grid_z: c_uint = @intCast(batch_a);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_x,
            grid_y,
            grid_z,
            block_dim,
            block_dim,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        var res_shape = a.shape;
        res_shape[a.ndim - 1] = N;

        return .{
            .ptr = c_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, c_ptr),
            .shape = res_shape,
            .ndim = a.ndim,
            .len = out_len,
            .dtype = .float32,
        };
    }

    /// 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    pub fn rmsNorm(x: Array, weight: ?Array, eps: f32, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_rms_norm orelse return error.KernelNotFound;

        const D = x.shape[x.ndim - 1];
        const rows = x.len / D;

        const out_ptr = try alloc(x.len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_x = x.ptr;
        var d_w: CUdeviceptr = if (weight) |w| w.ptr else 0;
        var d_out = out_ptr;
        var eps_val = eps;
        var rows_u32: u32 = @intCast(rows);
        var cols_u32: u32 = @intCast(D);

        var params = [_]?*anyopaque{
            &d_x,
            &d_w,
            &d_out,
            &eps_val,
            &rows_u32,
            &cols_u32,
        };

        const h_stream = if (stream) |s| s.handle else null;
        if (api.cuLaunchKernel(
            func,
            rows_u32,
            1,
            1,
            1,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = x.shape,
            .ndim = x.ndim,
            .len = x.len,
            .dtype = .float32,
        };
    }

    /// 逐元素加法 (Elementwise Add)
    pub fn add(a: Array, b: Array, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_add orelse return error.KernelNotFound;
        if (a.len != b.len) return error.ShapeMismatch;

        const c_ptr = try alloc(a.len * @sizeOf(f32));
        errdefer free(c_ptr);

        var d_a = a.ptr;
        var d_b = b.ptr;
        var d_c = c_ptr;
        var n_u32: u32 = @intCast(a.len);

        var params = [_]?*anyopaque{
            &d_a,
            &d_b,
            &d_c,
            &n_u32,
        };

        const block_dim: c_uint = 256;
        const grid_dim: c_uint = @intCast((a.len + block_dim - 1) / block_dim);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = c_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, c_ptr),
            .shape = a.shape,
            .ndim = a.ndim,
            .len = a.len,
            .dtype = .float32,
        };
    }

    /// 逐元素乘法 (Elementwise Multiply)
    pub fn multiply(a: Array, b: Array, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_multiply orelse return error.KernelNotFound;
        if (a.len != b.len) return error.ShapeMismatch;

        const c_ptr = try alloc(a.len * @sizeOf(f32));
        errdefer free(c_ptr);

        var d_a = a.ptr;
        var d_b = b.ptr;
        var d_c = c_ptr;
        var n_u32: u32 = @intCast(a.len);

        var params = [_]?*anyopaque{
            &d_a,
            &d_b,
            &d_c,
            &n_u32,
        };

        const block_dim: c_uint = 256;
        const grid_dim: c_uint = @intCast((a.len + block_dim - 1) / block_dim);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = c_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, c_ptr),
            .shape = a.shape,
            .ndim = a.ndim,
            .len = a.len,
            .dtype = .float32,
        };
    }

    /// 标量乘法 (Multiply Scalar)
    pub fn mulScalar(a: Array, scalar: f32, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_mul_scalar orelse return error.KernelNotFound;

        const c_ptr = try alloc(a.len * @sizeOf(f32));
        errdefer free(c_ptr);

        var d_a = a.ptr;
        var s_val = scalar;
        var d_c = c_ptr;
        var n_u32: u32 = @intCast(a.len);

        var params = [_]?*anyopaque{
            &d_a,
            &s_val,
            &d_c,
            &n_u32,
        };

        const block_dim: c_uint = 256;
        const grid_dim: c_uint = @intCast((a.len + block_dim - 1) / block_dim);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = c_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, c_ptr),
            .shape = a.shape,
            .ndim = a.ndim,
            .len = a.len,
            .dtype = .float32,
        };
    }

    /// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU)
    pub fn gelu(a: Array, approx: GeluApproximation, stream: ?Stream) CudaError!Array {
        _ = approx;
        try ensureContext();
        const api = try getApi();
        const func = fn_gelu orelse return error.KernelNotFound;

        const out_ptr = try alloc(a.len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_a = a.ptr;
        var d_out = out_ptr;
        var n_u32: u32 = @intCast(a.len);

        var params = [_]?*anyopaque{
            &d_a,
            &d_out,
            &n_u32,
        };

        const block_dim: c_uint = 256;
        const grid_dim: c_uint = @intCast((a.len + block_dim - 1) / block_dim);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = a.shape,
            .ndim = a.ndim,
            .len = a.len,
            .dtype = .float32,
        };
    }

    /// 归一化指数函数 (Softmax)
    pub fn softmax(a: Array, axis: c_int, stream: ?Stream) CudaError!Array {
        _ = axis;
        try ensureContext();
        const api = try getApi();
        const func = fn_softmax orelse return error.KernelNotFound;

        const cols = a.shape[a.ndim - 1];
        const rows = a.len / cols;

        const out_ptr = try alloc(a.len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_x = a.ptr;
        var d_out = out_ptr;
        var rows_u32: u32 = @intCast(rows);
        var cols_u32: u32 = @intCast(cols);

        var params = [_]?*anyopaque{
            &d_x,
            &d_out,
            &rows_u32,
            &cols_u32,
        };

        const h_stream = if (stream) |s| s.handle else null;
        if (api.cuLaunchKernel(
            func,
            rows_u32,
            1,
            1,
            1,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = a.shape,
            .ndim = a.ndim,
            .len = a.len,
            .dtype = .float32,
        };
    }

    /// 旋转位置编码 (Rotary Position Embedding, RoPE)
    pub fn fastRoPE(
        x: Array,
        head_dim: usize,
        traditional: bool,
        base: f32,
        scale: f32,
        start_pos: usize,
        stream: ?Stream,
    ) CudaError!Array {
        _ = scale;
        try ensureContext();
        const api = try getApi();
        const func = fn_rope orelse return error.KernelNotFound;

        const total_tokens = x.len / head_dim;
        const out_ptr = try alloc(x.len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_x = x.ptr;
        var d_out = out_ptr;
        var total_tokens_u32: u32 = @intCast(total_tokens);
        var head_dim_u32: u32 = @intCast(head_dim);
        var start_pos_u32: u32 = @intCast(start_pos);
        var base_val = base;
        var trad_val: u32 = if (traditional) 1 else 0;

        var params = [_]?*anyopaque{
            &d_x,
            &d_out,
            &total_tokens_u32,
            &head_dim_u32,
            &start_pos_u32,
            &base_val,
            &trad_val,
        };

        const h_stream = if (stream) |s| s.handle else null;
        if (api.cuLaunchKernel(
            func,
            total_tokens_u32,
            1,
            1,
            1,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = x.shape,
            .ndim = x.ndim,
            .len = x.len,
            .dtype = .float32,
        };
    }

    /// 将新词元的键与值原位写入键值缓存对应槽位 [pos, pos + seq_len)
    pub fn writeKvStep(
        dst_k: Array,
        dst_v: Array,
        src_k: Array,
        src_v: Array,
        capacity: usize,
        pos: usize,
        seq_len: usize,
        num_kv: usize,
        head_dim: usize,
        stream: ?Stream,
    ) CudaError!void {
        try ensureContext();
        const api = try getApi();
        const func = fn_kv_write_step orelse return error.KernelNotFound;

        var d_dst_k = dst_k.ptr;
        var d_dst_v = dst_v.ptr;
        var d_src_k = src_k.ptr;
        var d_src_v = src_v.ptr;
        var cap_u32: u32 = @intCast(capacity);
        var pos_u32: u32 = @intCast(pos);
        var seq_u32: u32 = @intCast(seq_len);
        var num_kv_u32: u32 = @intCast(num_kv);
        var head_dim_u32: u32 = @intCast(head_dim);

        var params = [_]?*anyopaque{
            &d_dst_k,
            &d_dst_v,
            &d_src_k,
            &d_src_v,
            &cap_u32,
            &pos_u32,
            &seq_u32,
            &num_kv_u32,
            &head_dim_u32,
        };

        const h_stream = if (stream) |s| s.handle else null;
        if (api.cuLaunchKernel(
            func,
            seq_u32,
            num_kv_u32,
            1,
            1,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }
    }

    /// 缓存感知注意力计算 (闪速注意力 FlashAttention 在线 Softmax 纯 PTX 实现)
    /// Q: [1, num_heads, seq_len, head_dim] (或 [num_heads, seq_len, head_dim])
    /// K: [1, num_kv_heads, capacity, head_dim]
    /// V: [1, num_kv_heads, capacity, head_dim]
    /// Out: [1, num_heads, seq_len, head_dim]
    pub fn cachedAttention(
        q: Array,
        k: Array,
        v: Array,
        scale: f32,
        is_sliding: bool,
        sliding_window: usize,
        pos: usize,
        seq_len: usize,
        stream: ?Stream,
    ) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_cached_attention orelse return error.KernelNotFound;

        if (q.ndim < 3 or k.ndim < 3 or v.ndim < 3) return error.InvalidShape;
        const head_dim = q.shape[q.ndim - 1];
        if (head_dim > 512) return error.OperationNotSupported;

        const num_heads = q.shape[q.ndim - 3];
        const num_kv_heads = k.shape[k.ndim - 3];
        const capacity = k.shape[k.ndim - 2];

        const out_len = num_heads * seq_len * head_dim;
        const out_ptr = try alloc(out_len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_q = q.ptr;
        var d_k = k.ptr;
        var d_v = v.ptr;
        var d_out = out_ptr;
        var scale_val = scale;
        var pos_u32: u32 = @intCast(pos);
        var seq_len_u32: u32 = @intCast(seq_len);
        var num_heads_u32: u32 = @intCast(num_heads);
        var num_kv_heads_u32: u32 = @intCast(num_kv_heads);
        var head_dim_u32: u32 = @intCast(head_dim);
        var capacity_u32: u32 = @intCast(capacity);
        var is_sliding_u32: u32 = if (is_sliding) 1 else 0;
        var sliding_window_u32: u32 = @intCast(sliding_window);

        var params = [_]?*anyopaque{
            &d_q,
            &d_k,
            &d_v,
            &d_out,
            &scale_val,
            &pos_u32,
            &seq_len_u32,
            &num_heads_u32,
            &num_kv_heads_u32,
            &head_dim_u32,
            &capacity_u32,
            &is_sliding_u32,
            &sliding_window_u32,
        };

        const h_stream = if (stream) |s| s.handle else null;
        if (api.cuLaunchKernel(
            func,
            seq_len_u32,
            num_heads_u32,
            1,
            1,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        var res_shape = q.shape;
        res_shape[q.ndim - 2] = seq_len;
        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = res_shape,
            .ndim = q.ndim,
            .len = out_len,
            .dtype = .float32,
        };
    }

    /// 词元嵌入查表 (Embedding Lookup): output[i, j] = weight[indices[i], j]
    pub fn embedding(w: Array, indices: Array, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_embedding orelse return error.KernelNotFound;

        if (w.ndim != 2) return error.InvalidShape;
        const V = w.shape[0];
        const D = w.shape[1];
        const N = indices.len;

        const out_len = N * D;
        const c_ptr = try alloc(out_len * @sizeOf(f32));
        errdefer free(c_ptr);

        var n_u32: u32 = @intCast(N);
        var d_u32: u32 = @intCast(D);
        var v_u32: u32 = @intCast(V);
        var d_idx = indices.ptr;
        var d_w = w.ptr;
        var d_out = c_ptr;

        var params = [_]?*anyopaque{
            &d_idx,
            &d_w,
            &d_out,
            &n_u32,
            &d_u32,
            &v_u32,
        };

        const block_x: c_uint = 16;
        const block_y: c_uint = 16;
        const grid_x: c_uint = @intCast((D + block_x - 1) / block_x);
        const grid_y: c_uint = @intCast((N + block_y - 1) / block_y);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_x,
            grid_y,
            1,
            block_x,
            block_y,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        var res_shape: [8]usize = undefined;
        for (0..indices.ndim) |i| res_shape[i] = indices.shape[i];
        res_shape[indices.ndim] = D;

        return .{
            .ptr = c_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, c_ptr),
            .shape = res_shape,
            .ndim = indices.ndim + 1,
            .len = out_len,
            .dtype = .float32,
        };
    }

    /// 软截断算子 (Soft Capping): y = cap * tanh(x / cap)
    pub fn softcap(x: Array, cap: f32, stream: ?Stream) CudaError!Array {
        try ensureContext();
        const api = try getApi();
        const func = fn_softcap orelse return error.KernelNotFound;

        const out_ptr = try alloc(x.len * @sizeOf(f32));
        errdefer free(out_ptr);

        var d_in = x.ptr;
        var d_out = out_ptr;
        var n_u32: u32 = @intCast(x.len);
        var cap_f32 = cap;

        var params = [_]?*anyopaque{
            &d_in,
            &d_out,
            &n_u32,
            &cap_f32,
        };

        const block_dim: c_uint = 256;
        const grid_dim: c_uint = @intCast((x.len + block_dim - 1) / block_dim);
        const h_stream = if (stream) |s| s.handle else null;

        if (api.cuLaunchKernel(
            func,
            grid_dim,
            1,
            1,
            block_dim,
            1,
            1,
            0,
            h_stream,
            &params,
            null,
        ) != .success) {
            return error.CudaOperationFailed;
        }

        return .{
            .ptr = out_ptr,
            .owner = try ArrayOwner.create(std.heap.smp_allocator, out_ptr),
            .shape = x.shape,
            .ndim = x.ndim,
            .len = x.len,
            .dtype = .float32,
        };
    }
};

// ============================================================================
// 底层强类型内核启动函数 (Low-Level Typed Kernel Launchers)
// ============================================================================

pub fn launchSgemm(
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    c_ptr: CUdeviceptr,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_sgemm orelse return error.KernelNotFound;
    var m_u32: u32 = @intCast(m);
    var n_u32: u32 = @intCast(n);
    var k_u32: u32 = @intCast(k);
    var str_a: u32 = @intCast(m * k);
    var str_b: u32 = @intCast(k * n);
    var str_c: u32 = @intCast(m * n);
    var d_a = a_ptr;
    var d_b = b_ptr;
    var d_c = c_ptr;
    var params = [_]?*anyopaque{
        &d_a, &d_b, &d_c, &m_u32, &n_u32, &k_u32, &str_a, &str_b, &str_c,
    };
    const block_dim: c_uint = 16;
    const grid_x: c_uint = @intCast((n + block_dim - 1) / block_dim);
    const grid_y: c_uint = @intCast((m + block_dim - 1) / block_dim);
    const grid_z: c_uint = @intCast(batch);
    if (api.cuLaunchKernel(func, grid_x, grid_y, grid_z, block_dim, block_dim, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchRmsNorm(
    x_ptr: CUdeviceptr,
    w_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    rows: usize,
    cols: usize,
    eps: f32,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_rms_norm orelse return error.KernelNotFound;
    var d_x = x_ptr;
    var d_w = w_ptr;
    var d_out = out_ptr;
    var eps_val = eps;
    var rows_u32: u32 = @intCast(rows);
    var cols_u32: u32 = @intCast(cols);
    var params = [_]?*anyopaque{
        &d_x, &d_w, &d_out, &eps_val, &rows_u32, &cols_u32,
    };
    if (api.cuLaunchKernel(func, rows_u32, 1, 1, 1, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchRope(
    x_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    total_tokens: usize,
    head_dim: usize,
    traditional: bool,
    base: f32,
    start_pos: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_rope orelse return error.KernelNotFound;
    var d_x = x_ptr;
    var d_out = out_ptr;
    var total_tokens_u32: u32 = @intCast(total_tokens);
    var head_dim_u32: u32 = @intCast(head_dim);
    var start_pos_u32: u32 = @intCast(start_pos);
    var base_val = base;
    var trad_val: u32 = if (traditional) 1 else 0;
    var params = [_]?*anyopaque{
        &d_x, &d_out, &total_tokens_u32, &head_dim_u32, &start_pos_u32, &base_val, &trad_val,
    };
    if (api.cuLaunchKernel(func, total_tokens_u32, 1, 1, 1, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchSoftmax(
    x_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_softmax orelse return error.KernelNotFound;
    var d_x = x_ptr;
    var d_out = out_ptr;
    var rows_u32: u32 = @intCast(rows);
    var cols_u32: u32 = @intCast(cols);
    var params = [_]?*anyopaque{
        &d_x, &d_out, &rows_u32, &cols_u32,
    };
    if (api.cuLaunchKernel(func, rows_u32, 1, 1, 1, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchAdd(
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    len: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_add orelse return error.KernelNotFound;
    var d_a = a_ptr;
    var d_b = b_ptr;
    var d_c = out_ptr;
    var n_u32: u32 = @intCast(len);
    var params = [_]?*anyopaque{
        &d_a, &d_b, &d_c, &n_u32,
    };
    const block_dim: c_uint = 256;
    const grid_dim: c_uint = @intCast((len + block_dim - 1) / block_dim);
    if (api.cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchMultiply(
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    len: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_multiply orelse return error.KernelNotFound;
    var d_a = a_ptr;
    var d_b = b_ptr;
    var d_c = out_ptr;
    var n_u32: u32 = @intCast(len);
    var params = [_]?*anyopaque{
        &d_a, &d_b, &d_c, &n_u32,
    };
    const block_dim: c_uint = 256;
    const grid_dim: c_uint = @intCast((len + block_dim - 1) / block_dim);
    if (api.cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchMulScalar(
    a_ptr: CUdeviceptr,
    scalar: f32,
    out_ptr: CUdeviceptr,
    len: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_mul_scalar orelse return error.KernelNotFound;
    var d_a = a_ptr;
    var s_val = scalar;
    var d_c = out_ptr;
    var n_u32: u32 = @intCast(len);
    var params = [_]?*anyopaque{
        &d_a, &s_val, &d_c, &n_u32,
    };
    const block_dim: c_uint = 256;
    const grid_dim: c_uint = @intCast((len + block_dim - 1) / block_dim);
    if (api.cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchGelu(
    x_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    len: usize,
    use_tanh: bool,
) CudaError!void {
    _ = use_tanh;
    try ensureContext();
    const api = try getApi();
    const func = fn_gelu orelse return error.KernelNotFound;
    var d_a = x_ptr;
    var d_out = out_ptr;
    var n_u32: u32 = @intCast(len);
    var params = [_]?*anyopaque{
        &d_a, &d_out, &n_u32,
    };
    const block_dim: c_uint = 256;
    const grid_dim: c_uint = @intCast((len + block_dim - 1) / block_dim);
    if (api.cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchSoftcap(
    x_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    len: usize,
    cap: f32,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_softcap orelse return error.KernelNotFound;
    var d_in = x_ptr;
    var d_out = out_ptr;
    var n_u32: u32 = @intCast(len);
    var cap_f32 = cap;
    var params = [_]?*anyopaque{
        &d_in, &d_out, &n_u32, &cap_f32,
    };
    const block_dim: c_uint = 256;
    const grid_dim: c_uint = @intCast((len + block_dim - 1) / block_dim);
    if (api.cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

pub fn launchEmbedding(
    weights_ptr: CUdeviceptr,
    indices_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    n: usize,
    d: usize,
    v: usize,
) CudaError!void {
    try ensureContext();
    const api = try getApi();
    const func = fn_embedding orelse return error.KernelNotFound;
    var n_u32: u32 = @intCast(n);
    var d_u32: u32 = @intCast(d);
    var v_u32: u32 = @intCast(v);
    var d_idx = indices_ptr;
    var d_w = weights_ptr;
    var d_out = out_ptr;
    var params = [_]?*anyopaque{
        &d_idx, &d_w, &d_out, &n_u32, &d_u32, &v_u32,
    };
    const block_x: c_uint = 16;
    const block_y: c_uint = 16;
    const grid_x: c_uint = @intCast((d + block_x - 1) / block_x);
    const grid_y: c_uint = @intCast((n + block_y - 1) / block_y);
    if (api.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, block_y, 1, 0, null, &params, null) != .success) {
        return error.CudaOperationFailed;
    }
}

/// 统一计算设备架构键值缓存 (Compute Unified Device Architecture Key-Value Cache, CudaKvCache)
pub const CudaKvCache = struct {
    k: Array,
    v: Array,
    capacity: usize,
    num_kv: usize,
    head_dim: usize,
    curr_len: usize = 0,

    pub fn init(capacity: usize, num_kv: usize, head_dim: usize, stream: ?Stream) !CudaKvCache {
        _ = stream;
        var k = try Array.zeros(&.{ 1, num_kv, capacity, head_dim });
        errdefer k.deinit();
        const v = try Array.zeros(&.{ 1, num_kv, capacity, head_dim });
        return .{
            .k = k,
            .v = v,
            .capacity = capacity,
            .num_kv = num_kv,
            .head_dim = head_dim,
            .curr_len = 0,
        };
    }

    pub fn initDefault(capacity: usize, num_kv: usize, head_dim: usize) !CudaKvCache {
        return init(capacity, num_kv, head_dim, null);
    }

    pub fn deinit(self: *CudaKvCache) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn reset(self: *CudaKvCache) void {
        self.curr_len = 0;
    }

    pub fn truncate(self: *CudaKvCache, new_len: usize) void {
        if (new_len < self.curr_len) {
            self.curr_len = new_len;
        }
    }

    pub fn ensureCapacity(self: *CudaKvCache, required: usize, stream: ?Stream) !void {
        if (required <= self.capacity) return;
        const add_cap: usize = ((required - self.capacity + 255) / 256) * 256;
        const new_cap = self.capacity + add_cap;

        var new_k = try Array.zeros(&.{ 1, self.num_kv, new_cap, self.head_dim });
        errdefer new_k.deinit();
        var new_v = try Array.zeros(&.{ 1, self.num_kv, new_cap, self.head_dim });
        errdefer new_v.deinit();

        if (self.curr_len > 0) {
            const api = try getApi();
            _ = stream;
            var kv: usize = 0;
            while (kv < self.num_kv) : (kv += 1) {
                const src_off = kv * self.capacity * self.head_dim * @sizeOf(f32);
                const dst_off = kv * new_cap * self.head_dim * @sizeOf(f32);
                const bytes = self.curr_len * self.head_dim * @sizeOf(f32);
                if (bytes > 0) {
                    if (api.cuMemcpyDtoD(new_k.ptr + dst_off, self.k.ptr + src_off, bytes) != .success) return error.CudaOperationFailed;
                    if (api.cuMemcpyDtoD(new_v.ptr + dst_off, self.v.ptr + src_off, bytes) != .success) return error.CudaOperationFailed;
                }
            }
        }

        self.k.deinit();
        self.v.deinit();
        self.k = new_k;
        self.v = new_v;
        self.capacity = new_cap;
    }

    pub fn writeStep(
        self: *CudaKvCache,
        k_new: Array,
        v_new: Array,
        is_sliding: bool,
        sliding_window: usize,
        pos: usize,
        seq_len: usize,
        stream: ?Stream,
    ) !void {
        _ = is_sliding;
        _ = sliding_window;
        const total_len = pos + seq_len;
        try self.ensureCapacity(total_len, stream);

        try Array.writeKvStep(
            self.k,
            self.v,
            k_new,
            v_new,
            self.capacity,
            pos,
            seq_len,
            self.num_kv,
            self.head_dim,
            stream,
        );
        self.curr_len = total_len;
    }
};

test "cuda: hardware probe and driver availability" {
    if (isAvailable()) {
        try std.testing.expect(getDeviceCount() > 0);
        var name_buf: [256]u8 = undefined;
        const name = getDeviceName(&name_buf);
        try std.testing.expect(name.len > 0);
        const mem = getDeviceMemory();
        try std.testing.expect(mem.total > 0);

        try ensureContext();

        // 测试内存分配与拷贝
        const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        var arr = try Array.fromSlice(&host_data, &.{ 2, 2 });
        defer arr.deinit();

        var read_back: [4]f32 = undefined;
        try arr.toHostSlice(&read_back);
        try std.testing.expectEqualSlices(f32, &host_data, &read_back);

        // 测试矩阵乘法 GEMM: [2, 2] * [2, 2]
        var res = try Array.matmul(arr, arr, null);
        defer res.deinit();
        var mat_res: [4]f32 = undefined;
        try res.toHostSlice(&mat_res);
        // [1 2; 3 4] * [1 2; 3 4] = [7 10; 15 22]
        try std.testing.expectEqual(@as(f32, 7.0), mat_res[0]);
        try std.testing.expectEqual(@as(f32, 10.0), mat_res[1]);
        try std.testing.expectEqual(@as(f32, 15.0), mat_res[2]);
        try std.testing.expectEqual(@as(f32, 22.0), mat_res[3]);

        // 测试均方根层归一化 RMSNorm
        var norm_res = try Array.rmsNorm(arr, null, 1e-5, null);
        defer norm_res.deinit();
        var norm_back: [4]f32 = undefined;
        try norm_res.toHostSlice(&norm_back);
        try std.testing.expect(norm_back[0] > 0.0);

        // 测试逐元素加法 Add
        var add_res = try Array.add(arr, arr, null);
        defer add_res.deinit();
        var add_back: [4]f32 = undefined;
        try add_res.toHostSlice(&add_back);
        try std.testing.expectEqual(@as(f32, 2.0), add_back[0]);
        try std.testing.expectEqual(@as(f32, 8.0), add_back[3]);

        // 测试标量乘法 MulScalar
        var scale_res = try Array.mulScalar(arr, 3.0, null);
        defer scale_res.deinit();
        var scale_back: [4]f32 = undefined;
        try scale_res.toHostSlice(&scale_back);
        try std.testing.expectEqual(@as(f32, 3.0), scale_back[0]);
        try std.testing.expectEqual(@as(f32, 12.0), scale_back[3]);

        // 测试高斯误差线性单元 GELU
        var gelu_res = try Array.gelu(arr, .tanh, null);
        defer gelu_res.deinit();
        var gelu_back: [4]f32 = undefined;
        try gelu_res.toHostSlice(&gelu_back);
        try std.testing.expect(gelu_back[0] > 0.0);

        // 测试归一化指数函数 Softmax
        var sm_res = try Array.softmax(arr, -1, null);
        defer sm_res.deinit();
        var sm_back: [4]f32 = undefined;
        try sm_res.toHostSlice(&sm_back);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sm_back[0] + sm_back[1], 0.001);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sm_back[2] + sm_back[3], 0.001);

        // 测试全零数组与 CudaKvCache 键值缓存
        var cache = try CudaKvCache.init(4, 1, 2, null);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 4), cache.capacity);
        try std.testing.expectEqual(@as(usize, 0), cache.curr_len);

        const k_data = [_]f32{ 1.0, 0.0 };
        const v_data = [_]f32{ 2.0, 3.0 };
        var k_arr = try Array.fromSlice(&k_data, &.{ 1, 1, 1, 2 });
        defer k_arr.deinit();
        var v_arr = try Array.fromSlice(&v_data, &.{ 1, 1, 1, 2 });
        defer v_arr.deinit();

        try cache.writeStep(k_arr, v_arr, false, 0, 0, 1, null);
        try std.testing.expectEqual(@as(usize, 1), cache.curr_len);

        // 测试缓存感知注意力计算 cachedAttention
        const q_data = [_]f32{ 1.0, 0.0 };
        var q_arr = try Array.fromSlice(&q_data, &.{ 1, 1, 1, 2 });
        defer q_arr.deinit();

        var attn_out = try Array.cachedAttention(q_arr, cache.k, cache.v, 1.0, false, 0, 0, 1, null);
        defer attn_out.deinit();
        var attn_back: [2]f32 = undefined;
        try attn_out.toHostSlice(&attn_back);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), attn_back[0], 0.01);
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), attn_back[1], 0.01);
    }
}

// Null device pointers isolate host ownership bookkeeping from GPU execution.
fn ownershipLifecycle(allocator: std.mem.Allocator) !void {
    var root = Array{ .ptr = 0, .owner = try ArrayOwner.create(allocator, 0), .len = 6, .ndim = 2, .shape = .{ 2, 3, 0, 0, 0, 0, 0, 0 } };
    defer root.deinit();
    var view = try root.reshape(&.{ 3, 2 });
    defer view.deinit();
    var nested = try view.reshape(&.{6});
    defer nested.deinit();
    var retained = try root.retain();
    defer retained.deinit();
    try std.testing.expectEqual(@as(usize, 4), root.owner.?.references.load(.monotonic));
    view.deinit();
    root.deinit();
    root.deinit();
    try std.testing.expectEqual(@as(usize, 2), nested.owner.?.references.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 6), nested.shape[0]);
    nested.deinit();
    try std.testing.expectEqual(@as(usize, 1), retained.owner.?.references.load(.monotonic));
    var last_view = try retained.reshape(&.{ 1, 6 });
    defer last_view.deinit();
    retained.deinit();
    try std.testing.expectEqual(@as(usize, 1), last_view.owner.?.references.load(.monotonic));
}

test "cuda: retained reshape ownership and control-record allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, ownershipLifecycle, .{});
}

test "cuda: invalid reshape and reference overflow leave ownership unchanged" {
    var root = Array{ .ptr = 0, .owner = try ArrayOwner.create(std.testing.allocator, 0), .len = 6 };
    defer root.deinit();
    try std.testing.expectError(error.ShapeMismatch, root.reshape(&.{7}));
    try std.testing.expectError(error.InvalidShape, root.reshape(&.{ std.math.maxInt(usize), 2 }));
    try std.testing.expectError(error.InvalidShape, root.reshape(&@as([9]usize, @splat(1))));
    try std.testing.expectEqual(@as(usize, 1), root.owner.?.references.load(.monotonic));
    root.owner.?.references.store(std.math.maxInt(usize), .monotonic);
    try std.testing.expectError(error.ReferenceCountOverflow, root.retain());
    try std.testing.expectError(error.ReferenceCountOverflow, root.reshape(&.{6}));
    root.owner.?.references.store(1, .monotonic);
    root.deinit();
    try std.testing.expectError(error.InvalidArray, root.retain());
    try std.testing.expectError(error.InvalidArray, root.reshape(&.{0}));
    try std.testing.expectEqual(@as(usize, 1), try checkedElements(&.{}));
    try std.testing.expectEqual(@as(usize, 0), try checkedElements(&.{ std.math.maxInt(usize), 2, 0 }));
    try std.testing.expectError(error.InvalidShape, Array.zeros(&.{ std.math.maxInt(usize), 2 }));
    try std.testing.expectError(error.InvalidShape, Array.fromSlice(&.{}, &.{ std.math.maxInt(usize), 2 }));
}

test "cuda: real reshape views survive parent release and clones are independent" {
    if (!isAvailable()) return error.SkipZigTest;
    var root = try Array.fromSlice(&.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer root.deinit();
    var view = try root.reshape(&.{4});
    defer view.deinit();
    var nested = try view.reshape(&.{ 1, 4 });
    defer nested.deinit();
    var cloned = try nested.clone();
    defer cloned.deinit();
    root.deinit();
    view.deinit();
    var output: [4]f32 = undefined;
    try nested.toHostSlice(&output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &output);
    try setZero(nested.ptr, 4 * @sizeOf(f32));
    try cloned.toHostSlice(&output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &output);
    nested.deinit();
    try cloned.toHostSlice(&output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &output);
}
