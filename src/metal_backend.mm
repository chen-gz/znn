#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "metal_backend.h"
#import <string.h>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> matmulPipeline = nil;

extern "C" int metal_init(void) {
    @autoreleasepool {
        if (device != nil) return 0; // Already initialized

        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal: System default device not found.");
            return -1;
        }

        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            NSLog(@"Metal: Command queue creation failed.");
            return -2;
        }

        // 2D Thread Matrix Multiplication Shader with Transpose and Accumulation Support
        NSString *shaderSource = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "struct MatrixDims {\n"
            "    int transA;\n"
            "    int transB;\n"
            "    int M;\n"
            "    int N;\n"
            "    int K;\n"
            "    float beta;\n"
            "};\n"
            "\n"
            "kernel void matmul_kernel(\n"
            "    device const float *A [[buffer(0)]],\n"
            "    device const float *B [[buffer(1)]],\n"
            "    device float *C [[buffer(2)]],\n"
            "    constant MatrixDims &dims [[buffer(3)]],\n"
            "    uint2 index [[thread_position_in_grid]]\n"
            ") {\n"
            "    int row = index.y;\n"
            "    int col = index.x;\n"
            "    if (row < dims.M && col < dims.N) {\n"
            "        float sum = 0.0f;\n"
            "        for (int k = 0; k < dims.K; k++) {\n"
            "            int idxA = (dims.transA != 0) ? (k * dims.M + row) : (row * dims.K + k);\n"
            "            int idxB = (dims.transB != 0) ? (col * dims.K + k) : (k * dims.N + col);\n"
            "            sum += A[idxA] * B[idxB];\n"
            "        }\n"
            "        if (dims.beta == 0.0f) {\n"
            "            C[row * dims.N + col] = sum;\n"
            "        } else {\n"
            "            C[row * dims.N + col] = sum + dims.beta * C[row * dims.N + col];\n"
            "        }\n"
            "    }\n"
            "}\n";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            NSLog(@"Metal: Failed to compile compute shader: %@", error);
            return -3;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"matmul_kernel"];
        if (!function) {
            NSLog(@"Metal: Failed to locate compute kernel function.");
            return -4;
        }

        matmulPipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!matmulPipeline) {
            NSLog(@"Metal: Failed to create compute pipeline state: %@", error);
            return -5;
        }

        NSLog(@"Metal GPU Backend initialized successfully.");
        return 0;
    }
}

struct MatrixDims {
    int transA;
    int transB;
    int M;
    int N;
    int K;
    float beta;
};

extern "C" void metal_matmul(
    int transA, int transB,
    int M, int N, int K,
    const float *A,
    const float *B,
    float *C,
    float beta
) {
    @autoreleasepool {
        if (device == nil) {
            if (metal_init() != 0) return;
        }

        id<MTLBuffer> bufferA = [device newBufferWithBytes:A length:M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B length:K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithBytes:C length:M * N * sizeof(float) options:MTLResourceStorageModeShared];

        MatrixDims dims = { transA, transB, M, N, K, beta };
        id<MTLBuffer> bufferDims = [device newBufferWithBytes:&dims length:sizeof(dims) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:matmulPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBuffer:bufferDims offset:0 atIndex:3];

        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], M * N * sizeof(float));
    }
}
