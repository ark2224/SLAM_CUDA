#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdint>


#define CUDA_CHECK(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)


// Extern declarations for Step 2 kernels
extern __global__ void fast_score_kernel(const uint8_t*, const uint8_t*, int,int,int,int,float,float*);
extern __global__ void nms3x3_kernel(const float*, int,int, float, uint2*, int, int*);


struct FrameBuffers {
    // Device
    uint8_t *d_img=nullptr, *d_msk=nullptr; float *d_scores=nullptr;
    uint2 *d_kp=nullptr;
    int *d_cnt=nullptr;
    // Host pinned
    uint8_t *h_img=nullptr, *h_msk=nullptr; int *h_cnt=nullptr;
    int W=0,H=0,max_kp=0; cudaStream_t stream=0; cudaEvent_t done;
};


extern "C" void step4_allocate(FrameBuffers& fb, int W, int H, int max_kp){
    fb.W=W;
    fb.H=H;
    fb.max_kp=max_kp;

    CUDA_CHECK(cudaHostAlloc(&fb.h_img, W*H, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&fb.h_msk, W*H, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&fb.h_cnt, sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&fb.d_img, W*H));
    CUDA_CHECK(cudaMalloc(&fb.d_msk, W*H));
    CUDA_CHECK(cudaMalloc(&fb.d_scores, W*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fb.d_kp, max_kp*sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(&fb.d_cnt, sizeof(int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&fb.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fb.done, cudaEventDisableTiming));
}


extern "C" void step4_process_async(FrameBuffers& fb){
    // H->D async copies
    CUDA_CHECK(cudaMemcpyAsync(fb.d_img, fb.h_img, fb.W*fb.H, cudaMemcpyHostToDevice, fb.stream));
    CUDA_CHECK(cudaMemcpyAsync(fb.d_msk, fb.h_msk, fb.W*fb.H, cudaMemcpyHostToDevice, fb.stream));
    CUDA_CHECK(cudaMemsetAsync(fb.d_cnt, 0, sizeof(int), fb.stream));

    dim3 BS(16,16), GS((fb.W+15)/16,(fb.H+15)/16);
    fast_score_kernel<<<GS,BS,0,fb.stream>>>(fb.d_img, fb.d_msk, fb.W, fb.H, fb.W, 20, 0.3f, fb.d_scores);
    size_t shmem=(16+2)*(16+2)*sizeof(float);
    nms3x3_kernel<<<GS,BS,shmem,fb.stream>>>(fb.d_scores, fb.W, fb.H, 5.0f, fb.d_kp, fb.max_kp, fb.d_cnt);

    // D->H async copy of count only (results buffer can remain device-resident for downstream)
    CUDA_CHECK(cudaMemcpyAsync(fb.h_cnt, fb.d_cnt, sizeof(int), cudaMemcpyDeviceToHost, fb.stream));
    CUDA_CHECK(cudaEventRecord(fb.done, fb.stream));
}


extern "C" bool step4_poll_complete(FrameBuffers& fb){ return cudaEventQuery(fb.done) == cudaSuccess; }
extern "C" int step4_get_count(const FrameBuffers& fb){ return *fb.h_cnt; }
extern "C" void step4_free(FrameBuffers& fb){
    if(fb.d_img) cudaFree(fb.d_img); if(fb.d_msk) cudaFree(fb.d_msk); if(fb.d_scores) cudaFree(fb.d_scores);
    if(fb.d_kp) cudaFree(fb.d_kp); if(fb.d_cnt) cudaFree(fb.d_cnt);
    if(fb.h_img) cudaFreeHost(fb.h_img); if(fb.h_msk) cudaFreeHost(fb.h_msk); if(fb.h_cnt) cudaFreeHost(fb.h_cnt);
    if(fb.stream) cudaStreamDestroy(fb.stream); if(fb.done) cudaEventDestroy(fb.done);
    fb = FrameBuffers{};
}