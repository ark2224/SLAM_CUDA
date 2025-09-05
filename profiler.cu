#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>


#ifdef __has_include
#   if __has_include(<nvToolsExt.h>)
#       include <nvToolsExt.h>
#       define HAS_NVTX 1
#   else
#       define HAS_NVTX 0
#   endif
#else
#   define HAS_NVTX 0
#endif


#define CUDA_CHECK(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)


__global__ void noop_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 1.000001f; // prevent optimization-away
}


struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~GpuTimer(){ cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(start_, s)); }
    float stop(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(stop_, s)); CUDA_CHECK(cudaEventSynchronize(stop_)); float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start_,stop_)); return ms; }
};


struct HostTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void tic(){ t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};


static double percentile(std::vector<double> v, double p){
    if (v.empty()) return 0.0;
    size_t k = (size_t)((p/100.0)*(v.size()-1));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
}

extern "C" void step1_profile_example(int N=1<<20, int warmup=5, int iters=50){
    const int BS=256; int GS=(N+BS-1)/BS;
    float *d_in=nullptr,*d_out=nullptr; CUDA_CHECK(cudaMalloc(&d_in,N*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_out,N*sizeof(float)));
    std::vector<float> h_in(N, 1.0f); CUDA_CHECK(cudaMemcpy(d_in,h_in.data(),N*sizeof(float),cudaMemcpyHostToDevice));
    std::vector<double> frame_ms; frame_ms.reserve(iters);

    // Reproducibility hint: fixed seeds (not shown) & fixed clocks (if possible)

    // Warmup
    for(int i=0;i<warmup;++i){
#if HAS_NVTX
        nvtxRangePushA("noop_warmup");
#endif
        noop_kernel<<<GS,BS>>>(d_out,d_in,N);
        CUDA_CHECK(cudaDeviceSynchronize());
#if HAS_NVTX
        nvtxRangePop();
#endif
    }


    // Measured runs
    for(int i=0;i<iters;++i){
        GpuTimer gt; gt.start();
#if HAS_NVTX
        nvtxRangePushA("noop_iter");
#endif
        noop_kernel<<<GS,BS>>>(d_out,d_in,N);
        float gpu_ms = gt.stop();
#if HAS_NVTX
        nvtxRangePop();
#endif
        frame_ms.push_back(gpu_ms);
    }


    double p50 = percentile(frame_ms, 50.0);
    double p95 = percentile(frame_ms, 95.0);
    printf("[Step1] noop_kernel N=%d => p50=%.3f ms, p95=%.3f ms (iters=%d)\n", N, p50, p95, iters);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}