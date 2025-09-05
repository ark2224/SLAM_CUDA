// =============================================================
// FILE: demo_main.cu
// Purpose: End-to-end driver that exercises Steps 1–5.
// Build: compiled together with the other five .cu files
// =============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)


// ---- Step 1
extern "C" void step1_profile_example(int N,int warmup,int iters);
// ---- Step 2
extern "C" void step2_frontend_example(int W,int H,int max_kp);

// ---- Step 3 (struct must match step3_backend_pcg.cu)
__global__ void vec_set(int n, float v, float* __restrict__ x);
struct CsrMatrix { int n; int nnz; int *rowPtr; int *colInd; float *vals; };
extern "C" bool step3_pcg_solve(const CsrMatrix& A, const float* d_b, float* d_x, int iters, float tol);

// ---- Step 4 (struct must match step4_streaming.cu)
struct FrameBuffers {
    unsigned char *d_img, *d_msk; float *d_scores; uint2 *d_kp; int *d_cnt;
    unsigned char *h_img, *h_msk; int *h_cnt; int W,H,max_kp; cudaStream_t stream; cudaEvent_t done;
};
extern "C" void step4_allocate(FrameBuffers& fb, int W, int H, int max_kp);
extern "C" void step4_process_async(FrameBuffers& fb);
extern "C" bool step4_poll_complete(FrameBuffers& fb);
extern "C" int step4_get_count(const FrameBuffers& fb);
extern "C" void step4_free(FrameBuffers& fb);

// ---- Step 5 (types must match step5_state_bus.cu)
#include "state_bus.hpp"
extern "C" void step5_publish_example(StateRing<256>& ring, TailLatencyGuard& g, double t_now,
                                      const float* T_wb_pos, const float* T_wb_quat, const float* vel,
                                      const float* cov_pos_diag, unsigned int seq, double last_frame_ms);


int main(){
    // Step 1: profile placeholder
    step1_profile_example(1<<20, 3, 20);

    // Step 2: front-end demo
    step2_frontend_example(640,480,2000);

    // Step 4: streaming example (async)
    FrameBuffers fb{}; step4_allocate(fb, 640, 480, 2000);
    // Fill host pinned buffers with dummy data
    for (int y = 0; y < fb.H; ++y) {
        for (int x = 0; x < fb.W; ++x) {
            bool white = (((x >> 4) + (y >> 4)) & 1) != 0;
            fb.h_img[y*fb.W + x] = white ? 220 : 30;
            fb.h_msk[y*fb.W + x] = 0;  // static
        }
    }
    
    step4_process_async(fb);
    while(!step4_poll_complete(fb)) { /* spin/yield */ }
    int kp = step4_get_count(fb);
    printf("[Step4] async kp count = %d\n", kp);
    step4_free(fb);

    // Step 3: PCG (toy 3x3 SPD system: [[4,1,0],[1,3,0],[0,0,2]])
    CsrMatrix A{};
    A.n=3;
    A.nnz=7;
    CUDA_CHECK(cudaMalloc(&A.rowPtr,(A.n+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.colInd,A.nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.vals,A.nnz*sizeof(float)));

    int h_rp[4] = {0, 2, 5, 7};
    int h_ci[7] = {0,1,  0,1,2,  1,2};
    float h_v[7]  = {4,1,  1,3,1,  1,2}; // [[4,1,0],[1,3,1],[0,1,2]]
    CUDA_CHECK(cudaMemcpy(A.rowPtr,h_rp,sizeof(h_rp),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.colInd,h_ci,sizeof(h_ci),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.vals,h_v,sizeof(h_v),cudaMemcpyHostToDevice));

    float h_b[3]={1,2,3};
    float *d_b=nullptr,*d_x=nullptr;
    CUDA_CHECK(cudaMalloc(&d_b,3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,3*sizeof(float),cudaMemcpyHostToDevice));
    // zero-initialize x
    { CUDA_CHECK(cudaMemset(d_x, 0, 3*sizeof(float))); }

    bool ok = step3_pcg_solve(A,d_b,d_x,100,1e-6f);
    float h_x[3]; CUDA_CHECK(cudaMemcpy(h_x,d_x,3*sizeof(float),cudaMemcpyDeviceToHost));
    printf("[Step3] PCG ok=%d solution ~ [%.4f %.4f %.4f]\n", ok, h_x[0],h_x[1],h_x[2]);
    cudaFree(A.rowPtr); cudaFree(A.colInd); cudaFree(A.vals); cudaFree(d_b); cudaFree(d_x);

    // Step 5: publisher demo
    StateRing<256> ring{}; TailLatencyGuard guard(64, /*p95 limit*/ 8.0);
    float pos[3]={0,0,0}, quat[4]={1,0,0,0}, vel[3]={0,0,0}, cov[3]={0.1f,0.1f,0.1f};
    for(int i=0;i<10;++i){ step5_publish_example(ring,guard, (double)i*0.02, pos, quat, vel, cov, i, /*last_frame_ms*/ 5.0 + (i%3)); }
    printf("[Step5] Degrade=%s (p95=%.2fms)\n", guard.degrade?"true":"false", guard.percentile95());

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}


/*

mkdir build && cd build
cmake ..
cmake --build . -j
./slam_cuda_demo

*/