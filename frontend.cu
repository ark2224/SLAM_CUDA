#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>


#define CUDA_CHECK(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)


// Minimal FAST corner score (Bresenham circle of 16, threshold t)
__device__ __forceinline__ int fast_score(const uint8_t* img, int w, int h, int x, int y, int t){
    if (x<3||y<3||x>=w-3||y>=h-3) return 0;
    const int off = y*w + x; int c = img[off];
    // circle offsets (12 o'clock start, clockwise)
    const int dx[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
    const int dy[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
    int brighter=0, darker=0;
    // Short early exits: need >= 9 contiguous; here we just count total as score proxy
    #pragma unroll
    for(int i=0;i<16;++i){ int v = img[(y+dy[i])*w + (x+dx[i])]; brighter += (v >= c + t); darker += (v <= c - t); }
    return max(brighter, darker); // proxy for FAST score
}

// Kernel: compute FAST scores with semantic mask down-weighting
// mask: 0=static, 1=dynamic (down-weight), 255=unknown (treat static)
__global__ void fast_score_kernel(const uint8_t* __restrict__ img, const uint8_t* __restrict__ mask,
                                  int w, int h, int stride, int t, float dyn_w,
                                  float* __restrict__ scores){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int s = fast_score(img, w, h, x, y, t);
    uint8_t m = mask ? mask[y*stride + x] : 0;
    float wgt = (m==1) ? dyn_w : 1.0f;
    scores[y*w + x] = wgt * (float)s;
}


// Simple nonmax suppression in 3x3 window (shared-memory tile)
__global__ void nms3x3_kernel(const float* __restrict__ scores, int w, int h, float thresh,
                              uint2* __restrict__ keypoints, int max_kp, int* __restrict__ out_count){
    extern __shared__ float tile[]; // (BLOCK_Y+2)*(BLOCK_X+2)
    constexpr int BX=16, BY=16;
    int tx=threadIdx.x, ty=threadIdx.y;
    int x = blockIdx.x*BX + tx; int y = blockIdx.y*BY + ty;


    // Load with 1-pixel halo
    int lx = tx+1, ly = ty+1; int W = BX+2; int H = BY+2;
    if (x<w && y<h) tile[ly*W + lx] = scores[y*w + x];
    if (tx==0 && x>0 && y<h) tile[ly*W + 0] = scores[y*w + (x-1)];
    if (tx==BX-1 && x+1<w && y<h) tile[ly*W + (lx+1)] = scores[y*w + (x+1)];
    if (ty==0 && y>0 && x<w) tile[0*W + lx] = scores[(y-1)*w + x];
    if (ty==BY-1 && y+1<h && x<w) tile[(ly+1)*W + lx] = scores[(y+1)*w + x];
    // corners of halo
    if (tx==0 && ty==0 && x>0 && y>0) tile[0*W+0] = scores[(y-1)*w + (x-1)];
    if (tx==BX-1 && ty==0 && x+1<w && y>0) tile[0*W+(lx+1)] = scores[(y-1)*w + (x+1)];
    if (tx==BX-1 && ty==BY-1 && x+1<w && y+1<h) tile[(ly+1)*W+(lx+1)] = scores[(y+1)*w + (x+1)];
    __syncthreads();


    if (x>=w || y>=h) return;
    float c = tile[ly*W + lx];
    if (c < thresh) return;
    bool is_max = true;
    #pragma unroll
    for(int dy=-1; dy<=1; ++dy){
        #pragma unroll
        for(int dx=-1; dx<=1; ++dx){
            if (dx==0 && dy==0) continue;
            is_max &= (c >= tile[(ly+dy)*W + (lx+dx)]);
        }
    }
    if (is_max){
        int idx = atomicAdd(out_count, 1);
        if (idx < max_kp) keypoints[idx] = make_uint2(x,y);
    }
}

// BRIEF descriptor (256-bit) with pre-defined pairs
__constant__ int2 c_brief_pairs[256];

__global__ void brief_kernel(const uint8_t* __restrict__ img, int w, int h, const uint2* __restrict__ kps,
                             int num_kp, uint32_t* __restrict__ desc){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_kp) return;
    int x = kps[i].x, y = kps[i].y;
    uint32_t d[8] = {0};
    #pragma unroll
    for(int b=0;b<256;++b){
        int2 p = c_brief_pairs[b];
        int x1 = min(max(x + (p.x>>8), 0), w-1); // high byte = dx1, low byte = dy1 (packing trick if desired)
        int y1 = min(max(y + (p.x & 0xFF) - 128, 0), h-1);
        int x2 = min(max(x + (p.y>>8), 0), w-1);
        int y2 = min(max(y + (p.y & 0xFF) - 128, 0), h-1);
        int bit = (img[y1*w+x1] < img[y2*w+x2]);
        d[b>>5] |= (bit << (b & 31));
    }
    // write 8 words (256 bits)
    #pragma unroll
    for(int j=0;j<8;++j) desc[i*8 + j] = d[j];
}

// Hamming distance for 256-bit BRIEF
__device__ __forceinline__ int hamming256(const uint32_t* a, const uint32_t* b){
    int d=0;
    #pragma unroll
    for(int i=0;i<8;++i) d += __popc(a[i]^b[i]);
    return d;
}


// Blocked matcher: for each kpA[i], find best match in kpB within window
__global__ void match_brief_kernel(const uint32_t* __restrict__ descA, const uint2* __restrict__ kpA, int nA,
                                   const uint32_t* __restrict__ descB, const uint2* __restrict__ kpB, int nB,
                                   int max_dist, int* __restrict__ match_idx, int* __restrict__ match_dist){
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=nA) return;
    const uint32_t* dA = &descA[i*8];
    int best_j=-1, best_d=1e9;
    for(int j=0;j<nB;++j){
        int d = hamming256(dA, &descB[j*8]);
        if (d < best_d){ best_d=d; best_j=j; }
    }
    if (best_d <= max_dist){ match_idx[i]=best_j; match_dist[i]=best_d; }
    else { match_idx[i]=-1; match_dist[i]=INT_MAX; }
}


// IMU-prior inlier filter: given a provisional Essential matrix E (from IMU delta-R,t dir),
// compute x2^T E x1 residuals and Huber weight.
__global__ void epipolar_inlier_kernel(const float* __restrict__ E9, // row-major 3x3
                                       const float2* __restrict__ pts1, const float2* __restrict__ pts2,
                                       int n, float huber, uint8_t* __restrict__ inlier){
    float E00=E9[0],E01=E9[1],E02=E9[2];
    float E10=E9[3],E11=E9[4],E12=E9[5];
    float E20=E9[6],E21=E9[7],E22=E9[8];
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=n) return;
    float x1=pts1[i].x, y1=pts1[i].y, x2=pts2[i].x, y2=pts2[i].y;
    float l0 = E00*x1 + E01*y1 + E02;
    float l1 = E10*x1 + E11*y1 + E12;
    float l2 = E20*x1 + E21*y1 + E22;
    float r = x2*l0 + y2*l1 + l2; // epipolar residual
    float a = fabsf(r);
    inlier[i] = (a < huber) ? 1 : 0;
}

// Host wrapper for Step 2 (toy wiring)
extern "C" void step2_frontend_example(int W, int H, int max_kp){
    const dim3 BS2(16,16), GS2((W+15)/16,(H+15)/16);
    // Allocate dummy image + mask
    std::vector<uint8_t> h_img(W*H), h_msk(W*H, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool white = (((x >> 4) + (y >> 4)) & 1) != 0;
            h_img[y*W + x] = white ? 220 : 30;
        }
    }

    uint8_t *d_img=nullptr,*d_msk=nullptr; CUDA_CHECK(cudaMalloc(&d_img,W*H)); CUDA_CHECK(cudaMalloc(&d_msk,W*H));
    CUDA_CHECK(cudaMemcpy(d_img,h_img.data(),W*H,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msk,h_msk.data(),W*H,cudaMemcpyHostToDevice));


    float *d_scores=nullptr; CUDA_CHECK(cudaMalloc(&d_scores,W*H*sizeof(float)));
    fast_score_kernel<<<GS2,BS2>>>(d_img,d_msk,W,H,W,20,0.3f,d_scores);


    uint2 *d_kp=nullptr; int *d_cnt=nullptr; CUDA_CHECK(cudaMalloc(&d_kp,max_kp*sizeof(uint2))); CUDA_CHECK(cudaMalloc(&d_cnt,sizeof(int))); CUDA_CHECK(cudaMemset(d_cnt,0,sizeof(int)));
    size_t shmem = (16+2)*(16+2)*sizeof(float);
    nms3x3_kernel<<<GS2,BS2,shmem>>>(d_scores,W,H,5.0f,d_kp,max_kp,d_cnt);


    int h_cnt=0; CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost)); h_cnt = std::min(h_cnt, max_kp);
    printf("[Step2] detected %d keypoints (capped at %d)\n", h_cnt, max_kp);

    // Clean up (descriptors/matching omitted in this short example)
    CUDA_CHECK(cudaFree(d_scores)); CUDA_CHECK(cudaFree(d_kp)); CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_msk));
}

