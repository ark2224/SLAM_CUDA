#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>


#define CUDA_CHECK(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)


struct CsrMatrix {
    int n; // dimension
    int nnz; // non-zeros
    int *rowPtr; // size n+1
    int *colInd; // size nnz
    float *vals; // size nnz
};


__global__ void spmv_csr_kernel(const int n, const int* __restrict__ rowPtr, const int* __restrict__ colInd,
                                const float* __restrict__ vals, const float* __restrict__ x, float* __restrict__ y){
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r>=n) return;
    float sum=0.f; int start=rowPtr[r], end=rowPtr[r+1];
    for(int i=start;i<end;++i) sum += vals[i]*x[colInd[i]];
    y[r]=sum;
}


__global__ void vec_axpy(int n, float a, const float* __restrict__ x, float* __restrict__ y){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i<n) y[i] += a*x[i];
}
__global__ void vec_scale(int n, float a, float* __restrict__ x){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=a;
}
__global__ void vec_copy(int n, const float* __restrict__ x, float* __restrict__ y){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i];
}


__global__ void vec_pointwise_div(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]/(b[i]+1e-12f);
}


__global__ void vec_mul(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]*b[i];
}


__global__ void vec_set(int n, float v, float* __restrict__ x){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=v;
}


// Dot product via block reduction
__global__ void dot_kernel(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
    extern __shared__ float s[]; int i=blockIdx.x*blockDim.x+threadIdx.x; float v=0.f;
    if (i<n) v = a[i]*b[i]; s[threadIdx.x]=v; __syncthreads();
    // reduce
    for(int sft=blockDim.x/2; sft>0; sft>>=1){
        if(threadIdx.x<sft) s[threadIdx.x]+=s[threadIdx.x+sft];
        __syncthreads();
    }
    if(threadIdx.x==0) out[blockIdx.x]=s[0];
}


static float device_dot(int n, const float* a, const float* b){
    const int BS=256; int GS=(n+BS-1)/BS; size_t sh=BS*sizeof(float);
    float *d_partial=nullptr; CUDA_CHECK(cudaMalloc(&d_partial, GS*sizeof(float)));
    dot_kernel<<<GS,BS,sh>>>(n,a,b,d_partial);
    std::vector<float> h_partial(GS); CUDA_CHECK(cudaMemcpy(h_partial.data(),d_partial,GS*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    double sum=0.0; for(float v: h_partial) sum+=v; return (float)sum;
}


// Jacobi preconditioner (diag of A)
__global__ void csr_diag_kernel(int n, const int* __restrict__ rowPtr, const int* __restrict__ colInd, const float* __restrict__ vals, float* __restrict__ diag){
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=n) return;
    float d=0.f;
    for(int i=rowPtr[r]; i<rowPtr[r+1]; ++i) if(colInd[i]==r){ d=vals[i]; break; }
    diag[r]=d;
}


extern "C" bool step3_pcg_solve(const CsrMatrix& A, const float* d_b, float* d_x, int iters=100, float tol=1e-4f){
    const int n=A.n; const int BS=256; int GS=(n+BS-1)/BS;
    float *d_r=nullptr,*d_p=nullptr,*d_Ap=nullptr,*d_z=nullptr,*d_M=nullptr;
    CUDA_CHECK(cudaMalloc(&d_r,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_p,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_z,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M,n*sizeof(float)));
    // M = diag(A)
    csr_diag_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_M);
    // r = b - A*x
    spmv_csr_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_x,d_Ap);
    CUDA_CHECK(cudaDeviceSynchronize());
    // r = b - Ap
    vec_copy<<<GS,BS>>>(n,d_b,d_r); vec_axpy<<<GS,BS>>>(n,-1.0f,d_Ap,d_r);

    // z = M^{-1} r (Jacobi)
    vec_pointwise_div<<<GS,BS>>>(n,d_r,d_M,d_z);
    vec_copy<<<GS,BS>>>(n,d_z,d_p);

    float rz_old = device_dot(n,d_r,d_z);
    float b_norm = sqrtf(device_dot(n,d_b,d_b)+1e-12f);

    bool ok=false;
    for(int k=0;k<iters;++k){
        spmv_csr_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_p,d_Ap);
        float pAp = device_dot(n,d_p,d_Ap) + 1e-12f;
        float alpha = rz_old / pAp;
        vec_axpy<<<GS,BS>>>(n, alpha, d_p, d_x); // x += alpha p
        vec_axpy<<<GS,BS>>>(n,-alpha, d_Ap, d_r); // r -= alpha A p


        float r_norm = sqrtf(device_dot(n,d_r,d_r));
        if (r_norm / b_norm < tol){ ok=true; break; }

        vec_pointwise_div<<<GS,BS>>>(n,d_r,d_M,d_z); // z = M^{-1} r
        float rz_new = device_dot(n,d_r,d_z);
        float beta = rz_new / (rz_old + 1e-12f);

        // p = z + beta p
        // p *= beta; p += z;
        vec_scale<<<GS,BS>>>(n,beta,d_p);
        vec_axpy<<<GS,BS>>>(n,1.0f,d_z,d_p);
        rz_old = rz_new;
    }

    CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_Ap)); CUDA_CHECK(cudaFree(d_z)); CUDA_CHECK(cudaFree(d_M));
    return ok;
}