// Copyright 2021 Roberto Lopez Castro
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <omp.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
// #include <sys/time.h>
// #include <sys/resource.h>

#include <cudnn.h>

#include "config_32Tx64x8.hpp"

#ifdef BASE
  #if __CUDA_ARCH__ < 800
  #include "convolutionForward_32x64x8_baseline.cu"
  #else
  #include "ampere/convolutionForward_32x64x8_baseline.cu"
  #endif
#else
  // #if __CUDA_ARCH__ < 800
  // #include "convolutionForward_32x64x8.cu"  
  // #else 
  #include "ampere/convolutionForward_32Tx64x8.cu"
  // #include "ampere/convolutionForward_40x40x8.cu"
  // #endif
#endif

/*
   In order to measure the elapsed time:

   resnfo: datatype defined to abstract the metric of the resources to use
   timenfo: datatype defined to abstract the time metric to use

   timestamp: it abstract the function used to take the time

   printtime: it abstracts the function used to print the time

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): function to obtain the
                       time between two measures
*/

// #ifdef _noWALL_
// typedef struct rusage resnfo;
// typedef struct _timenfo {
//   double time;
//   double systime;
// } timenfo;
// #define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
// #define printtime(t) printf("%15f s (%f user + %f sys) ",		\
// 			    t.time + t.systime, t.time, t.systime);
// #else
// typedef struct timeval resnfo;
// typedef double timenfo;
// #define timestamp(sample)     gettimeofday((sample), 0)
// #define printtime(t) printf("%15f s ", t);
// #endif

// void myElapsedtime(resnfo start, resnfo end, timenfo *t)
// {
// #ifdef _noWALL_
//   t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6))
//     - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
//   t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6))
//     - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
// #else
//   *t = (end.tv_sec + (end.tv_usec * 1E-6))
//     - (start.tv_sec + (start.tv_usec * 1E-6));
// #endif /*_noWALL_*/
// }

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    printf("    Error occurred: \n"); \
    std::exit(1); \
  } \
}

#define OPENCNN_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error occurred: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);  
   }
}

void tflops(int in_n, int in_w, int in_h, int in_c, int filt_w, int filt_h, int filt_k, int pad, int str, 
            int out_w, int out_h, float ms)
{
  
  double L = (double) 2.0*in_n*in_c*(in_h+2*PAD_H)*(in_w+2*PAD_W)*filt_k*3.0*3.0;

  printf("%.3f,%.2f", ms, L/(2.25 * ms * 1e9) );
}

__global__ void dev_const(float *px, float k, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
 
  curandState state;
  curand_init(clock64(), tid, 0, &state);

  if (tid < n)
    px[tid] = k;
}

__global__ void dev_const1(float *px, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
 
  curandState state;
  curand_init(clock64(), tid, 0, &state);

  if (tid < n)
    px[tid] = tid % (24*24) + 1.f;
    // px[tid] = k;
}

__global__ void dev_iota(float *px, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  curandState state;
  curand_init(clock64(), tid, 0, &state);
  
  if (tid < n)
    px[tid] = curand_uniform(&state);
}

__global__ void data_cpy(float *px, float *py, 
          int in_w, int in_h, int in_c, int in_n) {
  int tid = blockIdx.y + blockIdx.z*in_w + threadIdx.x*in_h*in_w + blockIdx.x*in_h*in_w*in_c;
  int id  = blockIdx.x + blockIdx.y*in_n + blockIdx.z*in_n*in_w + threadIdx.x*in_n*in_h*in_w;

  px[id] = py[tid];
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(12) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl << std::endl;
      }
    }
  }
  std::cout << std::endl;
}
  
void output_checker(float* A, float* B, int n, int len, int channel, int shift) {
  int error_cnt = 0, i, j, k, m;
  float max_error = 0;
  int kk = -1;
  for(k = 0; k < channel; k++){
    for (i = 0; i < len; i++) {
       if(k == kk)
          printf("[");
        for (j = 0; j < len; j++) {
        for (m = 0; m < n; m++) {
            float diff = fabs(
                A[k*len*len*n + i*len*n + j*n + m] - 
                B[m*len*len*channel + k*len*len + i*len + j]);
            if(k == kk){
            // if(i==0 && j==0){
              // printf("h:%d, w:%d, n:%d, c:%d -> %f vs %f : +- %f\n", i, j, m, k,
              // A[k*len*len*n + i*len*n + j*n + m],
              // B[m*len*len*channel + k*len*len + i*len + j], diff);              
              printf("(%.0f, %.0f, %d, %d)", 
              A[k*len*len*n + i*len*n + j*n + m],
              B[m*len*len*channel + k*len*len + i*len + j], j, i);
            //   printf("(%.0f, %.0f, %d)", 
            //   A[k*len*len*n + i*len*n + j*n + m],
            //   B[m*len*len*channel + k*len*len + i*len + j], k);
            }    
            if (diff > 1){ //1e-4
              error_cnt++;
              // printf("h:%d, w:%d, n:%d, c:%d -> %f vs %f : +- %f\n", i, j, m, k,
              // A[k*len*len*n + i*len*n + j*n + m],
              // B[m*len*len*channel + k*len*len + i*len + j], diff);
              // std::exit(1);
            }
            if (diff > max_error)
               max_error = diff;
        }
        }
        if(k == kk)
          printf("]\n"); 
    }
  }
  printf("[max_error: %f][error_cnt: %d] of %d\n", max_error, error_cnt, n*len*len*channel*shift);
}


cudaError_t convolutionForward(float *k, int in_h, int in_w, float *w, int out_h,
                                    int out_w, int out_c, float *C, float *Ww,
                                  int tiles_dim_w, int tiles_dim_h, int tile_size, int elems_dim_h,int elems_dim_w,
                                  int in_c, int filt_k, int filt_c, int filt_h, int filt_w,
                                  int alpha, int m){
  cudaError_t out;

  if(BN==32 && BK==64 && BC==8){
    out = convolutionForward_32Tx64x8(k, in_h, in_w, w, out_h,
                out_w, out_c, C, Ww,
                tiles_dim_w, tiles_dim_h, tile_size, in_c, filt_k, filt_c, filt_h, filt_w, alpha, m);
  // // } else 
  // if(BN==32 && BK==64 && BC==5){
  //    out = convolutionForward_40x40x8(k, in_h, in_w, w, out_h, out_w, out_n, out_c, C, Ww, n, tiles_dim, in_n, tile_size, in_c, filt_k, filt_c, filt_h, filt_w, alpha, m);
  } else {
    std::cout << "Configuration not supported yet" << std::endl;
  }

  return out;
}

cudaError_t init_data(float *in_data, float *in_data_open, float *filt_data, float *filt_data_open, int in_w, int in_h, int in_c, int in_n, int filt_w, int filt_h, int filt_c, int filt_k, int tile_size){

  int n = in_n*in_c*in_h*in_w;
  int blk_size = 256;

  dim3 dimBlock(blk_size);
  dim3 dimGrid((n + dimBlock.x -1)/dimBlock.x);

  // dev_iota<<<dimGrid, dimBlock>>>(in_data, n);
  dev_const1<<<dimGrid, dimBlock>>>(in_data, n);
  data_cpy<<<dim3(in_n, in_w, in_h), in_c>>>(in_data_open, in_data, in_w, in_h, in_c, in_n);

  n = filt_k*filt_c*filt_h*filt_w;
  dim3 dimGrid_f = dim3((n + dimBlock.x -1)/dimBlock.x);
  dev_const<<<dimGrid_f, dimBlock>>>(filt_data, 2.0f, n);
  // dev_iota<<<dimGrid_f, dimBlock>>>(filt_data, n);
  data_cpy<<<dim3(filt_k, filt_w, filt_h), dim3(filt_c)>>>(filt_data_open, filt_data, filt_w, filt_h, filt_c, filt_k);

  return cudaGetLastError();
}


void find_minmax(const float *val, const int l, float *mi, float *mx, int *mi_i, int *mx_i){
  *mi = FLT_MAX;
  *mx = -FLT_MAX;
  for(int i= 0; i < l; i++){
      if((*mi) > val[i]){
           (*mi) = val[i];
           (*mi_i) = i;
      }
      if((*mx) < val[i]){
           (*mx) = val[i];
           (*mx_i) = i;
      }
  }

}


int main(int argc, char *argv[]) {

 
  // ========== Set ImageBatch, filter, convolution and output parameters ========== //
  // ImageBatch
  const int in_n = (argc > 1)?atoi (argv[1]):N; // Number of images
  const int in_c = (argc > 2)?atoi (argv[2]):C_in; // Number of feature maps per image
  const int in_h = (argc > 3)?atoi (argv[3]):W; // Height of each feature map
  const int in_w = (argc > 4)?atoi (argv[4]):W; // Width of each feature map

  // Filter
  const int filt_k = (argc > 5)?atoi (argv[5]):K;
  const int filt_c = (argc > 6)?atoi (argv[6]):C_in;
  const int filt_h = (argc > 7)?atoi (argv[7]):R;
  const int filt_w = (argc > 8)?atoi (argv[8]):R;  

  std::cout << in_n << "," << in_c <<  "," << in_h <<  "," << filt_k <<  "," << filt_h << ",";

  // Convolution config
  const int pad_h = PAD_H; // Zero-padding height
  const int pad_w = PAD_W; // Zero-padding width
  const int str_h = STR_H; // Vertical filter stride
  const int str_w = STR_W; // Horizontal filter stride
  const int dil_h = DIL_H; // Filter height dilation
  const int dil_w = DIL_W; // Filter width dilation


  // Output
  int out_n; // Number of outputs
  int out_c; // Number of feature maps per output
  int out_h; // Height of each feature map
  int out_w; // Width of each feature map
     
  /*
   ####################################################################
   ======================= openCNN preparation =======================
   ####################################################################
   */
  // Winograd config
  const int m         = M;
  const int r         = filt_h;
  const int tile_size = m+r-1; // alpha value
  int elems_dim_w, elems_dim_h;
  int tiles_dim_w, tiles_dim_h;
  
  if(m==2){
    tiles_dim_w = ceil(ceil((double)(in_w+2)/2)-1);
    tiles_dim_h = ceil(ceil((double)(in_h+2)/2)-1);
    // tiles_dim = ceil(ceil((double)(min(in_w,in_h)+2)/2)-1);
    // tiles_dim = ceil(ceil((double)(in_h+2)/2)-1);
    elems_dim_w = tiles_dim_w*4;
    elems_dim_h = tiles_dim_h*4;
    fprintf(stderr, "%s: tiles_dim_w = %d, tiles_dim_h = %d \n", __func__, 
       tiles_dim_w, tiles_dim_h) ;
  } else {
    std::cout << "Configuration not supported yet" << std::endl;
    exit(0);
  }

  // Output
  out_n = in_n;   // Number of outputs
  out_c = filt_k; // Number of feature maps per output
  out_h = in_h;   // Height of each feature map
  out_w = in_w;   // Width of each feature map


	float mi, mx;
	int mi_i, mx_i;

  float *in_data_open;
  float *filt_data_open, *workspace;

  // ImageBatch openCNN
  OPENCNN_CALL(cudaMalloc(
        &in_data_open, in_n * in_c * in_h * in_w * sizeof(float))); 
  // Filter openCNN
  OPENCNN_CALL(cudaMalloc(
      &filt_data_open, filt_k * filt_c * filt_h * filt_w * sizeof(float)));  
  // Filter transformation
  OPENCNN_CALL(cudaMalloc(
      &workspace, filt_k * filt_c * tile_size * tile_size * sizeof(float)));  

  // Output openCNN    
  float *out_data;
  fprintf(stderr, "%s: out_n = %d,  out_c = %d, out_h = %d,  out_w = %d, total = %ld \n",
       __func__, out_n , out_c,  out_h, out_w, out_n * out_c * out_h * out_w *4);
  OPENCNN_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));  

  // =================== openCNN layouts =================== //    
  cudaMemcpyToSymbol(access_f_s, aux, 64*sizeof(int));
  cudaMemcpyToSymbol(access_s, aux2, 64*sizeof(int));
  // cudaMemcpyToSymbol(laneid, lid, 138*sizeof(int)); // no need for it now
  cudaMemcpyToSymbol(tileid, tid, 64*sizeof(int));
  #ifndef BASE
    #if defined(OPTLDS64)
    cudaMemcpyToSymbol(access_s_out, aux3, 32*sizeof(int));
    cudaMemcpyToSymbol(out_thread, aux4, 32*sizeof(int));  
    cudaMemcpyToSymbol(out_sgemm, aux5, 32*sizeof(int));
    cudaMemcpyToSymbol(exhange, aux6, 32*sizeof(int));
    #endif
  #else 
    cudaMemcpyToSymbol(access_s_out, aux3, 32*sizeof(int));
    cudaMemcpyToSymbol(out_thread, aux4, 16*sizeof(int));  
  #endif  

  /*
   ####################################################################
   ====================== cuDNN preparation ======================
   ####################################################################
   */




  float *in_data, *filt_data; 

  // ImageBatch cuDNN
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));
  // Filter cuDNN
  CUDA_CALL(cudaMalloc(
        &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

#if 1    
  // =================== Set descriptors =================== //
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // Input image Descriptors
  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW/*CUDNN_TENSOR_NHWC*/, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  // Filter Descriptors      
  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW/*CUDNN_TENSOR_NHWC*/,
        filt_k, filt_c, filt_h, filt_w));
  
  // Convolution Descriptors
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));  //CUDNN_CONVOLUTION
  
  
  // =================== Query output layout =================== //
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  // =================== Set and allocate output tensor descriptor ===================//
  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW/*CUDNN_TENSOR_NHWC*/, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));   

  float *out_data_cudnn;
  CUDA_CALL(cudaMalloc(
        &out_data_cudnn, out_n * out_c * out_h * out_w * sizeof(float)));   

  // =================== Query convolution forward algorithm =================== //
  // cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)6;
  // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // =================== Query workspace and allocate =================== //
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));


  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));
#endif
  
  // =================== Launch convolution on cuDNN =================== //
  float alpha = 1.f;
  float beta = 0.f;

  /*
   ####################################################################
   ============================= Init data =============================
   ####################################################################
   */

  OPENCNN_CALL(init_data(in_data, in_data_open, filt_data, filt_data_open, in_w, in_h, in_c, in_n,
    filt_w, filt_h, filt_c, filt_k, tile_size));

  /*
  ####################################################################
  ============================= Execution =============================
  ####################################################################
  */
  CUevent hStart, hStop;
  float ms;
  OPENCNN_CALL( cudaEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) ); // CU_EVENT_DEFAULT
  OPENCNN_CALL( cudaEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );
  
  // Loop of executions
  int iterations = 20;

  // Performs warmup operation
  OPENCNN_CALL(convolutionForward(in_data_open, in_h, in_w, filt_data_open, out_h,
                                  out_w, out_c, out_data, workspace,
                                   tiles_dim_w, tiles_dim_h, tile_size, elems_dim_w, elems_dim_h,
                                   in_c, filt_k, filt_c, filt_h, filt_w,
                                   tile_size, m));

  // ============================= openCNN exec =============================  
  cudaDeviceSynchronize();
  ( cudaEventRecord( hStart, NULL ) );
  for(int iter=0; iter<iterations; iter++){
    // fprintf(stderr, "%s: iter = %d \n", __func__, iter);
    OPENCNN_CALL(convolutionForward(in_data_open, in_h, in_w, filt_data_open, out_h,
                                    out_w, out_c, out_data, workspace,
                                   tiles_dim_w, tiles_dim_h, tile_size, elems_dim_w, elems_dim_h,
                                   in_c, filt_k, filt_c, filt_h, filt_w,
                                   tile_size, m));
  }
  ( cudaEventRecord( hStop, NULL ) );
  ( cudaEventSynchronize( hStop ) );
  ( cudaEventElapsedTime( &ms, hStart, hStop ) );
  ms = ms/iterations;
  tflops(in_n, in_w, in_h, in_c, filt_w, filt_h, filt_k, pad_w, str_w, out_w, out_h, ms);

  std::cout << ",";

#if 1
  // Performs warmup operation
  CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data,
    conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data_cudnn));


  // ============================= cuDNN exec =============================
  cudaDeviceSynchronize();
  ( cudaEventRecord( hStart, NULL ) );
  for(int iter=0; iter<iterations; iter++){
    CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data_cudnn));
  }
  ( cudaEventRecord( hStop, NULL ) );
  ( cudaEventSynchronize( hStop ) );
  ( cudaEventElapsedTime( &ms, hStart, hStop ) );
  ms = ms/iterations;
  tflops(in_n, in_w, in_h, in_c, filt_w, filt_h, filt_k, pad_w, str_w, out_w, out_h, ms);
#endif

  std::cout << std::endl;
  // ============================= Compare results =============================  
  std::cout << "********************************************" << std::endl;    
  float *tmp_openCNN = (float*) malloc (out_n*out_h*out_w*out_c*sizeof(float)),
      *tmp_cudnn   = (float*) malloc (out_n*out_h*out_w*out_c*sizeof(float)); 
  cudaMemcpy(tmp_openCNN, out_data, (out_n*out_h*out_w*out_c)<<2, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp_cudnn, out_data_cudnn, (out_n*out_h*out_w*out_c)<<2, cudaMemcpyDeviceToHost);


  find_minmax(tmp_openCNN, out_n*out_h*out_w*out_c, &mi, &mx, &mi_i, &mx_i);
	printf("openCNN: %f(%d), %f (%d) \n", mi, mi_i, mx, mx_i);
  find_minmax(tmp_cudnn, out_n*out_h*out_w*out_c, &mi, &mx, &mi_i, &mx_i);
	printf("cudnn: %f(%d), %f (%d) \n", mi, mi_i, mx, mx_i);
	
  
  output_checker(tmp_openCNN, tmp_cudnn, out_n, out_h, out_c, str_w);
  free(tmp_openCNN); free(tmp_cudnn); 


  CUDNN_CALL(cudnnDestroy(cudnn));
  CUDA_CALL(cudaFree(out_data_cudnn));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDA_CALL(cudaFree(in_data));
  CUDA_CALL(cudaFree(ws_data));

  CUDA_CALL(cudaFree(out_data));
  CUDA_CALL(cudaFree(filt_data_open));
  CUDA_CALL(cudaFree(workspace));
  CUDA_CALL(cudaFree(in_data_open));

  return 0;
}
