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
#include "mma.h"

#include "../FX_m2.cu"

#ifdef OPTLDS64
#include "store_and_transform_output_optLDS64.cuh"
#include "../outer_product.cuh"
#elif OPTSTS64_CMP
#include "store_and_transform_output_optSTS64_compact.cuh"
#include "../outer_product_ggml.cuh"
#else
// #include "store_and_transform_output_optSTS64.cuh"
#include "store_and_transform_output_optSTS64_32Tx64x8.cuh"
#include "../outer_product.cuh"
#endif

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

#ifndef _WINOGRAD_
#define _WINOGRAD_
extern "C"
{


#define d(input, i, j, off) ( input[(i<<2) + (j) + (off)] )

__device__ __forceinline__ void load_and_transform_input_tile(half *Btd, half *pOutputs){

  half workspace[3]; 
  int c_offset = BC*BN;
  int c_tensor = threadIdx.x*BC + threadIdx.y*2;
  int offset = 0, offset1 = 0;
  for(int k=0; k<2; k++){
    #pragma unroll
    for(int j=0; j<4; j++){
      workspace[0] = Btd[j+offset];
      workspace[1] = Btd[j+offset+4];
      workspace[2] = Btd[j+offset+8];

      Btd[j+offset]    = workspace[0] - workspace[2];
      Btd[j+4+offset]  = workspace[1] + workspace[2];
      Btd[j+8+offset]  = workspace[2] - workspace[1];
      Btd[j+12+offset] = workspace[1] - Btd[j+12+offset];
    }  
    
    #pragma unroll
    for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
      pOutputs[c_tensor+i*c_offset*4 + offset1] = d(Btd, i, 0, offset) - d(Btd, i, 2, offset);  
      pOutputs[c_tensor+i*c_offset*4+c_offset + offset1] = d(Btd, i, 1, offset) + d(Btd, i, 2, offset);
      pOutputs[c_tensor+i*c_offset*4+2*c_offset + offset1] = d(Btd, i, 2, offset) - d(Btd, i, 1, offset);
      pOutputs[c_tensor+i*c_offset*4+3*c_offset + offset1] = d(Btd, i, 1, offset) - d(Btd, i, 3, offset);
      // if(pOutputs[c_tensor+i*c_offset*4] < 0.f)
      //     printf(" A, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //          threadIdx.x, threadIdx.y, i, pOutputs[c_tensor+i*c_offset*4]);
      // if(pOutputs[c_tensor+i*c_offset*4+c_offset] < 0.f)
      //     printf(" B, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //          threadIdx.x, threadIdx.y, i, pOutputs[c_tensor+i*c_offset*4+c_offset]);
      // if(pOutputs[c_tensor+i*c_offset*4+2*c_offset] < 0.f)
      //     printf(" C, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //          threadIdx.x, threadIdx.y, i, pOutputs[c_tensor+i*c_offset*4+2*c_offset]);
      // if(pOutputs[c_tensor+i*c_offset*4+3*c_offset] < 0.f)
      //     printf(" D, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //          threadIdx.x, threadIdx.y, i, pOutputs[c_tensor+i*c_offset*4+3*c_offset]);
    }     
    offset += 16;
    offset1 += 1;
  }

}

__device__ __forceinline__ void load_filter_tile(const half *tiles, half *pOutputs, 
                                int filt_c, int filt_k){
 
  int c_tensor_s = threadIdx.x*BC + threadIdx.y*2;
  int c_offset_s = BK*BC;
  
  // each thread in row 0 puts its first element of 1st filter tile(loaded by the thread) in smem
  // taking 32 slots 
  // then puts its first element of 2nd filter tile immediately after, taking another 32 slots
  // then followed by threads in row 1, 2.. until 7

  // Note the next element is BK*BC (8*64) slots away, then another BK*BC ....
  // for every 64 values, the first 32 belongs to filter tile 1, the next 32 for filter tile 2 

  for(int k=0; k<2; k++){ // prefetch 2 filter tiles of 1 channel/thread
    for(int i=0; i<4; i++){
      #pragma unroll
      for(int j=0; j<4; j++){
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s]     = tiles[k*16 + i*4 + j];
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s + 1] = tiles[32 + k*16 + i*4 + j]; //32 = 2*BC
      }
    }
    // 2nd tile right behind the 1st?
    c_tensor_s += BN/2; // BN has nothing to do with input tiles
  }
  
}

__device__ __forceinline__ void prefetch_filter_tile(const half *pInputs, half *tiles, int filt_k){

  int c_offset = (filt_k<<4);
  int c_tensor = blockIdx.z*BK + threadIdx.y*2*c_offset + threadIdx.x; // Iny*filt_k*4*4

  // each threadIdx.y corresponds to 2 channels; there are 8 different threadIdx.y so 16 channels 
  
  //each thread (32 threads in x direction) loads 4 kernel tiles (2 for each channel and 32 in K direction apart)
  
  int acumm;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = (i*filt_k<<2);
      #pragma unroll
      for(int j=0; j<4; j++){
          tiles[(i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor];
          tiles[16 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor + BN];
          tiles[32 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor + c_offset];
          tiles[48 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor + BN + c_offset];
      }
  }
}

__device__ __forceinline__ void prefetch_input_tile(const half *pInputs, half *tile, int in_h, 
                       int in_w, int tw, int th, unsigned short mask){
  
  // load two input tiles per thread
  // int tx = in_w / gridDim.x, ty = in_h / gridDim.y;  
  int c_offset = in_h*in_w; 
  int c_tile = blockIdx.x * TW  + blockIdx.y * in_w * TH; 
  int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
                threadIdx.y*2*c_offset - (in_w+1);

      // + threadIdx.y*(in_h*in_w) + (in_w+1);
  // if(threadIdx.x/in_n != 0){
  //   printf(" %d, %d, %d, %d \n", blockIdx.x, blockIdx.y,  threadIdx.x, threadIdx.y);
  // }
  int acumm,x;
  //short x1,x2;     

  // if(blockIdx.y==0 && (threadIdx.x / tw) == 0)   mask&=0xFFF0;  // pad top row
  // if(blockIdx.y==gridDim.y-1 && threadIdx.x / tw == th-1) mask &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
  // if(blockIdx.x==gridDim.x-1 && (threadIdx.x % tw) == tw-1) mask &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
  // if(blockIdx.x == 0 && (threadIdx.x % tw) == 0)   mask&=0xeeee;  // pad left col
  
  // if(threadIdx.x > 0) return; // only thread needed per threadIdx.y

  // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
  //     && threadIdx.x == 31 && threadIdx.y == 0){
  //         printf("X, %hu \n", mask);   
  //   }
           
  if(mask==0xFFFF){
    #pragma unroll
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        tile[(i<<2) + j] = pInputs[acumm + j + c_tensor]; //1st channel
        tile[(i<<2) + j + 16] = pInputs[acumm + j + c_tensor + c_offset];//2nd channel
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 31 && threadIdx.y == 0){
        //      printf("A, %d, %d, %d, %f, %d\n", i, j, acumm+j, tile[(i<<2) + j],acumm + j + c_tensor);   
        // }
      }
    }

  } else {
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[x] = 0.f;
        if(mask&(1<<x)){
          tile[x]=pInputs[acumm + j + c_tensor];
          tile[x+16]=pInputs[acumm + j + c_tensor + c_offset];
        }
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 28 && threadIdx.y == 0){
        //      printf("B, %d, %d, %d, %d, %hu, %s, %f, %d\n", i, j, x, acumm+j, mask, mask&(1<<x)?"t":"f",tile[x],acumm + j + c_tensor);   
        // }        
      }
    }
  }
}


__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 32x16    
    for (int i = 0; i < 2; ++i)
    {        
      nvcuda::wmma::load_matrix_sync(frag[i], smem + i * 256, 16);
    }
}


__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 16x64    
    for (int i = 0; i < 4; ++i)
    {      
      nvcuda::wmma::load_matrix_sync(frag[i], smem + i * (16 * 16), 16);
    }
}



__global__ void Winograd_kernel(half *A, half *B, float *C,
                    int tiles_dim_w, int tiles_dim_h,
                    int in_c, int in_h, int in_w,
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int out_c,
                    int tile_2d_s, int out_h, int out_w){

  __align__(128) extern __shared__ unsigned char shared_mem[];
  half *input_smem  = reinterpret_cast<half *>(shared_mem);
  half *filter_smem = input_smem + 16*BC*BN;

  unsigned short m = 0xFFFF;
  // if((blockIdx.y/tiles_dim)==0)   m&=0xFFF0;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1)) m &= (!(in_w%2))?(0x0FFF):(0x00FF);
  // if(!((blockIdx.y+1)%tiles_dim)) m &= (!(in_w%2))?(0x7777):(0x3333);
  // if(!((blockIdx.y)%tiles_dim))   m&=0xeeee;

  if(blockIdx.y==0 && (threadIdx.x / X) == 0)   m &= 0xFFF0;  // pad top row
  if(tiles_dim_w % X == 0 && tiles_dim_h % Y == 0){
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
  }else if(tiles_dim_w % X == 0){
    int k = in_h % TH; 
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == k1-1) m &= (!(k%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }else if(tiles_dim_h % Y == 0){
    int k = in_w % TW;   
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 8*k1 tiles
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == k1-1) m &= (!(k%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist 
  }else{
    int kh = in_h % TH; 
    int kw = in_w % TW;   
    int kh1 =  kh % 2 ? (kh+1)/2 : kh/2; // there could be kh1*kw1 tiles
    int kw1 =  kw % 2 ? (kw+1)/2 : kw/2; 
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == kh1-1) m &= (!(kh%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > kh1-1) m &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == kw1-1) m &= (!(kw%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > kw1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }  
  if(blockIdx.x==0 && (threadIdx.x % X) == 0)   m &=0xeeee;  // pad left col
  
  half img_tile[32]; // Prefetch input from GMEM
  half filter_tile[64]; // Prefetch filter from GMEM

  // float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  // float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  // float4 accumulator[2][16] = {0.0f};  // Accumulators 

  half *A_frag; // Input data pointer
  int frag_offset = 2 * (BN*BC); // (2=8/4) SMEM input read offset

  half *B_frag; // Filter data pointer  
  int f_frag_offset = 2 * (BC*BK); // (2=8/4 with 4 being float4) SMEM filter read offset 

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[BN / wmmaM];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragB[BK / wmmaN];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[2 * BN / wmmaM * BK / wmmaN];

  for (int k = 0; k < 2; k++){
    for (int mii = 0; mii < BN / wmmaM; mii += 1){
      for (int nii = 0; nii < BK / wmmaN; nii += 1){
        nvcuda::wmma::fill_fragment(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + nii], 0.0);      
      }
    }
  }

  prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
  prefetch_filter_tile(B, filter_tile, filt_k);

  // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
  //     for(int j = 0; j < 16; j++){
  //       printf( "%f,", img_tile[j]);
  //     }
  //     printf("\n");
  //   }

  


  // Mainloop - iterates over the entire K dimension - not unrolled

  // wee need to do 16-batched 32x16x64 MM, each wmma will do 16x16x16 so 
  // we need to do 16 2x4 wmmas's 
  // we allocate 2 FragA and 4 FragB and 16 Accum, then in a loop of 2 iterations 
  // reuse 2 FragA and 4 FragB
  //    
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration

    

    // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    //     printf("A %d, [", iter);
    //     for(int j = 0; j < 16; j++){
    //       printf( "%f,", img_tile[j]);
    //     }
    //     printf("]\n");
    //   }

    load_and_transform_input_tile(img_tile, input_smem);
    load_filter_tile(filter_tile, filter_smem, filt_c, filt_k);

    __syncthreads();

    // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    //     // printf("A %d, %d, %f, %f, %f \n", iter, i, input_frag[1], input_frag[0], accumulator[1][0]);
    //     printf("iter: %d, \n ",iter);
    //     for(int i = 0; i < 16; i++){
    //       printf("%d ,[", i);
    //       for(int j = 0; j < 8; j++){
    //         printf( "%f,", input_smem[i*BC + j]);
    //       }
    //     printf("]\n");
    //     }
    //   }
    
    // now both input and filter tiles are in smem, we can load wmma frags and do wmma computation  
    
    for(int k = 0; k < 2; k++){
      A_frag = input_smem  + threadIdx.y*BN*BC + k*8*BN*BC;
      B_frag = filter_smem + threadIdx.y*BC*BK + k*8*BC*BK;
      loadFragA(FragA, A_frag, k);
      loadFragB(FragB, B_frag, k);
      for(int mii = 0; mii < BN / wmmaM; mii++){
        for(int nii = 0; nii < BK / wmmaN; nii++){
            // 16x16x16 for each wmma
            nvcuda::wmma::mma_sync(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BN / wmmaN) + nii], 
            FragA[mii], FragB[nii], Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + nii]);
        }
      }
    }
    
    
    A += BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
      prefetch_filter_tile(B, filter_tile, filt_k);
    }

    __syncthreads();
  }

  // Transpose, transform and store accumulated result
  store_output_tile(Accum, shared_mem, C, out_h, out_w, tiles_dim_w, tiles_dim_h, X, Y);
                  
                     
}

cudaError_t convolutionForward_32Tx64x8(half *k, int in_h, int in_w, half *w, int out_h,
                  int out_w, int out_c, float *C, half *Ww,                 
                int tiles_dim_w, int tiles_dim_h, int tile_size,
                int in_c, int filt_k, int filt_c, int filt_h, int filt_w, int alpha, int m){

  int tile_2d_s = tile_size*tile_size;
  // int tiles_2d_dim = tiles_dim*tiles_dim;
  int smem_size = (16*BN*BC + 16*BC*BK)*2;
  int X = 4, Y = 8;
  

  FX<<<dim3(filt_k/BK, filt_c/BC), dim3(32, BC)>>>(w, Ww, filt_k, filt_c, filt_h, filt_w, alpha);
        
  #ifdef OPTSTS64_CMP
  smem_size = 65536; // 64 KB
  cudaFuncSetAttribute(Winograd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  #endif
  // printf("launching %d blocks in y \n ", tiles_2d_dim); 

  // each thread block will load 32 tiles (4x4) from the single image input
  // we let X*Y = 32 and arbitraraly pick X = 4 and Y = 8
  // Winograd_kernel<<<dim3(1, tiles_2d_dim, filt_k/BK), dim3(BN, 8), smem_size>>>(k, Ww, C, 
  Winograd_kernel<<<dim3((tiles_dim_w+X-1)/X, (tiles_dim_h+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size>>>(k, Ww, C,
  tiles_dim_w, tiles_dim_h, in_c, in_h, in_w, tile_size, X, Y, filt_k, filt_c, out_c, tile_2d_s, out_h, out_w);

  return cudaGetLastError();
}

}
#endif
