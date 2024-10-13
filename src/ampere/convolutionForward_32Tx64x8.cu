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
// #include "../outer_product.cuh"
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

/*__device__ __forceinline__ void load_and_transform_input_tile(half *Btd, half *pOutputs){

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
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1 && threadIdx.y == 0){
    //   printf("[");
    //   for(int j = 0; j < 16; j++){
    //     printf( "%f,", __half2float(Btd[j+offset]));
    //   }
    //   printf("]\n");      
    //  }
    
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

}*/


// smem layout for input tile
// ___________32T(C0)______32T(C1)____... _____32T(C15) E0
// ___________32T(C0)______32T(C1)____... _____32T(C15) E1
// .....
// .....
// ___________32T(C0)______32T(C1)____... _____32T(C15) E15

__device__ __forceinline__ void load_and_transform_input_tile(half *Btd, half *pOutputs){

  half workspace[3]; 
  int c_offset = BC*BN;
  int c_tensor = threadIdx.x + threadIdx.y*2*BN;
  int offset = 0;
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
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1 && threadIdx.y == 0){
    //   printf("[");
    //   for(int j = 0; j < 16; j++){
    //     printf( "%f,", __half2float(Btd[j+offset]));
    //   }
    //   printf("]\n");      
    //  }
    int offset1 = ((threadIdx.x % 2) ^ k) * BN;
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
    // offset1 += 1;
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
    c_tensor_s += BN*BC; // BN has nothing to do with input tiles
  }

  // c_tensor_s = threadIdx.x*BC + threadIdx.y*2;
  // for(int k=0; k<2; k++){ // prefetch 2 filter tiles of 1 channel/thread
  //   for(int i=0; i<4; i++){
  //     #pragma unroll
  //     for(int j=0; j<4; j++){  
  //       pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s + 1] = tiles[32 + k*16 + i*4 + j]; //32 = 2*BC
  //     }
  //   }
  //   // 2nd tile right behind the 1st?
  //   c_tensor_s += BN*BC; // BN has nothing to do with input tiles
  // }
  
}

__device__ __forceinline__ void prefetch_filter_tile(const half *pInputs, half *tiles, int filt_k){

  int c_offset = (filt_k<<4);
  int c_tensor = blockIdx.z*BK + threadIdx.y*2*c_offset + threadIdx.x; // Iny*filt_k*4*4

  // each threadIdx.y corresponds to 2 channels; there are 8 different threadIdx.y so 16 channels 
  
  //each thread (32 threads in x direction) loads 4 kernel tiles (2 for each channel and 32 in K direction apart)
  
  int acumm, x, x1;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = i*(filt_k<<2);
      #pragma unroll
      for(int j=0; j<4; j++){
          x = (i<<2) + j;
          x1 = acumm + j*filt_k + c_tensor;
          tiles[x] = pInputs[x1];
          tiles[16 + x] = pInputs[x1 + BN];
          tiles[32 + x] = pInputs[x1 + c_offset];
          tiles[48 + x] = pInputs[x1 + BN + c_offset];
      }
  }
}


/*__device__ __forceinline__ void prefetch_filter_tile_async(const half *pInputs, half *smem, int filt_k, int ko){

  int c_offset = (filt_k<<4);
  int c_tensor = blockIdx.z*BK; //+ threadIdx.y*2*c_offset + threadIdx.x; // Iny*filt_k*4*4

  // each threadIdx.y corresponds to 2 channels; there are 8 different threadIdx.y so 16 channels 
  // each threadx load 16 filters in K 
  //each thread (32 threads in x direction) loads 4 kernel tiles (2 for each channel and 32 in K direction apart)
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty*32+tx;
  int cid = tid / BC; 
  tx = tx % 16;
  int eid = ( tx / 4) * (filt_k<<2) + (tx % 4) * filt_k;  
  
  for(int k = 0; k < 2; k++){ // each cp.async can load 16 bytes = 8 halfs, we need to load 16 halfs

    void *ptr = (void *)(smem + tx*(BC*16) + cid * 16 + k * 8);
    unsigned int smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                "l"(&pInputs[c_tensor + cid * c_offset + eid + k * 8 + ko * BC]),
                "n"(16));
  }
}*/


// smem layout for transformed filter weights 
// ___________16K(C0)______16K(C1)____... _____16K(C15) E0
// ___________16K(C0)______16K(C1)____... _____16K(C15) E1
// .....
// .....
// ___________16K(C0)______16K(C1)____... _____16K(C15) E15
// -- B_Frag1

// ___________16K(C0)______16K(C1)____... _____16K(C15) E0
// ___________16K(C0)______16K(C1)____... _____16K(C15) E1
// .....
// .....
// ___________16K(C0)______16K(C1)____... _____16K(C15) E15
// -- B_Frag2

// ___________16K(C0)______16K(C1)____... _____16K(C15) E0
// ___________16K(C0)______16K(C1)____... _____16K(C15) E1
// .....
// .....
// ___________16K(C0)______16K(C1)____... _____16K(C15) E15
// -- B_Frag3

// ___________16K(C0)______16K(C1)____... _____16K(C15) E0
// ___________16K(C0)______16K(C1)____... _____16K(C15) E1
// .....
// .....
// ___________16K(C0)______16K(C1)____... _____16K(C15) E15
// -- B_Frag4


__device__ __forceinline__ void prefetch_filter_tile_async(const half *pInputs, half *smem, int filt_k, int ko){

  int c_offset = (filt_k<<4);
  int c_tensor = blockIdx.z*BK; //+ threadIdx.y*2*c_offset + threadIdx.x; // Iny*filt_k*4*4

  // each threadIdx.y corresponds to 2 channels; there are 8 different threadIdx.y so 16 channels 
  // each threadx load 16 filters in K 
  //each thread (32 threads in x direction) loads 4 kernel tiles (2 for each channel and 32 in K direction apart)
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty*32+tx;
  int cid = (tid % 128) / 8;   
  int kid = tx % 8;
  // tx = tx % 16;
  // int eid = (tx / 4) * (filt_k<<2) + (tx % 4) * filt_k;  
  
  for(int k = 0; k < 8; k++){ // each cp.async can load 16 bytes = 8 halfs, we need to load 16 halfs

    void *ptr = (void *)(smem + k*2*(BC*16) + (ty/4)*(BC*16) + cid * 16 + kid*2);
    unsigned int smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                "l"(&pInputs[c_tensor + cid * c_offset + (k/2)*(filt_k<<2) + (k%2==0 ? (ty/4)*filt_k : (2+(ty/4))*filt_k) + kid*2 + ko * BC]),
                "n"(4));
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
        x = (i<<2) + j;
        tile[x] = pInputs[acumm + j + c_tensor]; //1st channel
        tile[x + 16] = pInputs[acumm + j + c_tensor + c_offset];//2nd channel
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 31 && threadIdx.y == 0){
        //      printf("A, %d, %d, %d, %f, %d\n", i, j, acumm+j, tile[(i<<2) + j],acumm + j + c_tensor);   
        // }
      }
    }

  } else {
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      // #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[x] = 0.f;
        tile[x+16] = 0.f;
        if(mask&(1<<x)){
          tile[x]=pInputs[acumm + j + c_tensor];
          tile[x+16]=pInputs[acumm + j + c_tensor + c_offset];
        }
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 0 && threadIdx.y == 0){
        //      printf("B, %d, %d, %d, %d, %d, %d, %hu, %s, %f, %f, %d, %d\n", i, j, x, acumm+j, c_tensor, c_offset, mask, mask&(1<<x)?"t":"f",
        //       __half2float(tile[x]), __half2float(tile[x+16]), acumm + j + c_tensor, acumm + j + c_tensor + c_offset);   
        // }        
      }
    }
  }
}


__device__ void loadFragA(unsigned int *frag, half *smem, int ki)
{
    // load 32x16    
    // we use mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 to do 16x16x16 mm,
    // so we need to fill 2 16x8 A matrices;
    // for each 16x8 matrix A, each thread loads 4 elements (a0, a1, a2, a3) and they are
    // row 0, col 0,1 and row 8 col 0,1
    // so from the point of view the 16x16 matrix, all 8 elemets for thread 0 are
    // row/tile 0, col/channel (0, 1, 8, 9) and row/tile 8, col/chennel (0, 1, 8, 9)
    // to avoid bank conflicts, we can make threads in a warp coordinate the loading by using 
    // specially designed offsets
    // T0, T4, T8,..T28 all 8 threads load the same channels (0 and 1) and successive super tiles,
    // which results in bank conflicts. We let them load in an interleaving way:
    // first (0, 1, 0, 1, 0, 1, 0, 1)
    // then  (1, 0, 1, 0, 1, 0, 1, 0)
    // so in each round, successive threads load different channels and avoid conflicts
    // similarly for the otehr 24 threads
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    half *fragA = (half *)frag;
    for (int i = 0; i < 2; ++i){        
      for (int k = 0; k < 2; ++k){      
        //                      | tile element  |   |   channel          |  |     super tile      |
        fragA[i*8+k*4+0] = smem[(ki*8+ty)*(BN*BC) + BN*access_s[0][tx]     + tx / 4 + k * 8 + i*16];
        fragA[i*8+k*4+1] = smem[(ki*8+ty)*(BN*BC) + BN*access_s[1][tx]     + tx / 4 + k * 8 + i*16];
        fragA[i*8+k*4+2] = smem[(ki*8+ty)*(BN*BC) + BN*(access_s[0][tx]+8) + tx / 4 + k * 8 + i*16];
        fragA[i*8+k*4+3] = smem[(ki*8+ty)*(BN*BC) + BN*(access_s[1][tx]+8) + tx / 4 + k * 8 + i*16];
      }      
    }
}


__device__ void loadFragB(unsigned int *frag, half *smem, int ki)
{
    // // load 16x16    
    // // for (int i = 0; i < 2; ++i)
    // // {      
    //   nvcuda::wmma::load_matrix_sync(frag, smem + threadIdx.y*(wmmaN*wmmaK)+ ki * 8 *(wmmaN*wmmaK) , 16);
    // // }
    // load 16x16    
    // we use mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 to do 16x16x16 mm,
    // so we need to fill 4 8x8 B matrices;
    // for each 8x8 matrix B, thread 0 loads 2 elements (a0, a1) and they are
    // row 0,1 col 0
    // so from the point of view of the 16x16 matrix, all 8 elements for thread 0 are
    // row/channel (0, 1) col/K 0 , row/channel (8, 9), col/K 0
    // row/channel (0, 1) col/K 8 , row/channel (8, 9), col/K 8
    // to avoid bank conflicts, we can make threads in a warp coordinate the loading by using 
    // specially designed offsets
    // T0, T4, T8,..T28 all 8 threads load the same channels (0 and 1) and successive filters in K,
    // which results in bank conflicts. We let them load in an interleaving way:
    // first (0, 1, 0, 1, 0, 1, 0, 1)
    // then  (1, 0, 1, 0, 1, 0, 1, 0)
    // so in each round, successive threads load different channels and avoid conflicts
    // similarly for the other 24 threads
    // note the code is very similar to loadFragA
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    half *fragB = (half *)frag;
    for (int k = 0; k < 2; ++k){
      //                  | tile element  |   |   channel          |  |       K      |
      fragA[k*4+0] = smem[(ki*8+ty)*(BC*BC) + BC*access_s[0][tx]     + tx / 4 + k * 8];
      fragA[k*4+1] = smem[(ki*8+ty)*(BC*BC) + BC*access_s[1][tx]     + tx / 4 + k * 8];
      fragA[k*4+2] = smem[(ki*8+ty)*(BC*BC) + BC*(access_s[0][tx]+8) + tx / 4 + k * 8];
      fragA[k*4+3] = smem[(ki*8+ty)*(BC*BC) + BC*(access_s[1][tx]+8) + tx / 4 + k * 8];
    }
}


// Fragments layouts for A and B
// used by mmaSync
// each mma.sync.aligned.m16n8k8 takes 2 FragA and 1 FragB
//                FragA
//   ______________________________
//  |              |               |
//  |      0       |        1      |
//  |              |               |
//  |______________|_______________|
//  |              |               |
//  |              |               |
//  |      2       |        3      |
//  |              |               |
//  |______________|_______________|


//                FragB
//   ______________________________
//  |              |               |
//  |      0       |        2      |
//  |              |               |
//  |______________|_______________|
//  |              |               |
//  |              |               |
//  |      1       |        3      |
//  |              |               |
//  |______________|_______________|


__device__ void mmaSync(unsigned int *fragA, unsigned int *fragB, float *accum)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[0]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[1]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[2]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[3]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));
}



__global__ void Winograd_kernel(half *A, half *B, float *C,
                    int tiles_dim_w, int tiles_dim_h,
                    int in_c, int in_h, int in_w,
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int out_c,
                    int tile_2d_s, int out_h, int out_w){

  extern __shared__ unsigned char shared_mem[];
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
  // half filter_tile[64]; // Prefetch filter from GMEM

  // float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  // float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  // float4 accumulator[2][16] = {0.0f};  // Accumulators 

  half *A_frag; // Input data pointer

  // half *B_frag; // Filter data pointer  
  half *B_frag1 =  filter_smem;
  half *B_frag2 =  B_frag1 + 4*BC*BK;  // 16*BC*BK/4 = 4*BC*BK
  half *B_frag3 =  B_frag2 + 4*BC*BK;
  half *B_frag4 =  B_frag3 + 4*BC*BK;



  // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragA[2 * BN / wmmaM];
  // nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB;  
  // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[2 * BN / wmmaM * BK / wmmaN];

  // for (int k = 0; k < 2; k++){
  //   for (int mii = 0; mii < BN / wmmaM; mii += 1){
  //     for (int nii = 0; nii < BK / wmmaN; nii += 1){
  //       nvcuda::wmma::fill_fragment(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + nii], 0.f);      
  //     }
  //   }
  // }

  unsigned int FragA[2 * BN / wmmaM * 4];      //  4 int32 = 8 half
  unsigned int FragB[4];      // 4 int32 = 8 half
  float Accum[2 * BN / wmmaM * BK / wmmaN * 8] = {0.0}; // [4, 2, 8]

  prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
  prefetch_filter_tile_async(B, B_frag1, filt_k, 0);  
  asm volatile("cp.async.commit_group;\n" ::);
  prefetch_filter_tile_async(B, B_frag2, filt_k, 1);  
  asm volatile("cp.async.commit_group;\n" ::);
  prefetch_filter_tile_async(B, B_frag3, filt_k, 2);  
  asm volatile("cp.async.commit_group;\n" ::);
  // int ko = 0;
  
  // prefetch_filter_tile(B, filter_tile, filt_k);

  // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1 && threadIdx.y == 0){
  //   printf("[");
  //   for(int j = 0; j < 16; j++){
  //     printf( "%f,", __half2float(img_tile[j]));
  //   }
  //   printf("]\n");
  //   printf("[");
  //   for(int j = 16; j < 16+16; j++){
  //     printf( "%f,", __half2float(img_tile[j]));
  //   }
  //   printf("]\n");
  // }

  // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1 && threadIdx.y == 0){
  //   printf("[");
  //   for(int j = 0; j < 16; j++){
  //     printf( "%f,", __half2float(filter_tile[j]));
  //   }
  //   printf("]\n");
  //   printf("[");
  //   for(int j = 16; j < 16+16; j++){
  //     printf( "%f,", __half2float(filter_tile[j]));
  //   }
  //   printf("]\n");
  // }

  


  // Mainloop - iterates over the entire K dimension - not unrolled

  // wee need to do 16-batched 32x16x64 MM, each wmma will do 16x16x16 so 
  // we need to do 16 2x4 wmmas's 
  // we allocate 2 FragA and 4 FragB and 16 Accum, then in a loop of 2 iterations 
  // reuse 2 FragA and 4 FragB
  //    
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration
  // for(int iter=0; iter<16; iter+=BC){ // Current iteration

    // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    //     printf("A %d, [", iter);
    //     for(int j = 0; j < 16; j++){
    //       printf( "%f,", img_tile[j]);
    //     }
    //     printf("]\n");
    //   }

    load_and_transform_input_tile(img_tile, input_smem);
    // load_filter_tile(filter_tile, filter_smem, filt_c, filt_k);

    __syncthreads();

    for(int k = 0; k < 2; k++){
      A_frag = input_smem  + threadIdx.y*BN*BC + k*8*BN*BC;
      // B_frag = filter_smem + threadIdx.y*BC*BK + k*8*BC*BK;
      
      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      //   // printf("A %d, %d, %f, %f, %f \n", iter, i, input_frag[1], input_frag[0], accumulator[1][0]);
      //   printf("iter, k: %d, %d \n ",iter, k);
      //   for(int j=0; j < 4; j++){
      //     printf("[");
      //     for(int i = 0; i < 256; i++){          
      //       // for(int j = 0; j < 8; j++){
      //       printf( "%.2f,", __half2float(A_frag[j*256+i]));
      //       // }
      //     }
      //     printf("]\n");
      //   }
      // }
      loadFragA(FragA + k * BN / wmmaM * 4, A_frag, k);
    }
  

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();   
    // now both input and filter tiles are in smem, we can load wmma frags and do wmma computation  
    // if(iter<(in_c-BC)){ // ???should there be a if here
      prefetch_filter_tile_async(B, B_frag4, filt_k, 3);  
      asm volatile("cp.async.commit_group;\n" ::);
    // }

    for(int k = 0; k < 2; k++){
      loadFragB(FragB, B_frag1, k);
      for(int mii = 0; mii < BN / wmmaM; mii++){
            // 16x16x16 for each wmma
             mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], FragB, &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + 0]);
          //   mmaSync(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 0],
          // FragA[k * BN / wmmaM + mii], FragB, Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 0]);
          // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0)
          //    printf("AA %d, %d, %d \n", k, mii, k *(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 0);
      }
      
      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0){
      //   printf("%d, %d, %d [", iter, k, threadIdx.x);
      //   // for(int t=0; t<Accum[0].num_elements; t++)
      //     //  printf("(%f,%f)", Accum[0].x[t],Accum[4].x[t]); 
      //   for(int t=0; t<FragB.num_elements; t++)
      //      printf("%f, ", __half2float(FragB.x[t]));    
      //     //  printf("%f", __half2float(FragA[0].x[t]), __half2float(FragA[1].x[t]));    
      //   printf("]\n");   
      // }
      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      //   // printf("A %d, %d, %f, %f, %f \n", iter, i, input_frag[1], input_frag[0], accumulator[1][0]);
      //   printf("iter, k: %d, %d \n ",iter, k);
      //   for(int j=0; j < 1; j++){
      //     printf("[");
      //     for(int i = 0; i < 256; i++){          
      //       // for(int j = 0; j < 8; j++){
      //       printf( "%.2f,", __half2float(B_frag1[j*256+i]));
      //       // }
      //     }
      //     printf("]\n");
      //   }
      // }

    }

    // __syncthreads();
    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();   
    for(int k = 0; k < 2; k++){
      loadFragB(FragB, B_frag2, k);
      for(int mii = 0; mii < BN / wmmaM; mii++){
            // 16x16x16 for each wmma
            mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], FragB, &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + 8]);
          //   nvcuda::wmma::mma_sync(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 1],
          // FragA[k * BN / wmmaM + mii], FragB, Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 1]);
      }     
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();   
    for(int k = 0; k < 2; k++){
      loadFragB(FragB, B_frag3, k);
      for(int mii = 0; mii < BN / wmmaM; mii++){     
            // 16x16x16 for each wmma
            mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], FragB, &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + 16]);
          //   nvcuda::wmma::mma_sync(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 2],
          // FragA[k * BN / wmmaM + mii], FragB, Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 2]);
      }     
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();   
    for(int k = 0; k < 2; k++){
      loadFragB(FragB, B_frag4, k);
      for(int mii = 0; mii < BN / wmmaM; mii++){
            // 16x16x16 for each wmma
            mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], FragB, &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + 24]);
          //   nvcuda::wmma::mma_sync(Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 3],
          // FragA[k * BN / wmmaM + mii], FragB, Accum[k*(BN / wmmaM * BK / wmmaN) + mii * (BK / wmmaN) + 3]);
      }     
    }
    
    A += BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
      // prefetch_filter_tile(B, filter_tile, filt_k);
      prefetch_filter_tile_async(B, B_frag1, filt_k, 0);  
      asm volatile("cp.async.commit_group;\n" ::);
      prefetch_filter_tile_async(B, B_frag2, filt_k, 1);  
      asm volatile("cp.async.commit_group;\n" ::);
      prefetch_filter_tile_async(B, B_frag3, filt_k, 2);  
      asm volatile("cp.async.commit_group;\n" ::);
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
        
  // #ifdef OPTSTS64_CMP
  smem_size = 65536; // 64 KB
  cudaFuncSetAttribute(Winograd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // #endif
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
