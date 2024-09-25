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


#define d(input, i, j) ( input[(i<<2) + (j)] )

__device__ __forceinline__ void load_and_transform_input_tile(float *Btd, float *pOutputs, int in_h, int in_w,
                                  int tiles_dim, int in_c, int tile_size, 
                                  int tiles_2d_dim, int tile_2d_s){

  float workspace[3]; 

  // if(threadIdx.y >= BC) return;
  
  #pragma unroll
  for(int j=0; j<4; j++){
    workspace[0] = Btd[j];
    workspace[1] = Btd[j+4];
    workspace[2] = Btd[j+8];

    Btd[j]    = workspace[0] - workspace[2];
    Btd[j+4]  = workspace[1] + workspace[2];
    Btd[j+8]  = workspace[2] - workspace[1];
    Btd[j+12] = workspace[1] - Btd[j+12];
  }
  
  int c_offset = BC*BN;
  int c_tensor = threadIdx.y*BN + threadIdx.x;
  
  #pragma unroll
  for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
    pOutputs[c_tensor+i*c_offset*4] = d(Btd, i, 0) - d(Btd, i, 2);  
    pOutputs[c_tensor+i*c_offset*4+c_offset] = d(Btd, i, 1) + d(Btd, i, 2);
    pOutputs[c_tensor+i*c_offset*4+2*c_offset] = d(Btd, i, 2) - d(Btd, i, 1);
    pOutputs[c_tensor+i*c_offset*4+3*c_offset] = d(Btd, i, 1) - d(Btd, i, 3);
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

}

__device__ __forceinline__ void load_filter_tile(float *tiles, float *pOutputs, 
                                int filt_c, int filt_k){
 
  int c_tensor_s = threadIdx.y*BK + threadIdx.x;
  int c_offset_s = BK*BC;
  // if(threadIdx.y >= BC) return;
  
  // each thread in row 0 puts its first element of 1st filter tile(loaded by the thread) in smem
  // taking 32 slots 
  // then puts its first element of 2nd filter tile immediately after, taking another 32 slots
  // then followed by threads in row 1, 2.. until 7

  // Note the next element is BK*BC (8*64) slots away, then another BK*BC ....
  // for every 64 values, the first 32 belongs to filter tile 1, the next 32 for filter tile 2 


  for(int k=0; k<2; k++){ // prefetch 2 filter tiles/thread
    for(int i=0; i<4; i++){
      #pragma unroll
      for(int j=0; j<4; j++){
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s] = tiles[k*16 + i*4 + j];
      }
    }
    // 2nd tile right behind the 1st?
    c_tensor_s += BN; // BN has nothing to do with input tiles
  }
  
}

__device__ __forceinline__ void prefetch_filter_tile(float *pInputs, float *tiles, int filt_k){

  int c_tensor = blockIdx.z*BK + (threadIdx.y*filt_k<<4) + threadIdx.x; // Iny*filt_k*4*4
  // each threadIdx.y corresponds to one channel; there are 8 different threadIdx.y so 8 channels 
  
  //each thread (32 threads in x direction) loads 2 kernel tiles (32 in K direction apart)
  // save the two tiles in a float[32] register, float[16] for each  
  
  int acumm;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = (i*filt_k<<2);
      #pragma unroll
      for(int j=0; j<4; j++){
          tiles[(i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor];
          tiles[16 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor+BN];
      }
  }
}

__device__ __forceinline__ void prefetch_input_tile(float *pInputs, float *tile, int in_h, 
                       int in_w, int tiles_dim, int tw, int th, unsigned short mask){
  
  // load one input tile
  int tx = in_w / gridDim.x, ty = in_h / gridDim.y;  
  int c_tile = blockIdx.x * tx  + blockIdx.y * in_w * ty; 
  int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
                threadIdx.y*(in_h*in_w) - (in_w+1);

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
        tile[(i<<2) + j] = pInputs[acumm + j + c_tensor];
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
        tile[x] = 0;
        if(mask&(1<<x))
          tile[x]=pInputs[acumm + j + c_tensor];
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 28 && threadIdx.y == 0){
        //      printf("B, %d, %d, %d, %d, %hu, %s, %f, %d\n", i, j, x, acumm+j, mask, mask&(1<<x)?"t":"f",tile[x],acumm + j + c_tensor);   
        // }        
      }
    }
  }
}


// this remains the same as 32x64x8 case
__device__  __forceinline__ void prefetch_filter_frag(float4 *filter_frag, float4 *B_frag, int f_frag_offset, int offset1, int offset2){

  // if(threadIdx.y >= BC) return;
  // from the land id table, 32 threads are actually divided into 2 big groups
  // first 16 and the last 16
  // each big group further divides into 8 pairs
  // threads within each pair load the same filter value   

  // the 2nd group just duplicates the 1st

  *((float4*) (filter_frag))     = *(B_frag + offset1); 
  *((float4*) (filter_frag + 1)) = *(B_frag + offset2); // + 32 floats (8 float4)

 // the next 8 floats are for the next next tile element 
  *((float4*) (filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
  *((float4*) (filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}


__device__  __forceinline__ void prefetch_input_frag(float4* input_frag, float4 *A_frag, int frag_offset, int offset1, int offset2){  

  *((float4*) (input_frag))     = *(A_frag + offset1); //ld_shared(A_frag + offset1);
  *((float4*) (input_frag + 1)) = *(A_frag + offset2);

  *((float4*) (input_frag + 2)) = *(A_frag + frag_offset + offset1);
  *((float4*) (input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}

__global__ void Winograd_kernel(float *A, float *B, float *C,
                    int tiles_dim, int in_c, int in_h, int in_w, 
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int tiles_2d_dim, int out_c, 
                    int tile_2d_s, int out_h, int out_w){

  extern __shared__ float shared_mem[];
  float *input_smem  = (float*)shared_mem;
  float *filter_smem = (float*)&shared_mem[16*BC*BN];

  unsigned short m = 0xFFFF;
  // if((blockIdx.y/tiles_dim)==0)   m&=0xFFF0;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1)) m &= (!(in_w%2))?(0x0FFF):(0x00FF);
  // if(!((blockIdx.y+1)%tiles_dim)) m &= (!(in_w%2))?(0x7777):(0x3333);
  // if(!((blockIdx.y)%tiles_dim))   m&=0xeeee;

  if(blockIdx.y==0 && (threadIdx.x / X) == 0)   m &= 0xFFF0;  // pad top row
  if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
  if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
  if(blockIdx.x == 0 && (threadIdx.x % X) == 0)   m &=0xeeee;  // pad left col
  
  



  float img_tile[16]; // Prefetch input from GMEM
  float filter_tile[32]; // Prefetch filter from GMEM

  float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 accumulator[2][16] = {0.0f};  // Accumulators 

  float4 *A_frag; // Input data pointer
  int frag_offset = 2 * (BN*BC); // (2=8/4) SMEM input read offset

  float4 *B_frag; // Filter data pointer  
  int f_frag_offset = 2 * (BC*BK); // (2=8/4 with 4 being float4) SMEM filter read offset 
        

  float4 *input_frag  = (float4*) input_frag_mem;
  float4 *filter_frag = (float4*) filter_frag_mem;

  float4 *swap_filter;
  float4 *swap_input;

  prefetch_input_tile(A, img_tile, in_h, in_w, tiles_dim, X, Y, m);
  prefetch_filter_tile(B, filter_tile, filt_k);

  // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
  //     for(int j = 0; j < 16; j++){
  //       printf( "%f,", img_tile[j]);
  //     }
  //     printf("\n");
  //   }

  float4 *input_frag_buffer  = (float4*) (input_frag+4);
  float4 *filter_frag_buffer = (float4*) (filter_frag+4);
  
  // Mainloop - iterates over the entire K dimension - not unrolled
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration

    A_frag = (float4*) (input_smem  + threadIdx.y*BN*BC);
    B_frag = (float4*) (filter_smem + threadIdx.y*BC*BK);

    // if(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    //     printf("A %d, [", iter);
    //     for(int j = 0; j < 16; j++){
    //       printf( "%f,", img_tile[j]);
    //     }
    //     printf("]\n");
    //   }

    load_and_transform_input_tile(img_tile, input_smem, in_h, in_w,
                 tiles_dim, in_c, tile_size,
                 tiles_2d_dim, tile_2d_s);
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
     

    prefetch_input_frag(input_frag, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
    prefetch_filter_frag(filter_frag, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);

    
    #pragma unroll
    for(int i=0; i<BC; i++){

      if(i<(BC-1)){
        A_frag += BN/4;     // This actually moves 32 float (A_frag is float4*)
                          // 32 float is also of size of supertile of one input channel   
        B_frag += BK/4;   // This actually moves 16*4=64 floats (B_frag is float4*), 
                          // 64 floats is also of size of one filter channel 

        prefetch_input_frag(input_frag_buffer, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
        prefetch_filter_frag(filter_frag_buffer, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
      }
     
      outer_product(input_frag, filter_frag, accumulator);

      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      //   printf("A %d, %d, %f, %f, %f \n", iter, i, input_frag[0].x, filter_frag[0].x, accumulator[0][0].x);
        // for(int j = 0; j < 16*8; j++){
        //   printf( "%f,", input_smem[i]);
        // }
        // printf("\n")
      // }

      swap_input = input_frag;
      input_frag = input_frag_buffer;
      input_frag_buffer = swap_input;

      swap_filter = filter_frag;
      filter_frag = filter_frag_buffer;
      filter_frag_buffer = swap_filter;
      
    }
    
    A += BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile(A, img_tile, in_h, in_w, tiles_dim, X, Y, m);
      prefetch_filter_tile(B, filter_tile, filt_k);
    }

    __syncthreads();
  }

  // Transpose, transform and store accumulated result
  store_output_tile(accumulator, shared_mem, C, out_h, out_w, tiles_dim, X, Y, input_frag_mem, filter_frag_mem, m);
                     
}

cudaError_t convolutionForward_32Tx64x8(float *k, int in_h, int in_w, float *w, int out_h,
                  int out_w, int out_c, float *C, float *Ww,                 
                int tiles_dim, int tile_size,
                int in_c, int filt_k, int filt_c, int filt_h, int filt_w, int alpha, int m){

  int tile_2d_s = tile_size*tile_size;
  int tiles_2d_dim = tiles_dim*tiles_dim;
  int smem_size = (16*BN*BC + 16*BC*BK)*4;
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
  Winograd_kernel<<<dim3((tiles_dim+X-1)/X, (tiles_dim+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size>>>(k, Ww, C, 
  tiles_dim, in_c, in_h, in_w, tile_size, X, Y, filt_k, filt_c, tiles_2d_dim, out_c, tile_2d_s, out_h, out_w);

  return cudaGetLastError();
}

}
#endif
