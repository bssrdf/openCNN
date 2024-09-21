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

#include "../config_ggml.hpp"

#ifndef _OUTPUT_KERNEL_OPT1_
#define _OUTPUT_KERNEL_OPT1_
extern "C"
{
    
// we need to gather GEMM results saved in accumulator to smem,
// transform them, then output to gmem

// there are only 4KB (16 element  * 64 in K) to be transformed, so smem is enough (no need to do it in 4 rounds)
// each thread row handles two elements (there are 8 rows, so 16 elements) 
// threadIdx.y        elements 
//    0                0, 2
//    1                1, 3 
//    2                4, 6 
//    3                5, 7 
//    4                8, 10
//    5                9, 11
//    6                12, 14
//    7                13, 15 
// within each row, each thread's accumulator has 2*8 values, 
// the 1st 8 for first element, the last 8 for 2nd element
// 0th thread: 0-3, 32-35        same for 2nd element
// 1th thread: 0-3, 32-35
// 2nd thread: 4-7, 36-39
// 3rd thread: 4-7, 36-39
// ...
// 14th  thread: 28-31, 60-63
// 15th  thread: 28-31, 60-63

// 16-31 repeat last 16

// so we only need odd or even threadIdx.x total up to 8 to retrieve all 64 values




__device__ __forceinline__ void store_output_tile(float acumm_smem[][8], float *shared_mem, float *C, 
 int out_h, int out_w, int tiles_dim, float* input_frag_mem, float4* filter_frag_mem,  short mask){
  


  float *output_smem = shared_mem;
  float *accumulator = (float *)acumm_smem;
  float *C_out = C;

  float At[8];
  int x, x1;  

  mask = 0x000F;
  if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003;
  if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005;
  
  // output transpose step
  int t=0;
  int acumm1, acumm2;
  // For transposing
  //acumm1 = access_s_out[Inx]; //* 4
  // acumm1 = ((threadIdx.x%8)/2)*34 + threadIdx.x%2 + (threadIdx.x/16)*2 + ((threadIdx.x/8)%2)*8;
  // acumm2 = acumm1+4;
                       
  int acumm4 = BN*16 ; //*4

  // int idx  = threadIdx.y * BN_p;
  // int idx2 = idx + BN_p*8; //(BN_p*2 *8)/2

  // For transformating
  // int offset = BN_p *2; //*2/2
  // int init = ( (threadIdx.y/4)*BN_p*16 + (threadIdx.y%4)*(32+2) ) *2 + threadIdx.x;

  int c_glb_offset = out_h*out_w;
  // int c_tensor = blockIdx.z*c_glb_offset*BK + (blockIdx.y%tiles_dim)*2 + 
  //               (blockIdx.y/tiles_dim)*out_w*2 + blockIdx.x*BN + (threadIdx.x%16)*2+
  //               ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;
  // c_tensor/=2; 

  // for(int i = 0;i < 2; i++){
  //    for(int j = 0; j < 8; j++){
  //     if(acumm_smem[i][j] < 0.f)
  //       printf(" (%d, %d,  %d), (%d, %d), %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, acumm_smem[i][j]);
  //    }
  // }

  int idx = threadIdx.y % 2 ? threadIdx.y * 2 - 1 : threadIdx.y * 2; 
  int idx1 = idx + 2;
  acumm1 =  threadIdx.x / 2;
  // use only the first 8 even threads
  if(threadIdx.x % 2 == 0 && threadIdx.x < 16){
    for(int i = 0;  i < 4; i++){
      output_smem[(4*acumm1 + i)*16 + idx      ] = acumm_smem[0][i];
      output_smem[(4*acumm1 + i)*16 + idx  + acumm4] = acumm_smem[0][i+4];
      output_smem[(4*acumm1 + i)*16 + idx1     ] = acumm_smem[1][i]; 
      output_smem[(4*acumm1 + i)*16 + idx1 + acumm4] = acumm_smem[1][i+4];
      // if(blockIdx.y == 0 && blockIdx.z == 0 && (4*acumm1 + i)*16 + idx == 2)
      //   printf("A (%d, %d), %d, %f \n",
      //        threadIdx.x, threadIdx.y, i, output_smem[(4*acumm1 + i)*16 + idx      ]);
      // if(blockIdx.y == 0 && blockIdx.z == 0 && (4*acumm1 + i)*16 + idx1 == 2)
      //   printf("B (%d, %d), %d, %f \n",
      //        threadIdx.x, threadIdx.y, i, output_smem[(4*acumm1 + i)*16 + idx1      ]);
      
    }
  }  
  __syncthreads();

  if( blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 0){
    for(int i = 0; i < BC; ++i){
      printf("%d, [", i);    
      for(int j = 0; j < 16; ++j)
        printf(" %f, ", output_smem[i*16+j]);
      printf("]\n");   
    }
  }
   
  // now smem contains all 64 4x4 tiles with elements of each tile contiguous 


   // At transform
  idx = threadIdx.x % 8; 
  idx1 = (idx+threadIdx.y*8)*16;
  // #pragma unroll
  // for(int j=0; j<2; j++){
  //   for(int i=0; i<4; j++){     
  At[0] =  output_smem[idx1+0]+output_smem[idx1+4]+output_smem[idx1+8];
  At[1] =  output_smem[idx1+1]+output_smem[idx1+5]+output_smem[idx1+9];
  At[2] =  output_smem[idx1+2]+output_smem[idx1+6]+output_smem[idx1+10];
  At[3] =  output_smem[idx1+3]+output_smem[idx1+7]+output_smem[idx1+11];
  At[4] =  output_smem[idx1+4]-output_smem[idx1+8]-output_smem[idx1+12];
  At[5] =  output_smem[idx1+5]-output_smem[idx1+9]-output_smem[idx1+13];
  At[6] =  output_smem[idx1+6]-output_smem[idx1+10]-output_smem[idx1+14];
  At[7] =  output_smem[idx1+7]-output_smem[idx1+11]-output_smem[idx1+15];
  //   }
  // }

  int c_tensor = blockIdx.z*c_glb_offset*BK + (blockIdx.y%tiles_dim) + 
                (blockIdx.y/tiles_dim)*out_w + (idx+threadIdx.y*8)*c_glb_offset;
  
  if(threadIdx.x < 8){
    #pragma unroll
    for(int i=0; i<2; i++){
      x = i*4;
      x1 = i*((tiles_dim-(out_w%2)) + (out_w%2)/2);
      if(mask&(1<<(i*2))){
        C[x1 + c_tensor] = At[x] + At[x+1] + At[x+2];      
        if(x1+c_tensor == 1)
          printf("X (%d, %d,  %d), (%d, %d), %d, %d, %d, %f, %f, %f, %f\n", blockIdx.x, blockIdx.y, blockIdx.z, 
              threadIdx.x, threadIdx.y, i, x, x1, C[x1 + c_tensor], At[x],  At[x+1], At[x+2]);
      }

      if(mask&(1<<(i*2+1))){
        C[x1 + c_tensor + 1] = At[x+1] - At[x+2] - At[x+3];    
        if(x1+c_tensor + 1 == 1)
          printf("Y (%d, %d,  %d), (%d, %d), %d, %d, %d, %f, %f, %f, %f\n", blockIdx.x, blockIdx.y, blockIdx.z, 
              threadIdx.x, threadIdx.y, i, x, x1, C[x1 + c_tensor], At[x],  At[x+1], At[x+2]);
      }
    } 
  }
  __syncthreads();
  
}

}
#endif     
