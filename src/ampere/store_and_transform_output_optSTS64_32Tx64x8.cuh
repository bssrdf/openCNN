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

#include "../config_32Tx64x8.hpp"

#ifndef _OUTPUT_KERNEL_OPT1_
#define _OUTPUT_KERNEL_OPT1_
extern "C"
{
    
__device__ __forceinline__ void  transform_output_tile(float *pOutputs, float2 *C_tile, float2 *At, 
    int round, int c_tensor, int c_glb_offset, int i1, int i2,
    unsigned short mask1, unsigned short mask2, int out_w)
{                     
  // c_tensor += (((round)/2)*32 + ((round)%2)*2)*c_glb_offset/2;  
  c_tensor += (((round)/2)*32 + ((round)%2)*2)*c_glb_offset;
  int x, x1;

  #pragma unroll
  for(int j=0; j<4; j++){

    At[j].x = C_tile[j].x + C_tile[4+j].x + C_tile[8+j].x;
    At[j].y = C_tile[j].y + C_tile[4+j].y + C_tile[8+j].y;

    At[4+j].x = C_tile[4+j].x - C_tile[8+j].x - C_tile[12+j].x;
    At[4+j].y = C_tile[4+j].y - C_tile[8+j].y - C_tile[12+j].y;

    // if(At[j].x < 0.f)
    //     printf(" A, (%d, %d,  %d), (%d, %d), %d, %f, %f, %f, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, j, At[j].x, C_tile[j].x, C_tile[4+j].x,  C_tile[8+j].x);
    // if(At[j].y < 0.f)
    //     printf(" B, (%d, %d,  %d), (%d, %d), %d, %f, %f, %f, %f\n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, j, At[j].y, C_tile[j].y, C_tile[4+j].y,  C_tile[8+j].y);
    // if(At[4+j].x < 0.f)
    //     printf(" C, (%d, %d,  %d), (%d, %d), %d, %f, %f, %f, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, j, At[4+j].x, C_tile[4+j].x, C_tile[8+j].x,  C_tile[12+j].x);
    // if(At[4+j].y < 0.f)
    //     printf(" D, (%d, %d,  %d), (%d, %d), %d, %f, %f, %f, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, j, At[4+j].y, C_tile[4+j].y, C_tile[8+j].y,  C_tile[12+j].y);
  }

  // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 0){
    //   printf("round, %d, [", round);
    //   for(int i = 0; i < 8; ++i)
    //     printf(" (%f, %f) ", At[i].x, At[i].y );
    //   printf("]\n");   
    // }


  #pragma unroll
  for(int i=0; i<2; i++){
    x = i*4;
    // x1 = i*((tiles_dim-(out_w%2)) + (out_w%2)/2);
    x1 = i*((out_w-(out_w%2)) + (out_w%2)/2);
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 0){
    //   printf("round, %d, %d, %d, %d, %d\n", round, i, x1, c_tensor, x1+c_tensor);
    // }

    if(mask1&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i1] = At[x].x + At[x+1].x + At[x+2].x;
    }
    if(mask2&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i2] = At[x].y + At[x+1].y + At[x+2].y;

      // if(pOutputs[x1 + c_tensor].x < 0.f)
      //   printf(" A, (%d, %d,  %d), (%d, %d), %d, %d, %f, %f, %f, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //        threadIdx.x, threadIdx.y, i, x, pOutputs[x1 + c_tensor].x, At[x].x, At[x+1].x, At[x+2].x);
      // if(pOutputs[x1 + c_tensor].y < 0.f)
      //   printf(" B, (%d, %d,  %d), (%d, %d), %d, %d, %f, %f, %f, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
      //        threadIdx.x, threadIdx.y, i, x, pOutputs[x1 + c_tensor].y, At[x].y, At[x+1].y, At[x+2].y);
    }
    if(mask1&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i1 + 1] = At[x+1].x - At[x+2].x - At[x+3].x;
    }
    if(mask2&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i2 + 1] = At[x+1].y - At[x+2].y - At[x+3].y;
    }
  } 
}

__device__ __forceinline__ unsigned short get_mask(int idd, int tiles_dim_w, int tiles_dim_h, 
         int tw, int th, int out_w, int out_h){

  unsigned short mask = 0x000F;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003; // pad bottom row
  // if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005; // pad right col
  // if(blockIdx.y==gridDim.y-1 && (idd / tw) == th-1 && out_h%2)  mask&=0x0003; // pad bottom row
  // if(blockIdx.x==gridDim.x-1 && (idd % tw) == tw-1 && out_w%2)  mask&=0X0005; // pad right col
  if(tiles_dim_w % tw == 0 && tiles_dim_h % th == 0){
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == th-1 && out_h%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == tw-1 && out_w%2)  mask&=0X0005; // pad right col
  }else if(tiles_dim_w % tw == 0){
    int k = out_h % TH;
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == k1-1 && k%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > k1-1) mask &= 0x0; //pad all zeros since this tile does not exist
  }else if(tiles_dim_h % th == 0){
    int k = out_w % TW;
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == k1-1 && k%2)  mask&=0X0005; // pad right col
    if(blockIdx.x==gridDim.x-1 && (idd % tw) > k1-1)  mask&=0X0; // pad all zeroes
  }else{
    int kh = out_h % TH;
    int kw = out_w % TW;
    int kh1 =  kh % 2 ? (kh+1)/2 : kh/2; // there could be kh1*kw1 tiles
    int kw1 =  kw % 2 ? (kw+1)/2 : kw/2;
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == kh1-1 && kh%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == kw1-1 && kw%2)  mask&=0X0005; // pad right col
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > kh1-1)  mask &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && (idd % tw) > kw1-1)  mask &= 0X0; // pad all zeroes
  }
  return mask;
}

__device__ __forceinline__ void store_output_tile(float4 acumm_smem[][16], float *shared_mem, float *C, 
int out_h, int out_w, int tiles_dim_w, int tiles_dim_h,  int tw, int th, 
float4 *input_frag_mem, float4* filter_frag_mem){
  
  float2 *output_smem = (float2 *) shared_mem;
  float2 *accumulator = (float2 *) acumm_smem;
  float2 *C_out = (float2*)C;

  float2 *C_tile = (float2*) input_frag_mem;
  float2 *At = (float2*) filter_frag_mem;

  int idd1 = tileid[0][threadIdx.x];
  int id1 = (idd1 % tw) * 2 + (idd1 / tw) * out_w * 2;
  int idd2 = tileid[1][threadIdx.x];
  int id2 = (idd2 % tw) * 2 + (idd2 / tw) * out_w * 2;

  // unsigned short mask1 = 0x000F;
  unsigned short mask1 = get_mask(idd1, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  unsigned short mask2 = get_mask(idd2, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  
  // output transpose step
  int t=0;
  int acumm1, acumm2;
  // For transposing
  //acumm1 = access_s_out[Inx]; //* 4
  acumm1 = ((threadIdx.x%8)/2)*34 + threadIdx.x%2 + (threadIdx.x/16)*2 + ((threadIdx.x/8)%2)*8;
  acumm2 = acumm1+4;
                       
  int acumm4 = BN_p*16 ; //*4
  int idx  = threadIdx.y * BN_p;
  int idx2 = idx + BN_p*8; //(BN_p*2 *8)/2

  // For transformating
  int offset = BN_p *2; //*2/2
  int init = ( (threadIdx.y/4)*BN_p*16 + (threadIdx.y%4)*(32+2) ) *2 + threadIdx.x;

  int c_glb_offset = out_h*out_w;
  // int c_tensor = blockIdx.z*c_glb_offset*BK + (blockIdx.y%tiles_dim)*2 + (blockIdx.y/tiles_dim)*out_w*2 + 
  //               blockIdx.x*BN + (threadIdx.x%16)*2+
  //               ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;

  // int tx = out_w / gridDim.x, ty = out_h / gridDim.y;  
  int tx = TW, ty = TH;  
  // int c_tile = blockIdx.x * tx  + blockIdx.y * in_w * ty; 
  // int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
  //               threadIdx.y*(in_h*in_w) - (in_w+1);

  int c_tensor = blockIdx.z*c_glb_offset*BK + blockIdx.x * tx  + blockIdx.y * out_w * ty +
                //  (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * out_w * 2 + 
                 ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;

  // c_tensor/=2; 

  // for(int i = 0;i < 2; i++){
  //     for(int j = 0; j < 16; i++){
  //     if(acumm_smem[i][j].x < 0.f)
  //       printf(" X, (%d, %d,  %d), (%d, %d), %d, %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, j, acumm_smem[i][j].x);
  //     if(acumm_smem[i][j].y < 0.f)
  //       printf(" Y, (%d, %d,  %d), (%d, %d), %d, %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, j, acumm_smem[i][j].y);
  //     if(acumm_smem[i][j].z < 0.f)
  //       printf(" Z, (%d, %d,  %d), (%d, %d), %d, %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, j, acumm_smem[i][j].z);  
  //     if(acumm_smem[i][j].w < 0.f)
  //       printf(" W, (%d, %d,  %d), (%d, %d), %d, %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, j, acumm_smem[i][j].w);
  //     }
  // }    


  int target = 16;  

  #pragma unroll                                  
  for(int round=0; round<4; round++){

    *( (float2*) (output_smem + idx + acumm1) )  = *(accumulator+t);
    *( (float2*) (output_smem + idx + acumm1 + 16) )  = *(accumulator+t+1); // float 4, t
    *( (float2*) (output_smem + idx + acumm2) )  = *(accumulator+t+2);
    *( (float2*) (output_smem + idx + acumm2 + 16) )  = *(accumulator+t+3); // float 4, t+1

    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && idx + acumm1 == target){
    //   printf("A. (%d, %d, %d, %d, %d) %d, %d\n ",blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, idx, acumm1);
    // }
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && idx + acumm1 + 16 == target){
    //   printf("B. (%d, %d, %d, %d, %d) %d, %d\n ",blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, idx, acumm1);
    // }
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && idx + acumm2 == target){
    //   printf("C. (%d, %d, %d, %d, %d) %d, %d, %d\n ",blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, idx, acumm1, acumm2);
    // }
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && idx + acumm2 + 16 == target){
    //   printf("D. (%d, %d, %d, %d, %d) %d, %d, %d\n ",blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, idx, acumm1, acumm2);
    // }

    *( (float2*) (output_smem + idx2 + acumm1) ) = *(accumulator+t+32);
    *( (float2*) (output_smem + idx2 + acumm1 + 16) ) = *(accumulator+t+33); // float 4, t+16
    *( (float2*) (output_smem + idx2 + acumm2) ) = *(accumulator+t+34);
    *( (float2*) (output_smem + idx2 + acumm2 + 16) ) = *(accumulator+t+35); // float 4, t+17

    // the above 8 float2 will be consumed by theadIdx.y = [0,1,2,3]

    // the following 8 float2 will be consumed by theadIdx.y = [4,5,6,7]

    *( (float2*) (output_smem + idx + acumm4 + acumm1) )  = *(accumulator+t+4); 
    *( (float2*) (output_smem + idx + acumm4 + acumm1 + 16) )  = *(accumulator+t+5); // float 4, t+2
    *( (float2*) (output_smem + idx + acumm4 + acumm2) )  = *(accumulator+t+6);
    *( (float2*) (output_smem + idx + acumm4 + acumm2 + 16) )  = *(accumulator+t+7); // float 4, t+3

    *( (float2*) (output_smem + idx2 + acumm4 + acumm1) ) = *(accumulator+t+36);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm1 + 16) ) = *(accumulator+t+37); // float 4, t+18
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2) ) = *(accumulator+t+38);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2 + 16) ) = *(accumulator+t+39); // float 4, t+19
    
    

    t+=8;

    __syncthreads();

  // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 0){
  //   printf("round, %d, [", round);
  //   for(int i = 0; i < 16; ++i)
  //     printf(" %f, ", shared_mem[i]);
  //   printf("]\n");   
  // }

    // for output transformation, the role of threadIdx.y changes again:
    // in the main loop, different threadIdx.y deal with different element of the 4x4 tile 
    // here, they are for 4 different groups of lane ids from optSTS64 layout
    // for init (and init+32), we need to identify its tile number (0-31) within the supertile     
    // first, from init, find out from which threadIdx.x it comes.
    // int idy = init - threadIdx.x;
    // if(idy > 204) idy -= BN_p*16*2; 
    // int idx = idy + threadIdx.x;
    // if(idx % 2 == 0)
    //     idx = idx / 2;
    // else
    //     idx = (idx-1) / 2;
    // int l = laneid[idx];
    // now we got l, which is the land id which computed accumulated sum for the tile element 
    // each lane id (or threadIdx.x) computed 8 tiles which are distributed into 4 locations spreading
    // over the smem. We need to find which of the 8 the current tile is.   
    // use tileid table to figure out
    // int id1 = tileid[0][l];


    // for 2nd tile
    // idx = idy + threadIdx.x + 32;
    // if(idx % 2 == 0)
    //     idx = idx / 2;
    // else
    //     idx = (idx-1) / 2;
    // l = laneid[idx];
    // int id2 = tileid[1][l];


    // int tx = 0, ty=1; 
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == tx  && threadIdx.y == ty)      
    //   printf("round, %d, [", round);
    for(int i=0; i<16; i++){
      C_tile[i].x = shared_mem[i*offset + init];
      C_tile[i].y = shared_mem[i*offset + init + 32];
      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == tx  && threadIdx.y == ty){
      //   printf("%d,", i*offset + init);
      // }
    }
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == tx  && threadIdx.y == ty)      
    //   printf("]\n");   

    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 0){
    //   printf("round, %d, [", round);
    //   for(int i = 0; i < 16; ++i)
    //     printf(" (%f, %f) ", C_tile[i].x, C_tile[i].y);
    //   printf("]\n");   
    // }

    // transform output tiles
    // transform_output_tile(C, C_tile, At, tiles_dim, round, c_tensor, c_glb_offset, id1, id2, mask, out_w);
    transform_output_tile(C, C_tile, At, round, c_tensor, c_glb_offset, id1, id2, mask1, mask2, out_w);
    __syncthreads();
  }
}

}
#endif     
