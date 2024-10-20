
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


#ifndef _FX_
#define _FX_

#include<cuda_fp16.h>

extern "C"
{

// Set of functions per row in Gw product
__device__ half f_row1(half *Gw, int j){
    return Gw[j];
  }
  __device__ half f_row2(half *Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[6+j] + Gw[3+j]);
  }
  __device__ half f_row3(half *Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[6+j] - Gw[3+j]);
  }
  __device__ half f_row4(half *Gw, int j){
    return Gw[6+j];
  }
  // Set of functions per column in GwGt product
  __device__ half f_col1(half *Gw, int j){
    return Gw[j];
  }
  __device__ half f_col2(half *Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[j+2] + Gw[j+1]);
  }
  __device__ half f_col3(half *Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[j+2] - Gw[j+1]);
  }
  __device__ half f_col4(half *Gw, int j){
    return Gw[j+2];
  }
  
  typedef half(*pointFunction_t)(half *, int);
  
  __global__ void FX(half *pInputs, half *pOutputs, int filt_k, 
                      int filt_c, int filt_h, int filt_w, int alpha){
    int Inx = threadIdx.x, Iny = threadIdx.y;
    int TileX = blockIdx.x, TileY = blockIdx.y;
  
    int c_glb_offset = filt_k*filt_h*filt_w;
    int c_kernel = TileY*BC*c_glb_offset + TileX*BK + Iny*c_glb_offset + Inx;
    // int c_glb_offset_s = filt_c*4*4;
    int c_glb_offset_s = filt_c*filt_k;
    int c_kernel_s = TileY*BC + TileX*BK*filt_c + Iny + Inx * filt_c;
  
    half Gw[21]; //9+12. In registers
    half *Gw_buffer = Gw+9;
  
    pointFunction_t func1[4] = {f_row1, f_row2, f_row3, f_row4};
    pointFunction_t func2[4] = {f_col1, f_col2, f_col3, f_col4};
  
    for(int bk=0; bk<BK; bk+=blockDim.x){
      for(int i=0; i<9; i++){
        Gw[i] = pInputs[c_kernel + i*filt_k];
      }
  
      int aux;
      for(int i=0; i<4; i++){
        aux = i*3;
        for(int j=0; j<3; j++){
          Gw_buffer[j+aux] = (*func1[i])(Gw, j);
        }
      }
  
      int aux2;
      for(int i=0; i<4; i++){
        aux = i*3; aux2 = i<<2;
        for(int j=0; j<4; j++){
          pOutputs[c_kernel_s+aux2*c_glb_offset_s+j*c_glb_offset_s] = (*func2[j])(Gw_buffer, aux);
        }
      }
  
      c_kernel   += blockDim.x;
      c_kernel_s += blockDim.x*filt_c;
    }
  }

}
#endif