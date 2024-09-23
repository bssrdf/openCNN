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


__device__  __forceinline__ void outer_product(float *input, float4* filter_frag, float accumulator[][8]){

    // if(input[0] < 0.f)
    //     printf(" A (%d, %d,  %d), (%d, %d), %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, input[0]);
    // if(input[1] < 0.f)
    //     printf(" B (%d, %d,  %d), (%d, %d), %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
    //          threadIdx.x, threadIdx.y, input[1]);
  
  //   for(int i = 0;i < 4; i++){
  //     if(filter_frag[i].x< 0.f)
  //       printf(" X, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, filter_frag[i].x);
  //     if(filter_frag[i].x< 0.f)
  //       printf(" Y, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, filter_frag[i].y);
  //     if(filter_frag[i].x< 0.f)
  //       printf(" Z, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, filter_frag[i].z);
  //     if(filter_frag[i].x< 0.f)
  //       printf(" W, (%d, %d,  %d), (%d, %d), %d, %f \n", blockIdx.x, blockIdx.y, blockIdx.z, 
  //            threadIdx.x, threadIdx.y, i, filter_frag[i].w);
  // }    


    accumulator[0][0] += input[0]*filter_frag[0].x;
    accumulator[0][1] += input[0]*filter_frag[0].y;
    accumulator[0][2] += input[0]*filter_frag[0].z;
    accumulator[0][3] += input[0]*filter_frag[0].w;
  
    //
    accumulator[0][4] += input[0]*filter_frag[1].x;
    accumulator[0][5] += input[0]*filter_frag[1].y;
    accumulator[0][6] += input[0]*filter_frag[1].z;
    accumulator[0][7] += input[0]*filter_frag[1].w;
  
    //////
    accumulator[1][0] += input[8]*filter_frag[2].x;
    accumulator[1][1] += input[8]*filter_frag[2].y;
    accumulator[1][2] += input[8]*filter_frag[2].z;
    accumulator[1][3] += input[8]*filter_frag[2].w;
    
    //
    accumulator[1][4] += input[8]*filter_frag[3].x;
    accumulator[1][5] += input[8]*filter_frag[3].y;
    accumulator[1][6] += input[8]*filter_frag[3].z;
    accumulator[1][7] += input[8]*filter_frag[3].w;
                                        
}