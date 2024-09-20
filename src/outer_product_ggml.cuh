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


__device__  __forceinline__ void outer_product(float input[], float4* filter_frag, float accumulator[][8]){

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
    accumulator[1][0] += input[1]*filter_frag[2].x;
    accumulator[1][1] += input[1]*filter_frag[2].y;
    accumulator[1][2] += input[1]*filter_frag[2].z;
    accumulator[1][3] += input[1]*filter_frag[2].w;
    
    //
    accumulator[1][4] += input[1]*filter_frag[3].x;
    accumulator[1][5] += input[1]*filter_frag[3].y;
    accumulator[1][6] += input[1]*filter_frag[3].z;
    accumulator[1][7] += input[1]*filter_frag[3].w;
                                        
}