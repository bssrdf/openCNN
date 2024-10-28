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


#ifndef COMMON_INCLUDE_FILE
#define COMMON_INCLUDE_FILE

#define BC 16
#define BN 32
#define BK 64
///////////////////// For Non-Fused version
#define BC_GEMM 8
#define BN_GEMM 128
#define BK_GEMM 128
///////////////////// For Non-Fused version

#define wmmaM  16
#define wmmaN  16
#define wmmaK  16

#define TW 8
#define TH 16

#ifdef OPTSTS64_CMP
#define BN_p 128
#elif BASE
#define BN_p 40
#else 
#define BN_p 138
#endif

#define N 128 // values: 32,64,96,128
#define C_in 256 // values: 64,128,256,512
#define W 14 // values: 56,28,14,7

#define K 256 // values: 64,128,256,512
#define R 3 // values: 3

#define PADDING 1

#define PAD_H 1
#define PAD_W 1
#define STR_H 1
#define STR_W 1
#define DIL_H 1
#define DIL_W 1

#define M 2 // values: 2

__constant__ int access_f_s[2][32];
__constant__ int access_f_f[2][32];
__constant__ int access_s[2][32];
__constant__ int access_t[2][32];
__constant__ int access_o[2][2];
__constant__ int access_p[2];
// __constant__ int laneid[138];
// __constant__ int tileid[2][32];

// access_f_s
const int aux[2][32] = {
                        {0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7},
                        {1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6}
                        };
// access_f_f
const int aux1[2][32] = {
                        { 0, 0, 0, 0, 1, 1, 1, 1,
                          2, 2, 2, 2, 3, 3, 3, 3,
                          4, 4, 4, 4, 5, 5, 5, 5,
                          6, 6, 6, 6, 7, 7, 7, 7},                         
                        { 8, 8, 8, 8, 9, 9, 9, 9,
                         10,10,10,10,11,11,11,11,
                         12,12,12,12,13,13,13,13,
                         14,14,14,14,15,15,15,15}                          
                        };
// access_s
const int aux2[2][32] = {
                         {0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7},                         
                         {1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6}
                        };        
// access_t
const int aux3[2][32] = {
                         {0,2,4,6,0,2,4,6,
                          0,2,4,6,0,2,4,6,
                          8,10,12,14,8,10,12,14,
                          8,10,12,14,8,10,12,14},                          
                         {8,10,12,14,8,10,12,14,
                          8,10,12,14,8,10,12,14,
                          0,2,4,6,0,2,4,6,
                          0,2,4,6,0,2,4,6}                          
                        };       
const int aux_offset[2][2] = {{0, 4}, {16, 20}}; 
const int aux_offset1[2] = {0, BN_p}; 

const int lid[138] = {0, 1, 16, 17, 0, 1, 16, 17, 8, 9, 24, 25, 8, 9, 24, 25, 0, 1, 
16, 17, 0, 1, 16, 17, 8, 9, 24, 25, 8, 9, 24, 25, -1, -1, 2, 3, 18, 19, 2, 3, 18, 19, 
10, 11, 26, 27, 10, 11, 26, 27, 2, 3, 18, 19, 2, 3, 18, 19, 10, 11, 26, 27, 10, 11, 
26, 27, -1, -1, 4, 5, 20, 21, 4, 5, 20, 21, 12, 13, 28, 29, 12, 13, 28, 29, 4, 5, 20, 21, 
4, 5, 20, 21, 12, 13, 28, 29, 12, 13, 28, 29, -1, -1, 6, 7, 22, 23, 6, 7, 22, 23, 14, 15, 
30, 31, 14, 15, 30, 31, 6, 7, 22, 23, 6, 7, 22, 23, 14, 15, 30, 31, 14, 15, 30, 31};                               

// const int tid[2][32] = {
//                         {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
//                          0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15},
//                         {16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
//                          16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31}  
//                         };  
                               
const int tid[2][32] = {
                        {0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,
                         0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29},
                        {2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,
                         2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31}
                        };  
                               
#endif
