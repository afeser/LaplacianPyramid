
/*
 * CUDA Basics page 51
 *
 * CALISTIRAMADIM UGRASMIYORUM DA
 */
#define BLOCK_SIZE 1
#define RADIUS 3

#include <stdio.h>

 __global__ void stencil_ld(unsigned *in, unsigned *out){
   __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
   int gindex = threadIdx.x + blockIdx.x * blockDim.x;
   int lindex = threadIdx.x;

   temp[lindex] = in[gindex];

   if(threadIdx.x < RADIUS){
     temp[lindex - RADIUS]     = in[gindex - RADIUS];
     temp[lindex + BLOCK_SIZE] = in[gindex - BLOCK_SIZE];
   }


   __syncthreads();

   int result = 0;
   for(int offset = -RADIUS; offset < RADIUS; offset++){
     result += temp[lindex + offset];
   }


   out[gindex] = result;

 }

#include "ppm.h"

int main(void){
  // Hep data/1600.ppm kullanalim kolaylik acisindan,
  // cikti da output/1600.ppm olsun


  unsigned* inDat   = (unsigned*) malloc(sizeof(unsigned) * 16);
  unsigned* outDat  = (unsigned*) malloc(sizeof(unsigned) * 10);

  for(int i = 0; i<16; i++){
    inDat[i]     = i;
    outDat[i%10] = 0;
  }

  unsigned* GinDat ;
  unsigned* GoutDat;
  cudaMalloc((void **)&GinDat , sizeof(unsigned) * 16);
  cudaMalloc((void **)&GoutDat, sizeof(unsigned) * 10);

  cudaMemcpy(GinDat, inDat, 16 * sizeof(unsigned), cudaMemcpyHostToDevice);

  stencil_ld<<<10,7>>>(GinDat, GoutDat);

  cudaMemcpy(outDat, GoutDat, 10 * sizeof(unsigned), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for(int i = 0; i<10;i++){
    printf("%d ", outDat[i]);
  }
  printf("\n");

}
