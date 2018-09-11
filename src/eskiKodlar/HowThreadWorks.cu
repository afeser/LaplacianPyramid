
/*
 * CUDA Basics page 51
 *
 * BU KODU DA ANLAMADIM VALLA
 */
#define BLOCK_SIZE 1
#define RADIUS 3
#define M      1
#define N      1

#include <stdio.h>

 __global__ void stencil_ld(unsigned *in, unsigned *out){
   printf("Thread %d : %d\n", threadIdx.x, in[threadIdx.x]);
   out[threadIdx.x] = 2 * in[threadIdx.x];
   printf("out location : %p\n", out+threadIdx.x);
   printf("in %d : %d\n" , threadIdx.x, in[threadIdx.x]);
   printf("out %d : %d\n", threadIdx.x, out[threadIdx.x]);

   __syncthreads();

   /*
   __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
   int gindex = threadIdx.x + blockIdx.x * blockDim.x;
   int lindex = threadIdx.x + RADIUS;

   temp[lindex] = in[gindex];

   if(threadIdx.x < RADIUS){
     temp[lindex - RADIUS]     = in[gindex - RADIUS];
     temp[lindex + BLOCK_SIZE] = in[gindex - BLOCK_SIZE];
   }

   int result = 0;
   for(int offset = -RADIUS; offset < RADIUS; offset++){
     result += temp[lindex + offset];
   }

   __syncthreads();

   out[gindex] = result;
   */
 }

#include "ppm.h"

int main(void){
  // Hep data/1600.ppm kullanalim kolaylik acisindan,
  // cikti da output/1600.ppm olsun


  unsigned* inDat   = (unsigned*) malloc(sizeof(unsigned) * 16);
  unsigned* outDat  = (unsigned*) malloc(sizeof(unsigned) * 16);

  for(int i = 0; i<16; i++){
    inDat[i]     = i;
    outDat[i] = 0;
  }

  unsigned* GinDat ;
  unsigned* GoutDat;
  cudaMalloc((void **)&GinDat , sizeof(unsigned) * 16);
  cudaMalloc((void **)&GoutDat, sizeof(unsigned) * 16);

  cudaMemcpy(GinDat, inDat, 16 * sizeof(unsigned), cudaMemcpyHostToDevice);

  printf("Real out location : %p\n", GoutDat);

  stencil_ld<<<1,16>>>(GinDat, GoutDat);

  cudaMemcpy(outDat, GoutDat, 16 * sizeof(unsigned), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(GinDat);
  cudaFree(GoutDat);

  for(int i = 0; i<16;i++){
    printf("%d ", outDat[i]);
  }
  printf("\n");

}
