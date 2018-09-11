/*
 * Resmin aynisini veriyor...
 * Neden oldugu hakkinda en ufak fikrim de yok...
 * KOD EVDEKI BILGISAYARDA GAYET GUZEL CALISTI!!! BELKI DE MEMORY ERROR HANDLING YAPMALIYIM! cuda-memcheck
 */


#include <stdio.h>

 __global__ void stencil_ld(unsigned *X, unsigned *out, int width, int height){
   int kernel[3][3] = { 0, -1, 0,
                       -1, 5, -1,
                        0, -1, 0};
   int cikti;
   int x  = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel

   //if(x/width<1 || x/width>height-1 || x%width == width-1 || x%width == 1) return; // kenar noktalarinda

   cikti  =(kernel[0][0]*X[x-width-1]       +
            kernel[0][1]*X[x-width]         +
            kernel[0][2]*X[x-width+1]       +
            kernel[1][0]*X[x-1]             +
            kernel[1][1]*X[x]               +
            kernel[1][2]*X[x+1]             +
            kernel[2][0]*X[x+width-1]       +
            kernel[2][1]*X[x+width-1]       +
            kernel[2][2]*X[x+width-1]);


    if(cikti < 0)        out[x] = 0;
    else if(cikti > 255) out[x] = 255;
    else                 out[x] = cikti;

 }

#include "ppm.h"

int main(void){
  // Hep data/1600.ppm kullanalim kolaylik acisindan,
  // cikti da output/1600.ppm olsun
  struct Picture pic;
  char inFile[] = "data/1600.ppm";
  pic = read(inFile);

  dim3 dimBlock(32, 32);
  dim3 dimGrid (pic.width/32, pic.height/32);

  int size = pic.width * pic.height * sizeof(unsigned);

  unsigned *GinDat;
  unsigned *GoutDat;

  cudaMalloc((void **)&GinDat , size);
  cudaMalloc((void **)&GoutDat, size);

  printf("GinDat adres  = %p\n", GinDat);
  printf("GoutDat adres = %p\n", GoutDat);

  cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(pic.R, GoutDat, size, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(pic.G, GoutDat, size, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(pic.B, GoutDat, size, cudaMemcpyDeviceToHost);

  cudaFree(GinDat);
  cudaFree(GoutDat);

  char outFile[] = "output/GPU1600.ppm";
  write(outFile, pic);

}
