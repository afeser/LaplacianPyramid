/*
 * 4'te 1'lik Gauss icin yapiyorum bunu(4 pixel girdi 1 pixel cikti)
 */


#include <stdio.h>

 __global__ void gaussian_filter(unsigned *in, unsigned *out, int width, int height){
   __shared__ int cikti;
   cikti = 0;

   __syncthreads();

   cikti += in[blockIdx.y*width*2 + blockIdx.x*2 + threadIdx.y*width + threadIdx.x];

   __syncthreads();

   out[blockIdx.y*width/2 + blockIdx.x] = cikti; // ciktiyi bir sayiya boldugumde garip bir sekilde resim karariyor???(oysa 4 sayiyi topluyoruz, neden ort almiyoruz???)
   
 }

#include "ppm.h"

int main(int argc, char **argv){
  // Hep data/1600.ppm kullanalim kolaylik acisindan,
  // cikti da output/1600.ppm olsun
  if(argc < 3){
    printf("Use : \n");
    printf("command input_file_name output_file_name\n");
    return 1;
  }

  struct Picture pic;
  pic = read(argv[1]);

  dim3 dimBlock(2, 2); // her ikiyi bir tane pixele donusturuyoruz simdi
  dim3 dimGrid (pic.width/2, pic.height/2);

  int size = pic.width * pic.height * sizeof(unsigned);

  unsigned *GinDat;
  unsigned *GoutDat;

  cudaMalloc((void **)&GinDat , size);
  cudaMalloc((void **)&GoutDat, size/4);


  struct Picture outPic;
  outPic.width  = pic.width/2;
  outPic.height = pic.height/2;
  outPic.R      = (unsigned*) malloc(sizeof(unsigned) * pic.width * pic.height / 4);
  outPic.G      = (unsigned*) malloc(sizeof(unsigned) * pic.width * pic.height / 4);
  outPic.B      = (unsigned*) malloc(sizeof(unsigned) * pic.width * pic.height / 4);

  cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);
  gaussian_filter<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(outPic.R, GoutDat, size / 4, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);
  gaussian_filter<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(outPic.G, GoutDat, size / 4, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);
  gaussian_filter<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
  cudaMemcpy(outPic.B, GoutDat, size / 4, cudaMemcpyDeviceToHost);

  cudaFree(GinDat);
  cudaFree(GoutDat);


  write(argv[2], outPic);

}
