#include "ppm.h"

__global__ void stencil_ld(pixelByte *in, pixelByte *out, int width, int height){
  int x  = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel


  if(x%2 == 0 || (x/width)%2 == 0){ //tek numarali rowlar da yok olacak ya...
    return; // yalnizca ciftlerde calistir(ya da teklerde)
  }


  if(x/width<3 || x/width>height-3 || x%width<3 || x%width>width-3){
    // kenarlar bunlar zaten
    return;
  }

  const int kernelSize = 5;
  float kernel[kernelSize][kernelSize] = {   6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            2.0755e-04,   8.3731e-02,   6.1869e-01,   8.3731e-02,   2.0755e-04,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08};

  float cikti;
  cikti = 0.0f ;
  for(int i1 = 0; i1<kernelSize; i1++){ // column'u gosteriyor
    for(int i2 = 0; i2<kernelSize; i2++){ // row'u gosteriyor
      cikti +=  in[x+(i2-(kernelSize/2))*width+(i1-(kernelSize/2))]*kernel[i2][i1];
    }
  }

  // (x/width - 1)                       = giren arkadasin kacinci satirda oldugu
  // (x/width - 1)/2                     = cikacak resmin kacinci satirda oldugu
  // (x/width - 1)/2*(width/2)           = cikacak resimdeki rowlar sebebiyle gelen nokta sayisi
  // (x%width)/2                         = o an bulundugu satirdan gelen nokta sayisi
  // (x/width-1)/2*(width/2)+(x%width)/2 = gerekli nokta
  int pixelNumber = ((x/width-1)/2*(width/2)+(x%width)/2);
  if(cikti < 0)        out[pixelNumber] = 0;
  else if(cikti > 255) out[pixelNumber] = 255;
  else                 out[pixelNumber] = (pixelByte)cikti;


}


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

 dim3 dimBlock(32, 32);
 dim3 dimGrid (pic.width/32, pic.height/32);

 int size = pic.width * pic.height * sizeof(pixelByte);

 pixelByte *GinDat;
 pixelByte *GoutDat;

 cudaMalloc((void **)&GinDat , size);
 cudaMalloc((void **)&GoutDat, size / 4); // cikti 4te biri olacak alan olarak

 struct Picture outPic;
 outPic.width  = pic.width/2;
 outPic.height = pic.height/2;
 outPic.R      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
 outPic.G      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
 outPic.B      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);

 printf("Calculating...\n");
 cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);
 stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
 cudaMemcpy(outPic.R, GoutDat, size/4, cudaMemcpyDeviceToHost);

/*
 cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);
 stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
 cudaMemcpy(outPic.G, GoutDat, size/4, cudaMemcpyDeviceToHost);

 cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);
 stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);
 cudaMemcpy(outPic.B, GoutDat, size/4, cudaMemcpyDeviceToHost);
*/
 cudaFree(GinDat);
 cudaFree(GoutDat);

 printf("Writing...\n");
 write(argv[2], outPic);

}
