/*
 * Resmin aynisini veriyor...
 * Neden oldugu hakkinda en ufak fikrim de yok...
 * KOD EVDEKI BILGISAYARDA GAYET GUZEL CALISTI!!! BELKI DE MEMORY ERROR HANDLING YAPMALIYIM! cuda-memcheck
 */
 #include "ppm.h"

__global__ void upSampling(pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel

  out[(x%width)*2 + x/width*(width*4)] = x[in];
}
__global__ void gaussian(pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel


  if(x/width<3 || x/width>height-3 || x%width<3 || x%width>width-3){
    // kenarlar bunlar zaten
    out[x] = in[x];
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


  if(cikti < 0)        out[x] = 0;
  else if(cikti > 255) out[x] = 255;
  else                 out[x] = (pixelByte)cikti;

  out[x] *= 1;
}



__global__ void stencil_ld(pixelByte *in, pixelByte *out, int width, int height){
   int x           = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel
   int pixelNumber = ((x/width-1)/2*(width/2)+(x%width)/2);

   if(x%2 == 0 || (x/width)%2 == 0){ //tek numarali rowlar da yok olacak ya...
     return; // yalnizca ciftlerde calistir(ya da teklerde)
   }


   if(x/width<3 || x/width>height-3 || x%width<3 || x%width>width-3){
     // kenarlar bunlar zaten
     out[pixelNumber] = in[x];
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
   if(cikti < 0)        out[pixelNumber] = 0;
   else if(cikti > 255) out[pixelNumber] = 255;
   else                 out[pixelNumber] = (pixelByte)cikti;



 }


void asagi(int argc, char **argv){
  int antiAliasing = 10;
  struct Picture pic;
  pic = read(argv[1]);

  dim3 dimBlock(32, 32);
  dim3 dimGrid (pic.width/32, pic.height/32);
  dim3 dimGridSampled(pic.width*2/32, pic.height*2/32);

  int size = pic.width * pic.height * sizeof(pixelByte);

  pixelByte *GinDat;
  pixelByte *sampled;
  pixelByte *convolved;


  cudaMalloc((void **)&GinDat, size);
  cudaMalloc((void **)&sampled, size * 4);
  cudaMalloc((void **)&convolved, size * 4);

  struct Picture outPic;
  outPic.width  = pic.width*2;
  outPic.height = pic.height*2;
  outPic.R      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width*2 * pic.height*2);
  outPic.G      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width*2 * pic.height*2);
  outPic.B      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width*2 * pic.height*2);

  printf("Calculating...\n");
  cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);

  upSampling<<<dimGrid, dimBlock>>>(GinDat, sampled, pic.width, pic.height);

  gaussian<<<dimGridSampled, dimBlock>>>(sampled, convolved, pic.width*2, pic.height*2);
  for(int i = 0; i<antiAliasing; i++){
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
  }
  cudaMemcpy(outPic.R, convolved, size*4, cudaMemcpyDeviceToHost);


  cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);

  upSampling<<<dimGrid, dimBlock>>>(GinDat, sampled, pic.width, pic.height);

  gaussian<<<dimGridSampled, dimBlock>>>(sampled, convolved, pic.width*2, pic.height*2);
  for(int i = 0; i<antiAliasing; i++){
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
  }
  cudaMemcpy(outPic.G, convolved, size*4, cudaMemcpyDeviceToHost);


  cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);

  upSampling<<<dimGrid, dimBlock>>>(GinDat, sampled, pic.width, pic.height);

  gaussian<<<dimGridSampled, dimBlock>>>(sampled, convolved, pic.width*2, pic.height*2);
  for(int i = 0; i<antiAliasing; i++){
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
    gaussian<<<dimGridSampled, dimBlock>>>(convolved, convolved, pic.width*2, pic.height*2);
  }

  cudaMemcpy(outPic.B, convolved, size*4, cudaMemcpyDeviceToHost);


  cudaFree(GinDat);
  cudaFree(sampled);
  cudaFree(convolved);


  printf("Writing...\n");
  write(argv[2], outPic);
}
void yukari(int argc, char **argv){
  struct Picture pic;
  pic = read(argv[1]);

  dim3 dimBlock(32, 32);
  dim3 dimGrid (pic.width/32, pic.height/32);

  int size = pic.width * pic.height * sizeof(pixelByte);

  pixelByte *GinDat;
  pixelByte *level1;
  pixelByte *level2;
  //pixelByte *level3;
  //pixelByte *level4;


  cudaMalloc((void **)&GinDat, size);
  cudaMalloc((void **)&level1, size / 4);
  cudaMalloc((void **)&level2, size / 16); // cikti 4te biri olacak alan olarak
  //cudaMalloc((void **)&level3, size / 64); // cikti 4te biri olacak alan olarak
  //cudaMalloc((void **)&level4, size / 256); // cikti 4te biri olacak alan olarak

  //stencil_ld<<<dimGrid, dimBlock>>>(level2, level3, pic.width/4, pic.height/4);
  //stencil_ld<<<dimGrid, dimBlock>>>(level3, level4, pic.width/8, pic.height/8);
/*
  upSampling<<<dimGrid, dimBlock>>>(GoutDat, GinDat, pic.width/2, pic.height/2);

  convolution<<<dimGrid, dimBlock>>>(GinDat, GinDat2, pic.width, pic.height);
*/

  struct Picture outPic;
  outPic.width  = pic.width/2;
  outPic.height = pic.height/2;
  outPic.R      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
  outPic.G      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
  outPic.B      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);

  printf("Calculating...\n");
  cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, level1, pic.width, pic.height);
  //stencil_ld<<<dimGrid, dimBlock>>>(level1, level2, pic.width/2, pic.height/2);
  cudaMemcpy(outPic.R, level1, size/4, cudaMemcpyDeviceToHost);


/*
  cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);

  upSampling<<<dimGrid, dimBlock>>>(GoutDat, GinDat, pic.width/2, pic.height/2);

  convolution<<<dimGrid, dimBlock>>>(GinDat, GinDat2, pic.width, pic.height);

  cudaMemcpy(pic.G, GinDat2, size, cudaMemcpyDeviceToHost);


  cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, GoutDat, pic.width, pic.height);

  upSampling<<<dimGrid, dimBlock>>>(GoutDat, GinDat, pic.width/2, pic.height/2);

  convolution<<<dimGrid, dimBlock>>>(GinDat, GinDat2, pic.width, pic.height);

  cudaMemcpy(pic.B, GinDat2, size, cudaMemcpyDeviceToHost);
*/

  cudaFree(GinDat);
  cudaFree(level1);
  cudaFree(level2);
  //cudaFree(level3);
  //cudaFree(level4);


  printf("Writing...\n");
  write(argv[2], outPic);

}
int main(int argc, char **argv){
  // Hep data/1600.ppm kullanalim kolaylik acisindan,
  // cikti da output/1600.ppm olsun
  if(argc < 4){
    printf("Use : \n");
    printf("command input_file_name output_file_name a-or-y\n");
    return 1;
  }

  if(argv[3][0] == 'a'){
    asagi(argc, argv);
  }else if(argv[3][0] == 'y'){
    yukari(argc, argv);
  }

}
