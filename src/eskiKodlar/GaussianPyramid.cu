/*
 * SORUNU ANLADIIIIIM!
 * 32x32 blok boyutu cok fazla oldugu icin resim cok kuculuyor ve her tarafi siyah oluyor!!!!
 * Buyuk resimlerle denedigim zaman hafif oldu gibi aslinda
 */
 #include "ppm.h"

#define antiAliasing 0
#define BLOCK_SIZE 2 // kare olan blogun bir kenari
#define pyramidHeight 2
#define blendLength 20 // kac pikselde transparency evaluate edilecek(en ust kademede)

__global__ void upSampling(pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  out[(x%width)*2 + x/width*(width*4)]           = x[in];
  out[(x%width)*2 + x/width*(width*4)+1]         = x[in]; // 1 sagi
  out[(x%width)*2 + x/width*(width*4)+width*2]   = x[in]; // 1 alti 1 sagi
  out[(x%width)*2 + x/width*(width*4)+1+width*2] = x[in]; // 1 sagi 1 alti

}
__global__ void gaussian(pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel


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
   int x           = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel
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


struct Picture asagi(struct Picture pic){

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (pic.width/BLOCK_SIZE, pic.height/BLOCK_SIZE);
  dim3 dimGridSampled(pic.width*2/BLOCK_SIZE, pic.height*2/BLOCK_SIZE);

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

  return outPic;

}
struct Picture yukari(struct Picture pic){
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (pic.width/BLOCK_SIZE, pic.height/BLOCK_SIZE);

  int size = pic.width * pic.height * sizeof(pixelByte);

  pixelByte *GinDat;
  pixelByte *level1;

  cudaMalloc((void **)&GinDat, size);
  cudaMalloc((void **)&level1, size / 4);

  struct Picture outPic;
  outPic.width  = pic.width/2;
  outPic.height = pic.height/2;
  outPic.R      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
  outPic.G      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);
  outPic.B      = (pixelByte*) malloc(sizeof(pixelByte) * pic.width/2 * pic.height/2);

  printf("Calculating...\n");
  cudaMemcpy(GinDat, pic.R, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, level1, pic.width, pic.height);
  cudaMemcpy(outPic.R, level1, size/4, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.G, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, level1, pic.width, pic.height);
  cudaMemcpy(outPic.G, level1, size/4, cudaMemcpyDeviceToHost);

  cudaMemcpy(GinDat, pic.B, size, cudaMemcpyHostToDevice);
  stencil_ld<<<dimGrid, dimBlock>>>(GinDat, level1, pic.width, pic.height);
  cudaMemcpy(outPic.B, level1, size/4, cudaMemcpyDeviceToHost);

  cudaFree(GinDat);
  cudaFree(level1);

  return outPic;
}
struct Picture blend(struct Picture pic1, struct Picture pic2){
  // Once yatay olarak blend edelim
  struct Picture composite;
  composite.width  = pic1.width;
  composite.height = pic1.height;
  composite.R      = (pixelByte*) malloc(sizeof(pixelByte) * composite.width * composite.height);
  composite.G      = (pixelByte*) malloc(sizeof(pixelByte) * composite.width * composite.height);
  composite.B      = (pixelByte*) malloc(sizeof(pixelByte) * composite.width * composite.height);

  int i = 0;
  for(i=0; i<composite.height*composite.width/2-blendLength*pic1.width/2+1; i++){
    composite.R[i] = pic1.R[i];
    composite.G[i] = pic1.G[i];
    composite.B[i] = pic1.B[i];
  }
  for(; i<composite.height*composite.width/2+blendLength*pic1.width/2; i++){
    float transparencyConstant = (float)(i/pic1.width - (composite.height*composite.width/2-blendLength*pic1.width/2)/pic1.width) / (blendLength);

    if(transparencyConstant<0.0 || transparencyConstant>1.0){
      printf("%f ", transparencyConstant);
    }

    composite.R[i] = pic1.R[i]*(1.0-transparencyConstant) + pic2.R[i]*transparencyConstant;
    composite.G[i] = pic1.G[i]*(1.0-transparencyConstant) + pic2.G[i]*transparencyConstant;
    composite.B[i] = pic1.B[i]*(1.0-transparencyConstant) + pic2.B[i]*transparencyConstant;
  }
  for(; i<composite.height*composite.width; i++){
    composite.R[i] = pic2.R[i];
    composite.G[i] = pic2.G[i];
    composite.B[i] = pic2.B[i];
  }

  return composite;
}

int main(void){
  // TODO Bu picture degiskenlerini free etmem lazim!!!!
  struct Picture picElma; // elma
  struct Picture picPortakal; // portakal
  struct Picture composite;

  char elmaFile[]     = "data/appleorange/apple.ppm";
  char portakalFile[] = "data/appleorange/orange.ppm";
  //char elmaFile[]     = "data/daylight.ppm";
  //char portakalFile[] = "data/bridge.ppm";

  printf("Dosyalar okunuyor...\n");
  picElma     = read(elmaFile);
  picPortakal = read(portakalFile);

  for(int i = 0; i<pyramidHeight; i++){
    picElma = yukari(picElma);
  }
  for(int i = 0; i<pyramidHeight; i++){
    picPortakal = yukari(picPortakal);
  }

  printf("Resimler birlestiriliyor...\n");
  composite = blend(picPortakal, picElma);


  for(int i = 0; i<pyramidHeight; i++){
    composite = asagi(composite);
  }

  printf("Cikti dosyasi yazdiriliyor...\n");
  char outFile[] = "HDD/elmaPortakal.ppm";
  write(outFile, composite);

}
