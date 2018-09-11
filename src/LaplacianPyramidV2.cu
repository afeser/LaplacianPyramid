/*
 * SORUNU ANLADIIIIIM!
 * 32x32 blok boyutu cok fazla oldugu icin resim cok kuculuyor ve her tarafi siyah oluyor!!!!
 * Buyuk resimlerle denedigim zaman hafif oldu gibi aslinda
 */
 #include "ppm.hpp"

#define BLOCK_SIZE 1 // kare olan blogun bir kenari
#define blendLength 20 // kac pikselde transparency evaluate edilecek(en ust kademede)



__global__ void conv2       (pixelByte *in, pixelByte *out, int width, int height, int kernelSize, float kernel[]){
  /*
   * Birakirken Octave ile kontrol ettim calisiyordu.
   */
  // cikacak image icin yaziyoruz ekran karti parametrelerini
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  float cikti;
  cikti = 0.0f ;
  for(int i1 = 0; i1<kernelSize; i1++){ // row'u gosteriyor
    for(int i2 = 0; i2<kernelSize; i2++){ // column'u gosteriyor
      int row    = i1 - kernelSize/2;
      int column = i2 - kernelSize/2;

      int gercek = x + row*width; // o anki kutucuk
      if(gercek/width != (gercek+column)/width) // column sebebiyle row degistiriyorsa
        continue;
      gercek += column;

      int sanal  = gercek - (kernelSize/2)*width
                          - 2*(gercek/width-kernelSize/2)*(kernelSize/2)
                          - (kernelSize/2);




      if(sanal < 0 || sanal > (width - 2*(kernelSize/2))*(height - 2*(kernelSize/2))-1){
        continue;
      }
      if(sanal/(width-2*(kernelSize/2)) != gercek/width-(kernelSize/2)){
        continue;
      }


      cikti += in[sanal]*kernel[kernelSize*kernelSize-1-(i2*kernelSize+i1)]; // tersten olmasi icin


      }
  }

  out[x] = (unsigned) cikti;

}
__global__ void reduce      (pixelByte *in, pixelByte *out, int width, int height){
  /*
   * Convolve with Gaussian kernel.
   * Normalde convoluyion resim boyutunu buyutuyor ama biz burada buyutmuyormus gibi yapalim, kenarlari atalim, kolaylik olmasi acisindan.
   */
  const int kernelSize = 5;
  float kernel[kernelSize*kernelSize] = {   6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            2.0755e-04,   8.3731e-02,   6.1869e-01,   8.3731e-02,   2.0755e-04,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08};
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  float cikti;
  cikti = 0.0f ;
  for(int i1 = 0; i1<kernelSize; i1++){ // row'u gosteriyor
    for(int i2 = 0; i2<kernelSize; i2++){ // column'u gosteriyor
      int row    = i1 - kernelSize/2;
      int column = i2 - kernelSize/2;

      int gercek = x + row*width; // o anki kutucuk
      if(gercek/width != (gercek+column)/width) // column sebebiyle row degistiriyorsa
        continue;
      gercek += column;

      if(gercek < 0 || gercek > width*height-1){
        continue;
      }

      cikti += in[gercek]*kernel[kernelSize*kernelSize-1-(i2*kernelSize+i1)]; // tersten olmasi icin
    }
  }

  out[x] = (unsigned) cikti;

}
__global__ void downSample2 (pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(x/width%2 == 1 || x%2 == 1){
    return;
  }

  out[(x/width/2)*width/2 + (x%width)/2] = in[x];
}
__global__ void upSample2   (pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(x > width*height-1){
    return;
  }

  out[(x%width)*2 + x/width*(width*4)]           = in[x];
  out[(x%width)*2 + x/width*(width*4)+1]         = in[x]; // 1 sagi
  out[(x%width)*2 + x/width*(width*4)+width*2]   = in[x]; // 1 alti 1 sagi
  out[(x%width)*2 + x/width*(width*4)+1+width*2] = in[x]; // 1 sagi 1 alti
}
__global__ void downSample  (pixelByte *in, pixelByte *out, int width, int height){
  /*
   * Denedim bu da calisiyordu.
   * Cift olan satirlari siliyoruz.
   * width and height are input width and height.
   */
   int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

   if((x/width)%2 == 0){
     out[x/width/2*width+x%width] = in[x];
   }
}
__global__ void upSample    (pixelByte *in, pixelByte *out, int width, int height){
  /*
   * Bunu da denedim calisiyor.
   * Satiri kopyaliyoruz, tek satirlari da 0 ile dolduruyoruz.(ilk satiri kopyala 1, satiri doldur)
   * width and height are input width and height.
   */
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  int satir = x/width;

  // o satiri input degerine
  out[x%width + 2*satir*width] = in[x];
  // bir sonraki satiri da 0'a esitle
  out[x%width+(2*satir+1)*width] = 0;

}
__global__ void getLaplacian(pixelByte *modified, pixelByte *original, pixelByte *out, int width, int height){
  /*
   * Get difference between two images for Gaussian filter, not reduced size.
   */
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  int i = 0;
  // TODO : Burayi cozemedim, ama min kayip icin 128 etrafina alip 0 alti ve 255 ustunu limitliyorum
  if(original[x] > modified[x]){
    i = 128; // offset ekleyelim
    i += modified[x] - original[x];
  }else{
    i = 128; // offset ekleyelim
    i += - original[x] + modified[x];
  }

  if(i<0){
    out[x] = 0;
  }else if(i>255){
    out[x] = 255;
  }else{
    out[x] = i;
  }
}
__global__ void setLaplacian(pixelByte *image, pixelByte *laplacian, int width, int height){
  /*
   * Set laplacian to recover the deleted data for Gaussian filter, not reduced size.
   */
   // return;
   int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

   int i;
   i = image[x] + laplacian[x] - 128;
   if(i < 0){
     image[x] = 0;
   }else if(i > 255){
     image[x] = 255;
   }else{
     image[x] = i;
   }
}

Picture blend(Picture pic1, Picture pic2){
  // 1) Blend the images...
  Picture composite = Picture(pic1.width, pic1.height, true);

  unsigned size = pic1.width * pic2.width * sizeof(pixelByte);

  // int i = 0;
  // Both of them are on GPU
  cudaMemcpy(composite.R, pic1.R, size/2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(composite.G, pic1.G, size/2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(composite.B, pic1.B, size/2, cudaMemcpyDeviceToDevice);

  cudaMemcpy(composite.R, pic2.R, size/2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(composite.G, pic2.G, size/2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(composite.B, pic2.B, size/2, cudaMemcpyDeviceToDevice);

  // for(i=0; i<composite.height*composite.width/2-blendLength*pic1.width/2+1; i++){
  //   composite.R[i] = pic1.R[i];
  //   composite.G[i] = pic1.G[i];
  //   composite.B[i] = pic1.B[i];
  // }
  // for(; i<composite.height*composite.width/2+blendLength*pic1.width/2; i++){
  //   float transparencyConstant = (float)(i/pic1.width - (composite.height*composite.width/2-blendLength*pic1.width/2)/pic1.width) / (blendLength);
  //
  //   composite.R[i] = pic1.R[i]*(1.0-transparencyConstant) + pic2.R[i]*transparencyConstant;
  //   composite.G[i] = pic1.G[i]*(1.0-transparencyConstant) + pic2.G[i]*transparencyConstant;
  //   composite.B[i] = pic1.B[i]*(1.0-transparencyConstant) + pic2.B[i]*transparencyConstant;
  // }
  // for(; i<composite.height*composite.width; i++){
  //   composite.R[i] = pic2.R[i];
  //   composite.G[i] = pic2.G[i];
  //   composite.B[i] = pic2.B[i];
  // }

  // 2) Blend the laplacians...
  Picture *l1, *l2;
  l1 = pic1.getLaplacian();
  l2 = pic2.getLaplacian();
  while(l1 != NULL){
    Picture newLaplacian = Picture(l1->width, l1->height, true);
    size = l1->width * l1->height * sizeof(pixelByte);

    cudaMemcpy(newLaplacian.R, l1->R, size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(newLaplacian.G, l1->G, size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(newLaplacian.B, l1->B, size/2, cudaMemcpyDeviceToDevice);

    cudaMemcpy(newLaplacian.R, l2->R, size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(newLaplacian.G, l2->G, size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(newLaplacian.B, l2->B, size/2, cudaMemcpyDeviceToDevice);

    // Add laplacian to the current picture
    composite.addLaplacian(newLaplacian);

    // for(i=0; i<height*width/2-blendLength*width/2+1; i++){
    //   newLaplacian.R[i] = l1->R[i];
    //   newLaplacian.G[i] = l1->G[i];
    //   newLaplacian.B[i] = l1->B[i];
    // }
    // for(; i<height*width/2+blendLength*width/2; i++){
    //   float transparencyConstant = (float)(i/width - (height*width/2-blendLength*width/2)/width) / (blendLength);
    //
    //   newLaplacian.R[i] = l1->R[i]*(1.0-transparencyConstant) + l2->R[i]*transparencyConstant;
    //   newLaplacian.G[i] = l1->G[i]*(1.0-transparencyConstant) + l2->G[i]*transparencyConstant;
    //   newLaplacian.B[i] = l1->B[i]*(1.0-transparencyConstant) + l2->B[i]*transparencyConstant;
    // }
    // for(; i<height*width; i++){
    //   newLaplacian.R[i] = l2->R[i];
    //   newLaplacian.G[i] = l2->G[i];
    //   newLaplacian.B[i] = l2->B[i];
    // }

    l1 = pic1.getLaplacian();
    l2 = pic2.getLaplacian();

  }
  return composite;
}
Picture yukari(Picture inPic){
  /*
   * Yeni resim dondurdukten sonra aldigi resmi bellekten siler.
   */
   dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

   int size = sizeof(pixelByte) * inPic.width * inPic.height;
   Picture outPic = Picture(inPic.width/2, inPic.height/2, true);
   Picture newLaplacianPic = Picture(inPic.width, inPic.height, true);

   outPic.laplacianPicture = inPic.laplacianPicture;

   pixelByte *GrawData;
   pixelByte *Ggaussed;
   pixelByte *GupSampled;

   cudaMalloc((void **)&Ggaussed, size);
   cudaMalloc((void **)&GupSampled, size);
   cudaMalloc((void **)&GrawData, size);


  {
    cudaMemcpy(GrawData, inPic.R, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.R, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, outPic.R, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (outPic.R, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.R, inPic.width, inPic.height);
  }
  {
    cudaMemcpy(GrawData, inPic.G, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.G, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, outPic.G, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (outPic.G, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.G, inPic.width, inPic.height);
  }
  {
    cudaMemcpy(GrawData, inPic.B, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.B, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, outPic.B, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (outPic.B, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.B, inPic.width, inPic.height);
  }

  outPic.addLaplacian(newLaplacianPic);

  cudaFree(Ggaussed);
  cudaFree(GupSampled);
  cudaFree(GrawData);

  return outPic;
}
Picture asagi(Picture inPic){
  Picture outPic = Picture(inPic.width*2, inPic.height*2, true);

  pixelByte *GupSampled;

  unsigned size = inPic.width*inPic.height*sizeof(pixelByte);

  cudaMalloc((void**)&GupSampled, 4*size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (inPic.width*2/BLOCK_SIZE, inPic.height*2/BLOCK_SIZE);

  Picture *laplacianPic = inPic.getLaplacian();
  outPic.laplacianPicture = inPic.laplacianPicture;

  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.R, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->R, outPic.width, outPic.height);

    cudaMemcpy(outPic.R, GupSampled, size, cudaMemcpyDeviceToDevice);
  }
  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.G, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->G, outPic.width, outPic.height);

    cudaMemcpy(outPic.G, GupSampled, size, cudaMemcpyDeviceToDevice);
  }
  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.B, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->B, outPic.width, outPic.height);

    cudaMemcpy(outPic.B, GupSampled, size, cudaMemcpyDeviceToDevice);
  }

  cudaFree(GupSampled);

  return outPic;

}
void blendElmaPortakal(){
  // Picture elma, portakal;
  // char inFileElma[]     = "data/appleorange/apple.ppm";
  // char inFilePortakal[] = "data/appleorange/orange.ppm";
  // char outFile[]        = "output/elmaPortakalBirlesim.ppm";
  // elma                  = Picture(inFileElma, true);
  // portakal              = Picture(inFilePortakal, true);

  // Picture elma1 = yukari(elma);
  // Picture elma2 = yukari(elma1);
  // Picture elma3 = yukari(elma2);

  // Picture portakal1 = yukari(portakal);
  // Picture portakal2 = yukari(portakal1);
  // Picture portakal3 = yukari(portakal2);

  // Picture picBlended = blend(elma1, portakal1);

  // Picture picBlended1 = asagi(picBlended);
  // Picture picBlended3 = asagi(picBlended2);
  // Picture picBlended2 = asagi(picBlended1);

  // portakal1.write(outFile);
}
void kucultBuyut(){
  Picture pic;
  char inFile[]  = "data/1600.ppm";
  char outFile[] = "output/laplacianDeneme1.ppm";
  pic = Picture(inFile, true);

  Picture pic1 = yukari(pic);
  Picture pic2 = yukari(pic1);

  Picture pic3 = asagi(pic2);
  pic3.laplacianPicture = pic1.laplacianPicture;
  Picture pic4 = asagi(pic3);

  pic4.write(outFile);

}

int main(void){
  blendElmaPortakal();
}
