/*
 * SORUNU ANLADIIIIIM!
 * 32x32 blok boyutu cok fazla oldugu icin resim cok kuculuyor ve her tarafi siyah oluyor!!!!
 * Buyuk resimlerle denedigim zaman hafif oldu gibi aslinda
 */
 #include "ppm.h"

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

void blenddene(Picture pic1, Picture pic2){
  // Once yatay olarak blend edelim

  int i = pic1.height*pic1.width/2-blendLength*pic1.width/2+1;
  for(; i<pic1.height*pic1.width/2+blendLength*pic1.width/2; i++){
    float transparencyConstant = (float)(i/pic1.width - (pic1.height*pic1.width/2-blendLength*pic1.width/2)/pic1.width) / (blendLength);

    pic1.R[i] = pic1.R[i]*(1.0-transparencyConstant) + pic2.R[i]*transparencyConstant;
    pic1.G[i] = pic1.G[i]*(1.0-transparencyConstant) + pic2.G[i]*transparencyConstant;
    pic1.B[i] = pic1.B[i]*(1.0-transparencyConstant) + pic2.B[i]*transparencyConstant;
  }
  for(; i<pic1.height*pic1.width; i++){
    pic1.R[i] = pic2.R[i];
    pic1.G[i] = pic2.G[i];
    pic1.B[i] = pic2.B[i];
  }

}
Picture blend(Picture pic1, Picture pic2){
  // Once yatay olarak blend edelim
  Picture composite = Picture(pic1.width, pic1.height);

  int i = 0;
  for(i=0; i<composite.height*composite.width/2-blendLength*pic1.width/2+1; i++){
    composite.R[i] = pic1.R[i];
    composite.G[i] = pic1.G[i];
    composite.B[i] = pic1.B[i];
  }
  for(; i<composite.height*composite.width/2+blendLength*pic1.width/2; i++){
    float transparencyConstant = (float)(i/pic1.width - (composite.height*composite.width/2-blendLength*pic1.width/2)/pic1.width) / (blendLength);

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
Picture yukari(Picture inPic){
  /*
   * Yeni resim dondurdukten sonra aldigi resmi bellekten siler.
   */
   dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

   int size = sizeof(pixelByte) * inPic.width * inPic.height;

   Picture outPic = Picture(inPic.width/2, inPic.height/2);


  outPic.laplacianPicture.R = (pixelByte*) malloc(inPic.width*inPic.height*sizeof(pixelByte));
  outPic.laplacianPicture.G = (pixelByte*) malloc(inPic.width*inPic.height*sizeof(pixelByte));
  outPic.laplacianPicture.B = (pixelByte*) malloc(inPic.width*inPic.height*sizeof(pixelByte));

   pixelByte *GrawData;
   pixelByte *Ggaussed;
   pixelByte *GdownSampled;
   pixelByte *GupSampled;
   pixelByte *laplaced;

   cudaMalloc((void **)&GrawData,     size);
   cudaMalloc((void **)&Ggaussed,   size);
   cudaMalloc((void **)&GdownSampled, size/4);
   cudaMalloc((void **)&GupSampled,   size);
   cudaMalloc((void **)&laplaced,   size);

  {
    cudaMemcpy(GrawData, inPic.R, size, cudaMemcpyHostToDevice);

    reduce<<<dimGrid2, dimBlock2>>>     (GrawData, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, laplaced, inPic.width, inPic.height);


    cudaMemcpy(outPic.laplacianPicture.R, laplaced, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(outPic.R, GdownSampled, size/4, cudaMemcpyDeviceToHost);
  }
  {
     cudaMemcpy(GrawData, inPic.G, size, cudaMemcpyHostToDevice);

     reduce<<<dimGrid2, dimBlock2>>>     (GrawData, Ggaussed, inPic.width, inPic.height);
     downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

     upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

     getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, laplaced, inPic.width, inPic.height);

     cudaMemcpy(outPic.laplacianPicture.G, laplaced, size, cudaMemcpyDeviceToHost);
     cudaMemcpy(outPic.G, GdownSampled, size/4, cudaMemcpyDeviceToHost);
  }
  {
     cudaMemcpy(GrawData, inPic.B, size, cudaMemcpyHostToDevice);

     reduce<<<dimGrid2, dimBlock2>>>     (GrawData, Ggaussed, inPic.width, inPic.height);
     downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

     upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

     getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, laplaced, inPic.width, inPic.height);

     cudaMemcpy(outPic.laplacianPicture.B, laplaced, size, cudaMemcpyDeviceToHost);
     cudaMemcpy(outPic.B, GdownSampled, size/4, cudaMemcpyDeviceToHost);
  }

  outPic.laplacianPicture.width  = inPic.width;
  outPic.laplacianPicture.height = inPic.height;


  cudaFree(GrawData);
  cudaFree(Ggaussed);
  cudaFree(GdownSampled);
  cudaFree(GupSampled);
  cudaFree(laplaced);

  return outPic;

}
Picture asagi(Picture inPic){
  Picture outPic = Picture(inPic.width*2, inPic.height*2);

  pixelByte *GrawData;
  pixelByte *GupSampled;
  pixelByte *laplacian;

  unsigned size = inPic.width*inPic.height*sizeof(pixelByte);
  cudaMalloc((void**)&GrawData, size);
  cudaMalloc((void**)&GupSampled, 4*size);
  cudaMalloc((void**)&laplacian, 4*size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (inPic.width*2/BLOCK_SIZE, inPic.height*2/BLOCK_SIZE);

  {
    cudaMemcpy(GrawData, inPic.R, size, cudaMemcpyHostToDevice);
    cudaMemcpy(laplacian, inPic.laplacianPicture.R, 4*size, cudaMemcpyHostToDevice);
    upSample2<<<dimGrid, dimBlock>>>(GrawData, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacian, outPic.width, outPic.height);

    cudaMemcpy(outPic.R, GupSampled, 4*size, cudaMemcpyDeviceToHost);
  }
  {
    cudaMemcpy(GrawData, inPic.G, size, cudaMemcpyHostToDevice);
    cudaMemcpy(laplacian, inPic.laplacianPicture.G, 4*size, cudaMemcpyHostToDevice);
    upSample2<<<dimGrid, dimBlock>>>(GrawData, GupSampled, inPic.width, inPic.height);


    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacian, outPic.width, outPic.height);

    cudaMemcpy(outPic.G, GupSampled, 4*size, cudaMemcpyDeviceToHost);
  }
  {
    cudaMemcpy(GrawData, inPic.B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(laplacian, inPic.laplacianPicture.B, 4*size, cudaMemcpyHostToDevice);
    upSample2<<<dimGrid, dimBlock>>>(GrawData, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacian, outPic.width, outPic.height);

    cudaMemcpy(outPic.B, GupSampled, 4*size, cudaMemcpyDeviceToHost);
  }

  cudaFree(GrawData);
  cudaFree(GupSampled);
  cudaFree(laplacian);

  return outPic;

}
void blendElmaPortakal(){
  Picture elma, portakal;
  char inFileElma[]     = "data/appleorange/apple.ppm";
  char inFilePortakal[] = "data/appleorange/orange.ppm";
  char outFile[]        = "output/elmaPortakalBirlesim.ppm";
  elma                  = Picture(inFileElma);
  portakal              = Picture(inFilePortakal);

  Picture elma1 = yukari(elma);
  Picture elma2 = yukari(elma1);
  Picture elma3 = yukari(elma2);

  Picture portakal1 = yukari(portakal);
  Picture portakal2 = yukari(portakal1);
  Picture portakal3 = yukari(portakal2);

  // bunlarin laplacian'lari da birlestirilmek durumunda
  Picture elma1lap     = Picture(2*elma1.width, 2*elma1.height);
  elma1lap.R           = elma1.laplacianPicture.R;
  elma1lap.G           = elma1.laplacianPicture.G;
  elma1lap.B           = elma1.laplacianPicture.B;
  Picture portakal1lap = Picture(2*elma1.width, 2*elma1.height);
  portakal1lap.R       = portakal1.laplacianPicture.R;
  portakal1lap.G       = portakal1.laplacianPicture.G;
  portakal1lap.B       = portakal1.laplacianPicture.B;
  blenddene(elma1lap, portakal1lap);

  Picture elma2lap     = Picture(2*elma2.width, 2*elma2.height);
  elma2lap.R           = elma2.laplacianPicture.R;
  elma2lap.G           = elma2.laplacianPicture.G;
  elma2lap.B           = elma2.laplacianPicture.B;
  Picture portakal2lap = Picture(2*elma2.width,2* elma2.height);
  portakal2lap.R       = portakal2.laplacianPicture.R;
  portakal2lap.G       = portakal2.laplacianPicture.G;
  portakal2lap.B       = portakal2.laplacianPicture.B;
  blenddene(elma2lap, portakal2lap);

  Picture elma3lap     = Picture(2*elma3.width, 2*elma3.height);
  elma3lap.R           = elma3.laplacianPicture.R;
  elma3lap.G           = elma3.laplacianPicture.G;
  elma3lap.B           = elma3.laplacianPicture.B;
  Picture portakal3lap = Picture(2*elma3.width, 2*elma3.height);
  portakal3lap.R       = portakal3.laplacianPicture.R;
  portakal3lap.G       = portakal3.laplacianPicture.G;
  portakal3lap.B       = portakal3.laplacianPicture.B;
  blenddene(elma3lap, portakal3lap);


  Picture picBlended = blend(elma3, portakal3);
  picBlended.laplacianPicture.R = elma3lap.R;
  picBlended.laplacianPicture.G = elma3lap.G;
  picBlended.laplacianPicture.B = elma3lap.B;
  Picture picBlended1 = asagi(picBlended);
  picBlended1.laplacianPicture.R = elma2lap.R;
  picBlended1.laplacianPicture.G = elma2lap.G;
  picBlended1.laplacianPicture.B = elma2lap.B;
  Picture picBlended2 = asagi(picBlended1);
  picBlended2.laplacianPicture.R = portakal1lap.R;
  picBlended2.laplacianPicture.G = portakal1lap.G;
  picBlended2.laplacianPicture.B = portakal1lap.B;
  Picture picBlended3 = asagi(picBlended2);

  picBlended3.write(outFile);
}
void kucultBuyut(){
  Picture pic;
  char inFile[]  = "data/1600.ppm";
  char outFile[] = "output/laplacianDeneme1.ppm";
  pic = Picture(inFile);

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
