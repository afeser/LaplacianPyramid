#include "ppm.hpp"

#define blendLength 20 // kac pikselde transparency evaluate edilecek(en ust kademede)



__global__ void conv2(pixelByte *in      , pixelByte *out, int width, int height, int kernelSize, float kernel[]){
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
__global__ void reduce(pixelByte *in      , pixelByte *out, int width, int height){
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
__global__ void downSample2(pixelByte *in      , pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(x/width%2 == 1 || x%2 == 1){
    return;
  }

  out[(x/width/2)*width/2 + (x%width)/2] = in[x];
}
__global__ void upSample2(pixelByte *in      , pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(x > width*height-1){
    return;
  }

  out[(x%width)*2 + x/width*(width*4)]           = in[x];
  out[(x%width)*2 + x/width*(width*4)+1]         = in[x]; // 1 sagi
  out[(x%width)*2 + x/width*(width*4)+width*2]   = in[x]; // 1 alti 1 sagi
  out[(x%width)*2 + x/width*(width*4)+1+width*2] = in[x]; // 1 sagi 1 alti
}
__global__ void downSample(pixelByte *in      , pixelByte *out, int width, int height){
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
__global__ void upSample(pixelByte *in      , pixelByte *out, int width, int height){
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
__global__ void setLaplacian(pixelByte *image   , pixelByte *laplacian, int width, int height){
  /*
   * Set laplacian to recover the deleted data for Gaussian filter, not reduced size.
   */
   //return;
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
__global__ void _blend(pixelByte *ch1     , pixelByte *ch2, pixelByte *composite, unsigned width, unsigned height){
  int i = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(i < height*width/2-blendLength*width/2+1){
    composite[i] = ch1[i];
  }else if(i > height*width/2+blendLength*width/2){
    composite[i] = ch2[i];
  }else{
    float transparencyConstant = (float)(i/width - (height*width/2-blendLength*width/2)/width) / (blendLength);

    composite[i] = ch1[i]*(1.0-transparencyConstant) + ch2[i]*transparencyConstant;
  }
}

void reduceCPU(pixelByte *in, pixelByte *out, int width, int height){
  const int kernelSize = 5;
  float kernel[kernelSize*kernelSize] = {   6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            2.0755e-04,   8.3731e-02,   6.1869e-01,   8.3731e-02,   2.0755e-04,
                                            2.8089e-05,   1.1332e-02,   8.3731e-02,   1.1332e-02,   2.8089e-05,
                                            6.9625e-08,   2.8089e-05,   2.0755e-04,   2.8089e-05,   6.9625e-08};

  for(int x = 0; x < width*height; x++){
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
}

Picture blend(Picture pic1, Picture pic2){
  // 1) Blend the images...
  Picture composite = Picture(pic1.width, pic1.height, true);

  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (pic1.width/BLOCK_SIZE, pic1.height/BLOCK_SIZE);

  _blend<<<dimGrid2, dimBlock2>>>(pic1.R, pic2.R, composite.R, pic1.width, pic1.height);
  _blend<<<dimGrid2, dimBlock2>>>(pic1.G, pic2.G, composite.G, pic1.width, pic1.height);
  _blend<<<dimGrid2, dimBlock2>>>(pic1.B, pic2.B, composite.B, pic1.width, pic1.height);

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

  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.R, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->R, outPic.width, outPic.height);

    cudaMemcpy(outPic.R, GupSampled, size*4, cudaMemcpyDeviceToDevice);
  }
  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.G, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->G, outPic.width, outPic.height);

    cudaMemcpy(outPic.G, GupSampled, size*4, cudaMemcpyDeviceToDevice);
  }
  {
    upSample2<<<dimGrid, dimBlock>>>(inPic.B, GupSampled, inPic.width, inPic.height);

    setLaplacian<<<dimGrid2, dimBlock2>>>(GupSampled, laplacianPic->B, outPic.width, outPic.height);

    cudaMemcpy(outPic.B, GupSampled, size*4, cudaMemcpyDeviceToDevice);
  }

  cudaFree(GupSampled);

  return outPic;

}

void sharpenEdges(char *nameIn, char *nameOut, const int sharpDepth){
  Picture inPic           = Picture(nameIn, true);
  Picture newLaplacianPic = Picture(inPic.width, inPic.height, true);

  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

  unsigned size = inPic.width*inPic.height*sizeof(pixelByte);

  pixelByte *GdownSampled;
  pixelByte *Ggaussed;
  pixelByte *GrawData;
  pixelByte *GupSampled;

  cudaMalloc((void**)&GdownSampled, size/4);
  cudaMalloc((void**)&Ggaussed, size);
  cudaMalloc((void**)&GrawData, size);
  cudaMalloc((void**)&GupSampled, size);



  {
    cudaMemcpy(GrawData, inPic.R, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.R, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.R, inPic.width, inPic.height);
  }
  {

    cudaMemcpy(GrawData, inPic.G, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.G, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.G, inPic.width, inPic.height);  }
  {

    cudaMemcpy(GrawData, inPic.B, size, cudaMemcpyDeviceToDevice);
    reduce<<<dimGrid2, dimBlock2>>>     (inPic.B, Ggaussed, inPic.width, inPic.height);
    downSample2<<<dimGrid2, dimBlock2>>>(Ggaussed, GdownSampled, inPic.width, inPic.height);

    upSample2<<<dimGrid2, dimBlock2>>>  (GdownSampled, GupSampled, inPic.width/2, inPic.height/2);

    getLaplacian<<<dimGrid2, dimBlock2>>>(GrawData, GupSampled, newLaplacianPic.B, inPic.width, inPic.height);
  }

  cudaFree(Ggaussed);
  cudaFree(GdownSampled);

  for(int i = 0; i<sharpDepth; i++){
    setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.R, newLaplacianPic.R, inPic.width, inPic.height);
    setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.G, newLaplacianPic.G, inPic.width, inPic.height);
    setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.B, newLaplacianPic.B, inPic.width, inPic.height);
  }

  inPic.write(nameOut);


}
void blendElmaPortakal(char *pic1, char *pic2, char *picOut, const unsigned pyramidHeight){
  /*
  * Blend given 2 images with given depth and parameters specified as
  * #define statement.
  */
  Picture elma[pyramidHeight];
  Picture portakal[pyramidHeight];
  elma[0]     = Picture(pic1, true);
  portakal[0] = Picture(pic2, true);


  for(int i = 1; i<pyramidHeight; i++){
    elma[i]     = yukari(elma[i-1]);
    portakal[i] = yukari(portakal[i-1]);
  }

  Picture picBlended = blend(elma[pyramidHeight-1], portakal[pyramidHeight-1]);

  for(int i = pyramidHeight-1; 0<i; i--){
    Picture picBlendedLaplacian = blend(*elma[i].getLaplacian(), *portakal[i].getLaplacian());
    picBlended.addLaplacian(picBlendedLaplacian);
    picBlended = asagi(picBlended);
  }

  picBlended.write(picOut);

}
void gaussianBenchmark(int block_size){
  char inFile[]  = "data/adrian/current.ppm";
  char outFile[] = "output/adrianCurrent.ppm";
  Picture pic1 = Picture(inFile, true);

  Picture pic2 = Picture(pic1.width, pic1.height, true);

  dim3 dimBlock2(block_size, block_size);
  dim3 dimGrid2 (pic1.width/block_size, pic1.height/block_size);

  reduce<<<dimGrid2, dimBlock2>>>(pic1.R, pic2.R, pic1.width, pic1.height);
  reduce<<<dimGrid2, dimBlock2>>>(pic1.G, pic2.G, pic1.width, pic1.height);
  reduce<<<dimGrid2, dimBlock2>>>(pic1.B, pic2.B, pic1.width, pic1.height);

  pic2.write(outFile);

}
void gaussianBenchmarkCPU(){
  char inFile[]  = "data/adrian/current.ppm";
  char outFile[] = "output/adrianCurrent.ppm";
  Picture pic1 = Picture(inFile, false);

  Picture pic2 = Picture(pic1.width, pic1.height, false);

  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (pic1.width/BLOCK_SIZE, pic1.height/BLOCK_SIZE);

  reduceCPU(pic1.R, pic2.R, pic1.width, pic1.height);
  reduceCPU(pic1.R, pic2.R, pic1.width, pic1.height);
  reduceCPU(pic1.R, pic2.R, pic1.width, pic1.height);

  pic2.write(outFile);

}
