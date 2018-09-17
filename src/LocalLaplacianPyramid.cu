#include "ppm.hpp"
#include "pyramid.hpp"

__global__ void _r           (pixelByte *I, pixelByte *O, pixelByte g, float sigma, float alpha, unsigned width, unsigned height){
  // f function is taken polinomial(at least a power function)
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  int i;
  if(I[x] < g){
    i = g - sigma*powf( (g - I[x]) / sigma, alpha);
  }else{
    i = g + sigma*powf( (I[x] - g) / sigma, alpha);
  }
  if(i > 255){
    O[x] = 255;
  }else if(i < 0){
    O[x] = 0;
  }else{
    O[x] = i;
  }
}
__global__ void _upSample2    (pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(x > width*height-1){
    return;
  }

  out[(x%width)*2 + x/width*(width*4)]           = in[x];
  out[(x%width)*2 + x/width*(width*4)+1]         = in[x]; // 1 sagi
  out[(x%width)*2 + x/width*(width*4)+width*2]   = in[x]; // 1 alti 1 sagi
  out[(x%width)*2 + x/width*(width*4)+1+width*2] = in[x]; // 1 sagi 1 alti
}
__global__ void _setLaplacian(pixelByte *inPic, pixelByte *laplacian, unsigned width, unsigned height){
  /*
   * Set laplacian to recover the deleted data for Gaussian filter, not reduced size.
   */

  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  int i;
  i = inPic[x] + laplacian[x] - 128;
  if(i < 0){
   inPic[x] = 0;
  }else if(i > 255){
   inPic[x] = 255;
  }else{
   inPic[x] = i;
  }

}

void localLaplacianPyramid(char *inputPath,
                           char *outputPath,
                           const float sigma,
                           const float alpha,
                           const int pyramidHeight,
                           const int number_of_additions){


  Picture inPic(inputPath, true);


  Pyramid *gaussianP = new Pyramid();
  Pyramid *laplacianP = new Pyramid();
  Pyramid *outputP = new Pyramid();

  gaussianP->createGaussian(&inPic, pyramidHeight); // COOOK GARIIP!!!! biz buna veri yolluyoruz ama yolladigimiz objenin destructor fonksiyonu bu fonksiyon bittiginde de cagiriliyor!!!!
  laplacianP->createLaplacian(&inPic, pyramidHeight);

  outputP->createLaplacian(&inPic, pyramidHeight);

  for(int l = 0; l<pyramidHeight; l++){

    unsigned height = inPic.height / std::pow(2, l);
    unsigned width  = inPic.width  / std::pow(2, l);

    for(int y = 0; y<height; y++){
      for(int x = 0; x<width; x++){

        // Get Gaussian average...
        Pixel g = gaussianP->getLayer(l)->getPixel(x, y);

        // Map to a new image
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid2 (width/BLOCK_SIZE, height/BLOCK_SIZE);
        Picture mapped(width, height, true);

        // Direk Picture pointer'i seklinde yolladigimda calismiyor nedense, siyah resim aliyorum.
        _r<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(l)->R, mapped.R, g.R, sigma, alpha, width, height);
        _r<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(l)->G, mapped.G, g.G, sigma, alpha, width, height);
        _r<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(l)->B, mapped.B, g.B, sigma, alpha, width, height);

        // Find new Laplacian Pyramid
        Pyramid nLaplacianP;
        nLaplacianP.createLaplacian(&mapped, 1); // burasi cooook buyuk memory kaplayacak is bittikten sonra silmezsek!!!

        // Update output pyramid
        Pixel p = nLaplacianP.getLayer(0)->getPixel(x, y);
        outputP->getLayer(l)->setPixel(x, y, p);
      }
    }
  }

  outputP->getLayer(0)->write("YAZKIZIM.ppm");

  // Collapse the pyramid
  for(int i = pyramidHeight-1; i > 0; i--){
    unsigned width  = gaussianP->getLayer(i-1)->width;
    unsigned height = gaussianP->getLayer(i-1)->height;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid (width/2/BLOCK_SIZE, height/2/BLOCK_SIZE);

    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2 (width/BLOCK_SIZE, height/BLOCK_SIZE);

    _upSample2<<<dimGrid, dimBlock>>>(gaussianP->getLayer(i)->R, gaussianP->getLayer(i-1)->R, width/2, height/2);
    _upSample2<<<dimGrid, dimBlock>>>(gaussianP->getLayer(i)->G, gaussianP->getLayer(i-1)->G, width/2, height/2);
    _upSample2<<<dimGrid, dimBlock>>>(gaussianP->getLayer(i)->B, gaussianP->getLayer(i-1)->B, width/2, height/2);

    for(int z = 0; z<number_of_additions; z++){
      _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->R, outputP->getLayer(i-1)->R, width, height);
      _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->G, outputP->getLayer(i-1)->G, width, height);
      _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->B, outputP->getLayer(i-1)->B, width, height);
    }
  }
  gaussianP->getLayer(0)->write(outputPath);

}
