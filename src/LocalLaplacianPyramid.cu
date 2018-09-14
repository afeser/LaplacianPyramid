#include "ppm.hpp"
#include "pyramid.hpp"

__global__ void _r(Picture I, Picture O, Pixel G, float sigma, float alpha){
  // f function is taken polinomial(at least a power function)
  int x = blockIdx.y*BLOCK_SIZE*I.width + blockIdx.x*BLOCK_SIZE + threadIdx.y*I.width + threadIdx.x; //current pixel
  pixelByte g;

  // Do for R, G and B
  g = G.R;
  if(I.R[x] < g){
    O.R[x] = g - sigma*powf( (g - I.R[x]) / sigma, alpha);
  }else{
    O.R[x] = g + sigma*powf( (I.R[x] - g) / sigma, alpha);
  }

  g = G.G;
  if(I.G[x] < g){
    O.G[x] = g - sigma*powf( (g - I.G[x]) / sigma, alpha);
  }else{
    O.G[x] = g + sigma*powf( (I.G[x] - g) / sigma, alpha);
  }

  g = G.B;
  if(I.B[x] < g){
    O.B[x] = g - sigma*powf( (g - I.B[x]) / sigma, alpha);
  }else{
    O.B[x] = g + sigma*powf( (I.B[x] - g) / sigma, alpha);
  }

}
__global__ void _setLaplacian(Picture inPic, Picture laplacian){
  /*
   * Set laplacian to recover the deleted data for Gaussian filter, not reduced size.
   */

  int x = blockIdx.y*BLOCK_SIZE*inPic.width + blockIdx.x*BLOCK_SIZE + threadIdx.y*inPic.width + threadIdx.x; //current pixel

  int i;
  i = inPic.R[x] + laplacian.R[x] - 128;
  if(i < 0){
   inPic.R[x] = 0;
  }else if(i > 255){
   inPic.R[x] = 255;
  }else{
   inPic.R[x] = i;
  }

  i = inPic.G[x] + laplacian.G[x] - 128;
  if(i < 0){
   inPic.G[x] = 0;
  }else if(i > 255){
   inPic.G[x] = 255;
  }else{
   inPic.G[x] = i;
  }

  i = inPic.B[x] + laplacian.B[x] - 128;
  if(i < 0){
   inPic.B[x] = 0;
  }else if(i > 255){
   inPic.B[x] = 255;
  }else{
   inPic.B[x] = i;
  }
}

void localLaplacianPyramid(char *inputPath,
                           char * outputPath,
                           const float sigma,
                           const float alpha,
                           const int pyramidHeight,
                           const int number_of_additions){

  Picture inPic = Picture(inputPath, true);

  Pyramid gaussianP;
  Pyramid laplacianP;
  Pyramid outputP;

  gaussianP.createGaussian(inPic, pyramidHeight);
  laplacianP.createLaplacian(inPic, pyramidHeight);

  outputP.createLaplacian(inPic, pyramidHeight);

  for(int l = 0; l<pyramidHeight; l++){
    unsigned height = inPic.height / std::pow(2, l);
    unsigned width  = inPic.width  / std::pow(2, l);

    for(int y = 0; y<height; y++){
      for(int x = 0; x<width; x++){
        // Get Gaussian average...
        Pixel g = gaussianP.getLayer(l).getPixel(x, y);

        // Map to a new image
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid2 (width/BLOCK_SIZE, height/BLOCK_SIZE);
        Picture mapped = Picture(width, height, true);
        _r<<<dimGrid2, dimBlock2>>>(inPic, mapped, g, sigma, alpha);

        // Find new Laplacian Pyramid
        Pyramid nLaplacianP;
        nLaplacianP.createLaplacian(mapped, pyramidHeight-l); // burasi cooook buyuk memory kaplayacak is bittikten sonra silmezsek!!!

        // Update output pyramid
        Pixel p = nLaplacianP.getLayer(l).getPixel(x, y);
        outputP.getLayer(l).setPixel(x, y, p);
      }
    }
  }

  // Simdilik sadece en alt katmani yaziyorum (ustler nasil collapse edilecek ki???)
  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

  for(int i = 0; i<number_of_additions; i++)
    _setLaplacian<<<dimGrid2, dimBlock2>>>(inPic, outputP.getLayer(0));

}
