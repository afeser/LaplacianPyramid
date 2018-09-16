#include "ppm.hpp"
#include "pyramid.hpp"

__global__ void _r(pixelByte *I, pixelByte *O, pixelByte g, float sigma, float alpha, unsigned width, unsigned height){
  // f function is taken polinomial(at least a power function)
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  if(I[x] < g){
    O[x] = g - sigma*powf( (g - I[x]) / sigma, alpha);
  }else{
    O[x] = g + sigma*powf( (I[x] - g) / sigma, alpha);
  }
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

  Picture inPic = Picture(inputPath, true);


  Pyramid gaussianP;
  Pyramid laplacianP;
  Pyramid outputP;

  gaussianP.createGaussian(&inPic, pyramidHeight); // COOOK GARIIP!!!! biz buna veri yolluyoruz ama yolladigimiz objenin destructor fonksiyonu bu fonksiyon bittiginde de cagiriliyor!!!!
  laplacianP.createLaplacian(&inPic, pyramidHeight);

  outputP.createLaplacian(&inPic, pyramidHeight);

  for(int l = 0; l<pyramidHeight; l++){

    unsigned height = inPic.height / std::pow(2, l);
    unsigned width  = inPic.width  / std::pow(2, l);

    for(int y = 0; y<height; y++){
      for(int x = 0; x<width; x++){
        // Get Gaussian average...
        Pixel g = gaussianP.getLayer(l)->getPixel(x, y);

        // Map to a new image
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid2 (width/BLOCK_SIZE, height/BLOCK_SIZE);
        Picture mapped = Picture(width, height, true);

        // Direk Picture pointer'i seklinde yolladigimda calismiyor nedense, siyah resim aliyorum.
        _r<<<dimGrid2, dimBlock2>>>(gaussianP.getLayer(l)->R, mapped.R, g.R, sigma, alpha, width, height);
        _r<<<dimGrid2, dimBlock2>>>(gaussianP.getLayer(l)->G, mapped.G, g.G, sigma, alpha, width, height);
        _r<<<dimGrid2, dimBlock2>>>(gaussianP.getLayer(l)->B, mapped.B, g.B, sigma, alpha, width, height);

        // Find new Laplacian Pyramid
        Pyramid nLaplacianP;
        nLaplacianP.createLaplacian(&mapped, pyramidHeight-l); // burasi cooook buyuk memory kaplayacak is bittikten sonra silmezsek!!!

        // Update output pyramid
        Pixel p = nLaplacianP.getLayer(pyramidHeight-l)->getPixel(x, y);
        outputP.getLayer(pyramidHeight-l)->setPixel(x, y, p);
      }
    }
  }

  // Simdilik sadece en alt katmani yaziyorum (ustler nasil collapse edilecek ki???)
  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);

  for(int i = 0; i<number_of_additions; i++){
    _setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.R, outputP.getLayer(0)->R, inPic.width, inPic.height);
    _setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.G, outputP.getLayer(0)->G, inPic.width, inPic.height);
    // _setLaplacian<<<dimGrid2, dimBlock2>>>(inPic.B, outputP.getLayer(0)->B, inPic.width, inPic.height);
  }
  inPic.write(outputPath);

}
