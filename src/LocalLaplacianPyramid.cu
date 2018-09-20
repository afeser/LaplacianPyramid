#include "ppm.hpp"
#include "pyramid.hpp"

__global__ void _r              (pixelByte *I, pixelByte *O, pixelByte g, float sigma, float alpha, unsigned width, unsigned height){
  // f function is taken polinomial(at least a power function)
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  double out;

  // Map to [0,1]
  double iD = (double) I[x] / 255;
  double gD = (double) g / 255;

  double diffAbs = iD - gD;
  int sign = 1;
  if(diffAbs < 0){
    sign = -1;
    diffAbs = -diffAbs;
  }

  if(diffAbs < sigma){
    out = gD + sign*sigma*powf( diffAbs / sigma, alpha); // r_d
  }else{
    out = gD + sign*(
                      powf( (diffAbs - sigma), alpha) + sigma
                    ) ; // r_e
  }

  // Remap to [0,255]
  int i = out * 255;

  if(i > 255){
    O[x] = 255;
  }else if(i < 0){
    O[x] = 0;
  }else{
    O[x] = i;
  }
}
__global__ void _upSample2      (pixelByte *in, pixelByte *out, int width, int height){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel


  if(x > width*height-1){
    return;
  }

  out[(x%width)*2 + x/width*(width*4)] = in[x];

  if((x+1)%width != 0){ // 1 sagi
    // No row change
    out[(x%width)*2 + x/width*(width*4)+1] = (in[x+1] + in[x]) / 2;
  }else{
    out[(x%width)*2 + x/width*(width*4)+1] = in[x];
  }
  if((x+width)/width < height){ // 1 alti
    out[(x%width)*2 + x/width*(width*4)+width*2] = (in[x+width] + in[x]) / 2;
  }else{
    out[(x%width)*2 + x/width*(width*4)+width*2] = in[x];
  }
  if((x+width)/width < height && (x+1)%width != 0){
    out[(x%width)*2 + x/width*(width*4)+1+width*2] = (in[x+1] + in[x] + in[x+width] + in[x+width+1]) / 4; // 1 sagi 1 alti
  }else{
    out[(x%width)*2 + x/width*(width*4)+1+width*2] = in[x];
  }
}
__global__ void _setLaplacian   (pixelByte *inPic, pixelByte *laplacian, unsigned width, unsigned height){
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
__global__ void _getNeighborhood(pixelByte *motherPic, pixelByte *outPic, unsigned edgeLen, unsigned width, unsigned height, unsigned x, unsigned y){
  int i = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  int gercek = (i/edgeLen)*width + i%edgeLen + (x - (edgeLen/2 - 1)) + (y - (edgeLen/2 - 1))*width;

  if(gercek/width != (gercek+edgeLen-i%edgeLen)/width){
    outPic[i] = 0;
    return;
  }
  if(gercek < 0 || gercek > width*height - 1){
    outPic[i] = 0;
    return;
  }

  outPic[i] = motherPic[gercek];

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

    unsigned width  = inPic.width / std::pow(2, l);
    unsigned height = inPic.height/ std::pow(2, l);

    unsigned edgeLen = (unsigned) std::pow(2, l);
    unsigned padding    = (unsigned) std::pow(2, l) * 2; // bunun sayesinde civarindaki Gaussian'lari hesaplayabilecegiz ( (kernelSize - 1) / 2)

    edgeLen += 2*padding; // 2 taraftam

    for(int y = 0; y<height; y++){
      for(int x = 0; x<width; x++){
        // Get Gaussian average for each layer
        Pixel g = gaussianP->getLayer(l)->getPixel(x, y);

        // Determine the area
        dim3 dimBlockEdge(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGridEdge (edgeLen/BLOCK_SIZE, edgeLen/BLOCK_SIZE);
        Picture area(edgeLen, edgeLen, true);
        _getNeighborhood<<<dimGridEdge, dimBlockEdge>>>(inPic.R, area.R, edgeLen, inPic.width, inPic.height, x, y);
        _getNeighborhood<<<dimGridEdge, dimBlockEdge>>>(inPic.G, area.G, edgeLen, inPic.width, inPic.height, x, y);
        _getNeighborhood<<<dimGridEdge, dimBlockEdge>>>(inPic.B, area.B, edgeLen, inPic.width, inPic.height, x, y);

        // Map to a new image
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);
        Picture mapped(edgeLen, edgeLen, true);

        // Converting the base image to a new mapped image
        _r<<<dimGrid2, dimBlock2>>>(area.R, mapped.R, g.R, sigma, alpha, edgeLen, edgeLen);
        _r<<<dimGrid2, dimBlock2>>>(area.G, mapped.G, g.G, sigma, alpha, edgeLen, edgeLen);
        _r<<<dimGrid2, dimBlock2>>>(area.B, mapped.B, g.B, sigma, alpha, edgeLen, edgeLen);

        // Find new Laplacian Pyramid for the mapped image
        Pyramid nLaplacianP;
        nLaplacianP.createLaplacian(&mapped, l+1);

        // Update output pyramid
        Pixel p = nLaplacianP.getLayer(l)->getPixel(edgeLen/2 - 1, edgeLen/2 - 1);
        outputP->getLayer(l)->setPixel(x, y, p);
      }
    }
  }

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
