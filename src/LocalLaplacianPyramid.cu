#include "ppm.hpp"
#include "pyramid.hpp"

__global__ void _llf(pixelByte *I, pixelByte *O, unsigned width, unsigned height, int fact, float ref, float sigma){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  O[x] = fact*(I[x] - ref) * expf(
                                - (I[x]-ref) * (I[x]-ref) / (2 * sigma * sigma)
                              );

}
__global__ void updateOutputLaplacian(pixelByte *tempLaplace, pixelByte *outLaplace, pixelByte *gaussian, unsigned width, unsigned height, float ref, float discretisation_step){
  int x = blockIdx.y*BLOCK_SIZE*width + blockIdx.x*BLOCK_SIZE + threadIdx.y*width + threadIdx.x; //current pixel

  outLaplace[x] += (fabsf(gaussian[x] - ref) < discretisation_step) *
                   tempLaplace[x] *
                   (1 - fabsf(gaussian[x] - ref) / discretisation_step);

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

  inPic[x] = inPic[x] - laplacian[x];


}

void localLaplacianPyramid(char *inputPath,
                           char *outputPath,
                           const float sigma,
                           const int pyramidHeight,
                           const int fact,
                           const int N
                           ){


  Picture inPic(inputPath, true);

  const float discretisation_step =  1.0 /  (N-1); // linespace tanimi boyle cunku

  Pyramid *gaussianP = new Pyramid();
  Pyramid *outputP = new Pyramid();

  gaussianP->createGaussian(&inPic, pyramidHeight);

  outputP->createLaplacian(&inPic, pyramidHeight);

  for(float ref = 0; ref<=1; ref+=discretisation_step){
    // Map to a new image
    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2 (inPic.width/BLOCK_SIZE, inPic.height/BLOCK_SIZE);
    Picture mapped(inPic.width, inPic.height, true);

    // Converting the base image to a new mapped image
    _llf<<<dimGrid2, dimBlock2>>>(inPic.R, mapped.R, inPic.width, inPic.height, fact, ref, sigma);
    _llf<<<dimGrid2, dimBlock2>>>(inPic.G, mapped.G, inPic.width, inPic.height, fact, ref, sigma);
    _llf<<<dimGrid2, dimBlock2>>>(inPic.B, mapped.B, inPic.width, inPic.height, fact, ref, sigma);

    // Find new Laplacian Pyramid from the mapped image
    Pyramid tempLaplacian;
    tempLaplacian.createLaplacian(&mapped, pyramidHeight);

    // Do for all layers
    for(int l = 0; l<pyramidHeight; l++){
      unsigned width  = inPic.width /std::pow(2, l);
      unsigned height = inPic.height/std::pow(2, l);

      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
      dim3 dimGrid (width/BLOCK_SIZE, height/BLOCK_SIZE);

      updateOutputLaplacian<<<dimGrid, dimBlock>>>(tempLaplacian.getLayer(l)->R, outputP->getLayer(l)->R, gaussianP->getLayer(l)->R, width, height, ref, discretisation_step);
      updateOutputLaplacian<<<dimGrid, dimBlock>>>(tempLaplacian.getLayer(l)->G, outputP->getLayer(l)->G, gaussianP->getLayer(l)->G, width, height, ref, discretisation_step);
      updateOutputLaplacian<<<dimGrid, dimBlock>>>(tempLaplacian.getLayer(l)->B, outputP->getLayer(l)->B, gaussianP->getLayer(l)->B, width, height, ref, discretisation_step);
    }

  }
  Pyramid piramit = Pyramid();
  piramit.createLaplacian(&inPic, 5);
  piramit.getLayer(0)->write("GERCEKLAPLAS.ppm");
  outputP->getLayer(0)->write("LAPLACIAN.ppm");

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

    _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->R, outputP->getLayer(i-1)->R, width, height);
    _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->G, outputP->getLayer(i-1)->G, width, height);
    _setLaplacian<<<dimGrid2, dimBlock2>>>(gaussianP->getLayer(i-1)->B, outputP->getLayer(i-1)->B, width, height);
  }
  gaussianP->getLayer(0)->write(outputPath);

}
