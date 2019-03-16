#include "ppm.hpp"
#include "mathematicalFunctions.cu"


int main(void){
  char inputDat[] = "data/flower.ppm";
  char inputFilter[] = "data/filter.ppm";
  char outputFile[] = "data/convOutput.ppm";

  Picture p = Picture(inputDat, true);
  Picture kernel = Picture(inputFilter, true);
  Picture o = Picture(p.width, p.height, true);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (p.width/BLOCK_SIZE, p.height/BLOCK_SIZE);

  afeser::mathematics::conv2<<<dimGrid, dimBlock>>>(p.R, o.R, p.width, p.height, 18*18, kernel.R);
  afeser::mathematics::conv2<<<dimGrid, dimBlock>>>(p.G, o.G, p.width, p.height, 18*18, kernel.G);
  afeser::mathematics::conv2<<<dimGrid, dimBlock>>>(p.B, o.B, p.width, p.height, 18*18, kernel.B);

  o.write(outputFile);
}
