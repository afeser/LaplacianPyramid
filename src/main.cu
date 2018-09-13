#include "LaplacianPyramid.cu"

int main(void){
  char inFileElma[] = "data/1600.ppm";
  char outFile[]    = "output/LaplacianL1.ppm";
  Picture in = Picture(inFileElma, true);

  Picture yukari1 = yukari(in);

  (*yukari1.getLaplacian()).write(outFile);
}

