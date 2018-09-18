#define BLOCK_SIZE 1 // kare olan blogun bir kenari

#include "LaplacianPyramid.cu"
#include "LocalLaplacianPyramid.cu"
#include <string.h>

void help(){
  printf("Use : \n");
  printf("command sharpen number_of_additions\n");
  printf("command sharpen input_name output_name number_of_additions\n\n");

  printf("command pyramid laplacian height\n");
  printf("command pyramid gaussian height\n\n");

  printf("command localLaplacian height number_of_additions\n");
}


int main(int argc, char *argv[]){
  if(argc<2){
    help();
    return 0;
  }

  if(!strcmp(argv[1], "sharpen")){
    char inFileElma[]   = "data/Sample.ppm";
    char outFile[]      = "output/SampleSharpened.ppm";
    char inputPath[50]  = "data/";
    char outputPath[50] = "output/";

    if(argc == 3){
      printf("Input  = %s\n", inFileElma);
      printf("Output = %s\n", outFile);

      sharpenEdges(inFileElma, outFile, atoi(argv[2]));
    }else if(argc == 5){
      printf("Input  = %s\n", strcat(inputPath, argv[2]));
      printf("Output = %s\n", strcat(outputPath, argv[3]));

      sharpenEdges(inputPath, outputPath, atoi(argv[4]));
    }else{
      help();
      return 0;
    }
  }

  if(!strcmp(argv[1], "pyramid")){
    char inFileElma[]    = "data/Sample.ppm";
    char outFile[]       = "output/SampleLayer";
    char outFileTemp[50];

    Picture in = Picture(inFileElma, true);

    int height = atoi(argv[3]);

    if(!strcmp(argv[2], "laplacian")){
      Pyramid p;
      p.createLaplacian(&in, height);

      for(int i = 0; i<height; i++){
        // C tarzi oldu malesef
        sprintf(outFileTemp, "%s%d.ppm", outFile, i);

        p.getLayer(i)->write(outFileTemp);
      }
    }else if(!strcmp(argv[2], "gaussian")){
        Pyramid p;
        p.createGaussian(&in, height);

        for(int i = 0; i<height; i++){
          // C tarzi oldu malesef
          sprintf(outFileTemp, "%s%d.ppm", outFile, i);

          p.getLayer(i)->write(outFileTemp);
        }
    }else{
      help();
      return 0;
    }
  }

  if(!strcmp(argv[1], "localLaplacian")){
    char inFileElma[]    = "data/Sample.ppm";
    char outFile[]       = "output/LocalLaplacianSample.ppm";

    const float sigma         = 0.4f;
    const float alpha         = 0.25f;

    int pyramidHeight       = atoi(argv[2]);
    int number_of_additions = atoi(argv[3]);

    localLaplacianPyramid(inFileElma,
                          outFile,
                          sigma,
                          alpha,
                          pyramidHeight,
                          number_of_additions
                        );
  }

}
