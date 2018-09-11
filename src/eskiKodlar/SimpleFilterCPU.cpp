/*
 * Zannediyorum bu kod hatali!!!!
 * Cunku GPU kodu ile arasinda daglar kadar fark var!
 */

#include "ppm.h"
#include<stdlib.h>

void filterCPU(unsigned* X, int height, int width, int kernel[3][3]){
  unsigned* temp = (unsigned*) malloc(sizeof(unsigned) * height * width);

  int S = 1;
  /*
  int S = 0;
  for(int i = 0; i<9; i++){
    S += *(&kernel[0][0]+i);
  }*/
  for(int x = 0; x<width*height; x++){
    if(0 && x%width < 3 || x/width/3 < 1 || x%width > (width-3) || x/width/3 > (height-1)){ //if it is at the edge(vertical or horizontal)
      temp[x] = X[x];
    }else{
      temp[x] = (kernel[0][0]*X[x-width-1]       +
                 kernel[0][1]*X[x-width]         +
                 kernel[0][2]*X[x-width+1]       +
                 kernel[1][0]*X[x-1]             +
                 kernel[1][1]*X[x]               +
                 kernel[1][2]*X[x+1]             +
                 kernel[2][0]*X[x+width-1]       +
                 kernel[2][1]*X[x+width-1]       +
                 kernel[2][2]*X[x+width-1]) / S;
        if(temp[x] < 0 ){
          temp[x] = 0;
        }else if(temp[x] > 255){
          temp[x] = 255;
        }
    }
  }
  for(int x = 0; x<width*height; x++){
    X[x] = temp[x];
  }
}



int main(int argc, char **argv){
  if(argc < 3){
    printf("Use : \n");
    printf("command input_file_name output_file_name\n");
    return 1;
  }

  int kernel[3][3] = { 0,  -1,   0,
                      -1,   5,  -1,
                       0,  -1,   0 };

  struct Picture pic = read(argv[1]);

  filterCPU(pic.R, pic.height, pic.width, kernel);
  filterCPU(pic.G, pic.height, pic.width, kernel);
  filterCPU(pic.B, pic.height, pic.width, kernel);

  write(argv[2], pic);


}
