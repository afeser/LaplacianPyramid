#include<stdio.h>
#include<stdlib.h>


__global__ void testKernel(void){
	printf("Naber Millet!!\n");
}


int main(void){
	testKernel<<<1,1>>>();

	cudaDeviceSynchronize();

	return 0;
}
