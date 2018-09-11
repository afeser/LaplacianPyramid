#include<stdio.h>
#include<stdlib.h>


// CUDA runtime
#include <cuda_runtime.h>



__global__ void add(int *result, int *num1, int *num2){
	*result = *num1 + *num2;
}


int main(int argc, char *argv[]){
	if(argc < 3){
		return 1;
	}

	int num1, num2, result;
	int *num1G, *num2G, *resultG;

	cudaMalloc((void **)&num1G, sizeof(int));
	cudaMalloc((void **)&num2G, sizeof(int));
	cudaMalloc((void **)&resultG, sizeof(int));

	num1 = atoi(argv[1]);
	num2 = atoi(argv[2]);

	cudaMemcpy(num1G, &num1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(num2G, &num2, sizeof(int), cudaMemcpyHostToDevice);

	add<<<1,1>>>(resultG, num1G, num2G);

	cudaMemcpy(&result, resultG, sizeof(int), cudaMemcpyDeviceToHost); // ilk parametre hep ustune yazilacak olan

	cudaFree(num1G);
	cudaFree(num2G);
	cudaFree(resultG);


	printf("Result is : %d\n", result);

	return 0;
}
