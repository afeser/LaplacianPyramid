#include<stdio.h>
#include<stdlib.h>

__global__ void add(int *result, int *num1, int *num2){
	printf("Number 1 : %d", num1);
	*result = *num1 + *num2;
}
int main(int argc, char *argv[]){
	if(argc < 3){
		return 1;
	}

	int num1, num2, result;

	num1 = atoi(argv[1]);
	num2 = atoi(argv[2]);

	add<<<1,1>>>(&result, &num1, &num2);
	

	printf("Result is : %d", result);

	return 0;
}	
