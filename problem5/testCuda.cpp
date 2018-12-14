#include <iostream>
#include <stdio.h>

__global__  void  AplusB( char  *ret,  int  a,  int  b) {
	if (threadIdx.x == 99) {
		ret[threadIdx.x] = (char)36;
	} else {
		ret[threadIdx.x] = (char) (a + b + threadIdx.x);
	}
}

int main() {
    char *ret;
    cudaMallocManaged(&ret, 100 * sizeof(char));
	
	//<<<numBlocks, threadsPerBlock>>>
    AplusB<<< 1, 100 >>>(ret, 10, 10);
	cudaDeviceSynchronize();

    for(int i=0; i<100; i++)
        printf("%d: A+B = %c\n", i, ret[i]); 
    cudaFree(ret); 
    return  0;
}
