#include <iostream>
#include <stdio.h>


//__global__ void cuda_crack(size_t *password, int *possibleLen, int *setSize, bool *found, char guess[], char guessMatrix[]) {


__device__ char map(int convert){
  if (convert < 10) {
    return (char) convert + 48;
  } else {
    return (char) convert + 87;
  }
}

__global__  void  cuda_crack( char  *ret,  int  a,  int  b, char* tempChar) {

  ret[threadIdx.x] = (char)36;

  int setSize = 36;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int currLen = (int)(logf(index) / logf(setSize)) + 1;
  char guess[5];

    for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
      guess[guessIndex] = map((index / (int) powf(setSize, guessIndex)) % (int) setSize);
    }

    if (index == 90) {
        memcpy(tempChar, guess, 3);
    }
}

int main() {
  char *ret, *tempChar;

  cudaMallocManaged(&ret, 100 * sizeof(char));
  cudaMallocManaged(&tempChar, 5 * sizeof(char));

  printf("Check before: %s\n", tempChar);
  
  //<<<numBlocks, threadsPerBlock>>>
  cuda_crack<<< 1, 100 >>>(ret, 10, 10, tempChar);
  cudaDeviceSynchronize();

  for(int i=0; i<100; i++)
    printf("%d: A+B = %c\n", i, ret[i]); 
  cudaFree(ret); 

  printf("Stirng: %s\n", tempChar);
  cudaFree(tempChar);
  return  0;
}
