#include <iostream>
#include <stdio.h>


//__global__ void cuda_crack(size_t *password, int *possibleLen, int *setSize, bool *found, char guess[], char guessMatrix[]) {

__global__  void  AplusB( char  *ret,  int  a,  int  b, char* tempChar) {

  if (threadIdx.x == 99) {
    ret[threadIdx.x] = (char)36;



  int setSize = 36;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int currLen = (int)(logf(index) / logf(*setSize)) + 1;
  char guess[] = new char[currLen];

  for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
    guess[guessIndex] = map((index / (int) powf(setSize, guessIndex)) % (int) setSize);
  }

  memcpy(tempChar, guess, 3);
  } else {
    ret[threadIdx.x] = (char) (a + b + threadIdx.x);
  }
}

int main() {
  char *ret, *tempChar;

  cudaMallocManaged(&ret, 100 * sizeof(char));
  cudaMallocManaged(&tempChar, 5 * sizeof(char));

  printf("Check before: %s\n", tempChar);
  
  //<<<numBlocks, threadsPerBlock>>>
  AplusB<<< 1, 100 >>>(ret, 10, 10, tempChar);
  cudaDeviceSynchronize();

  for(int i=0; i<100; i++)
    printf("%d: A+B = %c\n", i, ret[i]); 
  cudaFree(ret); 

  printf("Stirng: %s\n", tempChar);
  cudaFree(tempChar);
  return  0;
}