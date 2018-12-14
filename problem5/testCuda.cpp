#include <iostream>
#include <stdio.h>

__global__  void  AplusB( char  *ret,  int  a,  int  b, char* tempChar) {
  int index = threadIdx.x;
  char guess[] = "temp";
  if (threadIdx.x == 99) {
    ret[threadIdx.x] = (char)36;
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

/*
int index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Values: %d\t %s\n", currLen, password);
    int currLen = (int)(logf(index) / logf(*setSize)) + 1;
      char* guess1 = guessMatrix + index*sizeof(char)*(*possibleLen);

    //printf("Pass: %d\t Thread: %d\t Start: %d\t End: %d\n", currLen, currThread, passStart, passStart + partitionOfPass);

    // Set guess
    for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
      char temp = map((index / (int) pow(*setSize, guessIndex)) % (int) *setSize);
      guess1[guessIndex] = temp;
    }
    //printf("Iteration: %d\tGuess: %s\n", index, guess);

    // Check if it compares
    if (*password == RSHash(guess1, *possibleLen)) {
      memcpy(guess, guess1, sizeof(char)*currLen);
      *found = true;
    }
    //printf("Thread: %d Finished! Iterations: %d\n", currThread, count);
    free(guess1);
  }
  */