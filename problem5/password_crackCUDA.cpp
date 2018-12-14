#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <functional>
#include <string>

#include <math.h>
#include <signal.h>
#include <ctime>
#include <pthread.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__device__ char map(int convert){
  if (convert < 10) {
    return (char) convert + 48;
  } else {
    return (char) convert + 87;
  }
}

__device__ int RSHash(char str[], int s)
{
  unsigned int b = 378551;
  unsigned int a = 63689;
  unsigned int hash = 0;

  for(int i = 0; i < s; i++)
  {
    hash = hash * a + str[i];
    a    = a * b;
  }

  return (hash & 0x7FFFFFFF);
}


__global__  void  cuda_crack(int password, int possibleLen, int setSize, bool *found, char* retGuess) {

  if (!(*found)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int currLen = (int)(logf(index) / logf(setSize)) + 1;

    char guess[5];

    for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
      guess[guessIndex] = map((index / (int) powf(setSize, guessIndex)) % (int) setSize);
    }

    if (password == RSHash(guess, 5)) {
        memcpy(retGuess, guess, 5);
        *found = true;
    }
  }
}


int RSHash_cpu(char str[], int s)
{
  unsigned int b = 378551;
  unsigned int a = 63689;
  unsigned int hash = 0;

  for(int i = 0; i < s; i++)
  {
    hash = hash * a + str[i];
    a    = a * b;
  }

  return (hash & 0x7FFFFFFF);
}

//Set size is 36 characters and one blank character
int main() {
    char passwordStr[] = "aabca";

    int possibleLen = strlen(passwordStr);
    int password = RSHash_cpu(passwordStr, possibleLen);
    int setSize = 36;

    bool *found;
    char *guess;
    cudaMallocManaged(&found, sizeof(bool));
    cudaMallocManaged(&guess, sizeof(char)*possibleLen);
    *found = false;


    int permutations = 0;
    for (int i = 1; i <= possibleLen; i++) {
      permutations += pow(setSize, i);
    }


    int threadsPerBlock = 1024;
    int numBlocks = permutations / threadsPerBlock;

    //<<<numBlocks, threadsPerBlock>>>
    cuda_crack<<<numBlocks, threadsPerBlock>>>(password, possibleLen, setSize, found, guess);

    cudaDeviceSynchronize();

    printf("Password: %s\n", guess);

    cudaFree(found);
    cudaFree(guess);

    return 0;
}

