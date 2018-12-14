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


__global__  void  cuda_crack(int password, int possibleLen, int setSize, bool *found, char* retGuess, long lastIndex) {
  if (!(*found)) {
    long index = (blockIdx.x * blockDim.x + threadIdx.x) + lastIndex;

    long currLen = (int)(logf(index) / logf(setSize)) + 1;

    char guess[6];

    for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
      guess[guessIndex] = map((index / (long) powf(setSize, guessIndex)) % setSize);
    }

    if (password == RSHash(guess, possibleLen)) {
        memcpy(retGuess, guess, possibleLen);
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
int original_main() {
    char passwordStr[] = "aabca";

    int possibleLen = strlen(passwordStr);
    int password = RSHash_cpu(passwordStr, possibleLen);
    int setSize = 36;

    bool *found;
    char *guess;
    cudaMallocManaged(&found, sizeof(bool));
    cudaMallocManaged(&guess, sizeof(char)*possibleLen);
    *found = false;


    long permutations = 0;
    for (int i = 1; i <= possibleLen; i++) {
      permutations += pow(setSize, i);
    }
    //printf("perm: %ld\n", permutations);
    long maxBlocks = 65000;
    long threadsPerBlock = 1024;


    int numGrids = (permutations / (threadsPerBlock * maxBlocks)) + 1;
    int numBlocks;
    for (long grid = 0; grid < numGrids; grid++) {
      if (permutations < threadsPerBlock*maxBlocks) {
        numBlocks = permutations / threadsPerBlock;
      } else {
        permutations -= threadsPerBlock*maxBlocks;
        numBlocks = threadsPerBlock*maxBlocks;
      }
      //<<<numBlocks, threadsPerBlock>>>
      cuda_crack<<<numBlocks, threadsPerBlock>>>(password, possibleLen, setSize, found, guess, (grid*maxBlocks*threadsPerBlock));
      cudaDeviceSynchronize();

      if (*found) {
        break;
      }
    }



    printf("Password: %s\n", guess);

    cudaFree(found);
    cudaFree(guess);

    return 0;
}

int speedtest(char* input_string) {
    char passwordStr[] = input_string;

    int possibleLen = strlen(passwordStr);
    int password = RSHash_cpu(passwordStr, possibleLen);
    int setSize = 36;

    struct timespec start, finish;
    double elapsed;
    printf("-Starting CUDA Password Cracker-\n");
    // Start Timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    bool *found;
    char *guess;
    cudaMallocManaged(&found, sizeof(bool));
    cudaMallocManaged(&guess, sizeof(char)*possibleLen);
    *found = false;


    long permutations = 0;
    for (int i = 1; i <= possibleLen; i++) {
      permutations += pow(setSize, i);
    }
    //printf("perm: %ld\n", permutations);
    long maxBlocks = 65000;
    long threadsPerBlock = 1024;


    int numGrids = (permutations / (threadsPerBlock * maxBlocks)) + 1;
    int numBlocks;
    for (long grid = 0; grid < numGrids; grid++) {
      if (permutations < threadsPerBlock*maxBlocks) {
        numBlocks = permutations / threadsPerBlock;
      } else {
        permutations -= threadsPerBlock*maxBlocks;
        numBlocks = threadsPerBlock*maxBlocks;
      }
      //<<<numBlocks, threadsPerBlock>>>
      cuda_crack<<<numBlocks, threadsPerBlock>>>(password, possibleLen, setSize, found, guess, (grid*maxBlocks*threadsPerBlock));
      cudaDeviceSynchronize();

      if (*found) {
        break;
      }
    }

    printf("Password: %s\n", guess);

    cudaFree(found);
    cudaFree(guess);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time: %f\n", elapsed);

    return 0;
}


int main()
{
  char* passwords[] = {(char*)"p", (char*)"ar", (char*)"esd", (char*)"reds",
                      (char*)"pooty", (char*)"spoots", (char*)"paral12", (char*)"bv37qi#f"};
  for(int i=0; i<10; i++){
    speedtest(passwords[i]);
  }
  return 0;
}
