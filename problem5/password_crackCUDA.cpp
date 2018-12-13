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

using namespace std;

__device__ char map(int convert){
  if (convert < 10) {
    return (char) convert + 48;
  } else {
    return (char) convert + 87;
  }
}

__global__ void cuda_crack(size_t *password, int *possibleLen, int *setSize, bool *found, char guess[]) {
  if(!*found) {
    hash<string> ptr_hash;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Values: %d\t %s\n", currLen, password);
    int currLen = (int)(logf(index) / logf(*setSize));
    memset(guess, '\0', *possibleLen);

    //printf("Pass: %d\t Thread: %d\t Start: %d\t End: %d\n", currLen, currThread, passStart, passStart + partitionOfPass);

    // Set guess
    for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
      char temp = map((index / (int) pow(*setSize, guessIndex)) % (int) *setSize);
      guess[guessIndex] = temp;
    }
    //printf("Iteration: %d\tGuess: %s\n", index, guess);

    // Check if it compares
    if (*password == ptr_hash(std::string(guess))) {

      printf("Match Found Parallel!! Guess: %s\t ", guess);
      *found = true;
    }
    //printf("Thread: %d Finished! Iterations: %d\n", currThread, count);
  }
}


//Set size is 36 characters and one blank character
int main() {

    size_t *password;
    int *setSize, *possibleLen;
    bool *found;
    char *guess;

    cudaMallocManaged(&password, sizeof(size_t));
    cudaMallocManaged(&setSize, sizeof(int));
    cudaMallocManaged(&possibleLen, sizeof(int));
    cudaMallocManaged(&found, sizeof(bool));

    char passwordStr[] = "aabca";
    hash<string> ptr_hash;

    *password = ptr_hash(string(passwordStr));
    *setSize = 36;
    *possibleLen = strlen(passwordStr);
    *found = false;

    cudaMallocManaged(&guess, sizeof(char) * (*possibleLen));

    int permutations = 0;
    for (int i = 1; i <= *possibleLen; i++) {
      permutations += pow(*setSize, i);
    }

    int threadsPerBlock = 256;
    int numBlocks = (permutations + threadsPerBlock -1) / threadsPerBlock;

    cuda_crack<<<threadsPerBlock, numBlocks>>>(password, possibleLen, setSize, found, guess);

    cudaDeviceSynchronize();

    printf("Password: %s\n", guess);

    cudaFree(password);
    cudaFree(setSize);
    cudaFree(possibleLen);
    cudaFree(found);

    return 0;
}

