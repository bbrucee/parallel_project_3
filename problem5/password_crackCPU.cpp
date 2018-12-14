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

char map(int convert){
  if (convert < 10) {
    return (char) convert + 48;
  } else {
    return (char) convert + 87;
  }
}

//Set size is 36 characters and one blank character
float setSize = 36;
bool done = false;

int original_main() {
    char passwordStr[] = "aabca";

    int possibleLen = strlen(passwordStr);

    hash<string> ptr_hash;
    size_t password = ptr_hash(string(passwordStr));    
    printf("-Starting Non-Parallel Password Cracker-\n");


    struct timespec start, finish;
    double elapsed;

    // Start Timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Loop through len 1 - possible len
    for (int currLen = 1; currLen <= possibleLen; ++currLen) {
    // Loop for all possible combinations
      char* guess = new char[currLen + 1];
      memset(guess, '\0', currLen +1);
  
      for (int currChar = 0; currChar <= (pow(setSize, (float) currLen)); ++currChar) {
        // Set guess
        for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
          char temp = map((currChar / (int) pow(setSize, guessIndex)) % (int) setSize);
          guess[guessIndex] = temp;
        }
        //printf("Iteration: %d\tGuess: %s\n", currChar, guess);

        // Check if it compares
        if ( password == ptr_hash(string(guess))) {
          printf("Match Found Single!! \nLen: %d\tGuess: %s\n",currLen, guess);
          break;
        }
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time: %f\n", elapsed);

    return 0;
}

void* crack(void* args){
  // Get arguments
  struct params *params = (struct params*) args;
  size_t password = params->password;
  int possLen = params->passLen;
  int totalThreads = params->totalThreads;
  int currThread = params->currThread;
  int count = 0;

  hash<string> ptr_hash;

  //printf("Values: %d\t %s\n", currLen, password);
  for(int currLen = 1; currLen <= possLen; ++currLen) {
    char* guess = new char[currLen+1];
    memset(guess, '\0', currLen+1);

    int partitionOfPass = pow(setSize, (float) currLen) / totalThreads;
    int passStart = currThread * partitionOfPass;

    //printf("Pass: %d\t Thread: %d\t Start: %d\t End: %d\n", currLen, currThread, passStart, passStart + partitionOfPass);

    for (int currChar = passStart; currChar <= passStart + partitionOfPass; ++currChar) {
      if (done == true) {
        break;
      }
      count++;

      // Set guess
      for (int guessIndex = 0; guessIndex < currLen; ++guessIndex) {
        char temp = map((currChar / (int) pow(setSize, guessIndex)) % (int) setSize);
        guess[guessIndex] = temp;
      }
      //printf("Iteration: %d\tGuess: %s\n", currChar, guess);

      // Check if it compares
      if (password == ptr_hash(string(guess))) {

        printf("Match Found Parallel!! \nLen: %d\tGuess: %s\t Iterations: %d\t Current Count: %d\n",currLen, guess, count, currChar);
        done = true;
        return (void*) guess;
      }
    }
  }
   //printf("Thread: %d Finished! Iterations: %d\n", currThread, count);
  return NULL;
}

int speedtest(char* input_string) {
  char passwordStr[] = input_string;

  int possibleLen = strlen(passwordStr);

  hash<string> ptr_hash;
  size_t password = ptr_hash(string(passwordStr));    
  printf("-Starting pthread Password Cracker-\n");
  int numThreads = 4;

  struct timespec start, finish;
  double elapsed;

  // Start Timer
  clock_gettime(CLOCK_MONOTONIC, &start);

  pthread_t thread[numThreads];

    // Outer loop to check different lengths of passwords
    for (int i = 0; i < numThreads; i++){
      struct params* args = new struct params;
      args->password = password;
      args->passLen = possibleLen;
      args->totalThreads = numThreads;
      args->currThread = i;
      pthread_create( &thread[i], NULL, crack, (void*) args);
    }

    //printf("All threads running\n");

    for (int i = 0; i < numThreads; i++){
      char* pass_guess;
      pthread_join(thread[i], (void**) &pass_guess);
      if(pass_guess != NULL){
        printf("Returned Value: %s\n", pass_guess);
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time: %f\n", elapsed);
    
    return 0;
}

int main()
{
  char* passwords[] = {"bv37qi#f", "racecar", "141njif", "alx9123n", "123dnar3", "password", "parall12", "a2312", "912341", "1332"};
  for(int i=0; i<10; i++){
    speedtest(passwords[i]);
  }
  return 0;
}
