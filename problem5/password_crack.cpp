#include "password_crack.h"

//Set size is 36 characters and one blank character
float setSize = 36;
bool done = false;

int main() {
    char passwordStr[] = "aabca";

    int possibleLen = strlen(passwordStr);

    std::hash<std::string> ptr_hash;
    std::size_t password = ptr_hash(std::string(passwordStr));

    int numThreads = 4;

    struct timespec start, finish;
    double elapsed;

    /*
    printf("-Starting Non-Parallel Password Cracker-\n");

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
        if (strcmp(password, guess) == 0) {
          printf("Match Found Single!! \nLen: %d\tGuess: %s\n",currLen, guess);
          break;
        }
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time: %f\n", elapsed);
    */
    
    
    /*
    // Test crack function
    printf("-Testing function-\n");
    start = std::clock();

    struct params* args = new struct params;
    args->password = password;
    args->passLen = 5;
    args->totalThreads = numThreads;
    args->currThread = 1;

    char* pass_guess = (char*) crack((void*) args);

    printf("Returned Value: %s\n", pass_guess);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    printf("Time: %f\n", duration);

    */

    
    // Use Pthreads

    printf("-Starting Parallel PThread Cracker-\n");
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

void* crack(void* args){
  // Get arguments
  struct params *params = (struct params*) args;
  size_t password = params->password;
  int possLen = params->passLen;
  int totalThreads = params->totalThreads;
  int currThread = params->currThread;
  int count = 0;

  std::hash<std::string> ptr_hash;

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
      if (password == ptr_hash(std::string(guess))) {

        printf("Match Found Parallel!! \nLen: %d\tGuess: %s\t Iterations: %d\t Current Count: %d\n",currLen, guess, count, currChar);
        done = true;
        return (void*) guess;
      }
    }
  }
  //printf("Thread: %d Finished! Iterations: %d\n", currThread, count);
  return NULL;
}


char map(int convert){
  if (convert < 10) {
    return (char) convert + 48;
  } else {
    return (char) convert + 87;
  }
}
