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

int main() {
    char passwordStr[] = "aabca";

    int possibleLen = strlen(passwordStr);

    hash<string> ptr_hash;
    size_t password = ptr_hash(string(passwordStr));    
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
    
    return 0;
}
