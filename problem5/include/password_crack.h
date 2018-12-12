#ifndef password_crack_h
#define password_crack_h

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

char map(int convert);
void* crack(void* args);
struct params {
	size_t password;
	int passLen;
	int totalThreads;
	int currThread;
};

#endif