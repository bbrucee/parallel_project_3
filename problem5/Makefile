CC=nvcc
CFLAGS=-std=c++11 -x cu
HEADER=include/password_crack.h

default: crack

clean:
	rm crack

crack:
	$(CC) $(CFLAGS) password_crack.cpp -arch=compute_35 -o crack
