CC=nvcc
<<<<<<< HEAD
CFLAGS=-std=c++11 -x cu
=======
CFLAGS=-x cu
>>>>>>> bc77a5515c5ed2df938519db54fd53588d2df990
HEADER=include/password_crack.h

default: crack

clean:
	rm crack

crack:
	$(CC) $(CFLAGS) password_crack.cpp -arch=compute_35 -o crack
