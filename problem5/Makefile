CC=nvcc
CFLAGS=-x cu
HEADER=include/password_crack.h

default: crack

clean:
	rm crack

crack:
	$(CC) $(CFLAGS) password_crack.cpp -arch=compute_35 -o crack
