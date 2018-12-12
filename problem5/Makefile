CC=nvcc
CFLAGS=-x cu
HEADER=include/password_crack.h

default: crack

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ crack

crack: 
	nvcc -x cu passwrod_crack.cpp -arch=compute_35 -o crack