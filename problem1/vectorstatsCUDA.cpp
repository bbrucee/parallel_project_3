#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// https://stackoverflow.com/questions/41353787/cuda-copying-arrays-to-gpu
double *CopyArrayToGPU(double *HostArray, int NumElements)
{
    int bytes = sizeof(double) * NumElements;
    void *DeviceArray;

    // Allocate memory on the GPU for array
    if (cudaMalloc(&DeviceArray, bytes) != cudaSuccess)
    {
        printf("CopyArrayToGPU(): Couldn't allocate mem for array on GPU.");
        return NULL;
    }

    // Copy the contents of the host array to the GPU
    if (cudaMemcpy(DeviceArray, HostArray, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("CopyArrayToGPU(): Couldn't copy host array to GPU.");
        cudaFree(DeviceArray);
        return NULL;
    }

    return DeviceArray;
}

/*
int vectorstatsCUDAtest()
{
	// Test function compares our function outputs to values computed using numpy externally
    double array[6] = {1.2342, 2.232421, 1.214124, 4.3252, 5.12314, 2.343241};
    int size = 6;
    printf("Homebrew max is %f\n", find_max(array, size));
    printf("Homebrew min is %f\n", find_min(array, size));
    printf("Homebrew mean is %f\n", find_mean(array, size));
    printf("Homebrew std is %f\n", find_std(array, size));
    printf("Expected max is %f\n", 5.12314);
    printf("Expected min is %f\n", 1.214124);
    printf("Expected mean is %f\n", 2.7453876666666663);
    printf("Expected std is %f\n", 1.4833984905493947);
    return 0; 
}
*/
int main()
{
	vectorstatsCUDAtest();
	return 0;
}