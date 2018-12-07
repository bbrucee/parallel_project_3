#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>

// Pseudocode:
// Spawn threads for each element in the array
// 1. First half of threads compare themself with an element on the other half of the array and sets the max of the two
// 2. Half of those threads then compares itself to the thread on the other half of the the half and sets the max of the two
// 3. Repeat 
// 4. Maximum element is now in the 0th index and is written
// This is in general a reduction and a similar pseduocode is used for mean/min/std calculations


__global__ void find_maxKernel(double* input_array, double array_max)
{
	extern __shared__ double maximum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    maximum[tid] = input_array[i];
    __syncthreads;


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x){
   			if(maximum[index] < maximum[index+s])
   				maximum[index] = maximum[index+s];
   		}
   		__syncthreads;
   	}

	if (tid==0) array_max = maximum[0];
}

extern double find_max(double* input_array, int input_size)
{
	double max_value = 0;
    find_maxKernel<<<1, input_size>>>(input_array, max_value);
    cudaDeviceSynchronize();
    return max_value;
}

__global__ void find_minKernel(double* input_array, double array_min)
{
	extern __shared__ double minimum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    minimum[tid] = input_array[i];
    __syncthreads;


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x){
   			if(minimum[index] < minimum[index+s])
   				minimum[index] = minimum[index+s];
   		}
   		__syncthreads;
   	}

	if (tid==0) array_min = minimum[0];
}

extern double find_min(double* input_array, int input_size)
{
	double min_value = 0;
    find_minKernel<<<1, input_size>>>(input_array, min_value);
    cudaDeviceSynchronize();
    return min_value;
}

__global__ void find_sumKernel(double* input_array, double out_sum)
{
	extern __shared__ double array_sum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    array_sum[tid] = input_array[i];
    __syncthreads;


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x){
			array_sum[index] += array_sum[index+s];
   		}
   		__syncthreads;
   	}

	if (tid==0) out_sum = array_sum[0];
}

extern double find_mean(double* input_array, int input_size)
{
	double mean_value = 0;
    find_sumKernel<<<1, input_size>>>(input_array, mean_value);
    cudaDeviceSynchronize();
    return mean_value/input_size;
}


__global__ void find_squaresumKernel(double* input_array, double out_sum)
{
	extern __shared__ double array_sum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    array_sum[tid] = input_array[i]*input_array[i];
    __syncthreads;


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x){
			array_sum[index] += array_sum[index+s]*array_sum[index+s];
   		}
   		__syncthreads;
   	}

	if (tid==0) out_sum = array_sum[0];
}


extern double find_std(double* input_array, int input_size)
{
	double mean_value = 0;
	double squaresum_value = 0;
    find_sumKernel<<<1, input_size>>>(input_array, mean_value);
 	find_squaresumKernel<<1, input_size>>(input_array, squaresum_value);
    cudaDeviceSynchronize();
    mean_value = mean_value/input_size;
    squaresum_value = squaresum_value/input_size;
    return sqrt(squaresum_value - mean_value*mean_value);
}



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

int main()
{
	vectorstatsCUDAtest();
	return 0;
}