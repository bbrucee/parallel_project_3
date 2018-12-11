#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#define N 1100

// https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf
double test_array[N];

// Pseudocode:
// Spawn threads for each element in the array
// 1. First half of threads compare themself with an element on the other half of the array and sets the max of the two
// 2. Half of those threads then compares itself to the thread on the other half of the the half and sets the max of the two
// 3. Repeat 
// 4. Maximum element is now in the 0th index and is written
// This is in general a reduction and a similar pseduocode is used for mean/min/std calculations


__global__ void find_maxKernel(double* input_array, double* array_max, int input_size)
{
	extern __shared__ double maximum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    maximum[tid] = input_array[i];
    __syncthreads();


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x && ((index+s) < input_size)){
   			if(maximum[index] < maximum[index+s])
   				maximum[index] = maximum[index+s];
   		}
   		__syncthreads();
   	}

	if (tid==0) array_max[blockIdx.x] = maximum[0];
}

extern double find_max(double* input_array, int input_size)
{
	int size = input_size*sizeof(double);
	double* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);

	double* d_B;
	cudaMalloc(&d_B, 1*sizeof(double));

	double max_value[1] = {0};
    find_maxKernel<<<1, input_size, input_size*sizeof(double)>>>(d_A, d_B, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(max_value, d_B, 1*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
  	cudaFree(d_B);

    return max_value[0];
}

__global__ void find_minKernel(double* input_array, double* array_min, int input_size)
{
	extern __shared__ double minimum[];

	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    minimum[tid] = input_array[i];
    __syncthreads();

   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x){
   			if((minimum[index] > minimum[index+s]) && ((index+s) < input_size)){
   				minimum[index] = minimum[index+s];
   			}
   		}
   		__syncthreads();
   	}
	if (tid==0) array_min[blockIdx.x] = minimum[0];
}

extern double find_min(double* input_array, int input_size)
{
	int size = input_size*sizeof(double);
	double* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);

	double* d_B;
	cudaMalloc(&d_B, 1*sizeof(double));

	double min_value[1] = {0};
    find_minKernel<<<1, input_size, input_size*sizeof(double)>>>(d_A, d_B, input_size);
	cudaDeviceSynchronize();

    cudaMemcpy(min_value, d_B, 1*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
  	cudaFree(d_B);

    return min_value[0];
}

__global__ void find_sumKernel(double* input_array, double* out_sum, int input_size)
{
	extern __shared__ double array_sum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    array_sum[tid] = input_array[i];
    __syncthreads();


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x  && ((index+s) < input_size)){
			array_sum[index] += array_sum[index+s];
   		}
   		__syncthreads();
   	}

	if (tid==0) out_sum[blockIdx.x] = array_sum[0];
}

extern double find_mean(double* input_array, int input_size)
{
	int size = input_size*sizeof(double);
	double* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);

	double* d_B;
	cudaMalloc(&d_B, 1*sizeof(double));

	double mean_value[1] = {0};
    find_sumKernel<<<1, input_size, input_size*sizeof(double)>>>(d_A, d_B, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(mean_value, d_B, 1*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
  	cudaFree(d_B);


    return mean_value[0]/input_size;
}


__global__ void find_squaresumKernel(double* input_array, double* out_sum, int input_size)
{
	extern __shared__ double array_sum[];
	// Each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;
    array_sum[tid] = input_array[i]*input_array[i];
    __syncthreads();


   	for(int s=1; s<blockDim.x; s*=2){
   		int index = 2*s*tid;
   		if(index < blockDim.x  && ((index+s) < input_size)){
			array_sum[index] += array_sum[index+s];
   		}
   		__syncthreads();
   	}

	if (tid==0) out_sum[blockIdx.x] = array_sum[0];
}


extern double find_std(double* input_array, int input_size)
{
	int size = input_size*sizeof(double);
	double* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);

	double* d_B;
	cudaMalloc(&d_B, 1*sizeof(double));

	double* d_C;
	cudaMalloc(&d_C, 1*sizeof(double));

	double mean_value[1] = {0};
	double squaresum_value[1] = {0};
    find_sumKernel<<<1, input_size, input_size*sizeof(double)>>>(d_A, d_B, input_size);
    cudaDeviceSynchronize();
    cudaMemcpy(mean_value, d_B, 1*sizeof(double), cudaMemcpyDeviceToHost);
 	find_squaresumKernel<<<1, input_size, input_size*sizeof(double)>>>(d_A, d_C, input_size);
    cudaDeviceSynchronize();
    cudaMemcpy(squaresum_value, d_C, 1*sizeof(double), cudaMemcpyDeviceToHost);
    mean_value[0] = mean_value[0]/input_size;
    squaresum_value[0] = squaresum_value[0]/input_size;

    cudaFree(d_A);
 	cudaFree(d_B);
  	cudaFree(d_C);

    return sqrt(squaresum_value[0] - mean_value[0]*mean_value[0]);
}



int vectorstatsCUDAtest1()
{
	printf("Running vectorstatsCUDAtest1() \n -------------------------- \n");
	// Test function compares our function outputs to values computed using numpy externally
    double array[6] = {1.2342, 2.232421, 1.214124, 4.3252, 5.12314, 2.343241};
    int size = 6;
    printf("Homebrew min is %f\n", find_min(array, size));
    printf("Homebrew max is %f\n", find_max(array, size));
    printf("Homebrew mean is %f\n", find_mean(array, size));
    printf("Homebrew std is %f\n", find_std(array, size));
    printf("Expected min is %f\n", 1.214124);
    printf("Expected max is %f\n", 5.12314);
    printf("Expected mean is %f\n", 2.7453876666666663);
    printf("Expected std is %f\n", 1.4833984905493947);
    return 0; 
}

int vectorstatsCUDAtest2()
{
	printf("Running vectorstatsCUDAtest2(), N is %d \n -------------------------- \n", N);	
	for(int i = 0; i < N; i++){
		double random_double = (double)rand() / RAND_MAX;
    	test_array[i] = random_double * (1000);
	}
    printf("Homebrew min is %f\n", find_min(test_array, N));
    printf("Homebrew max is %f\n", find_max(test_array, N));
    printf("Homebrew mean is %f\n", find_mean(test_array, N));
    printf("Homebrew std is %f\n", find_std(test_array, N));
    
    return 0; 
}

int main()
{
	vectorstatsCUDAtest1();
	vectorstatsCUDAtest2();
	return 0;
}