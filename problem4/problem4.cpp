#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>

using namespace std;

//Only works for arrays smaller than threadsperblock because our CUDA stuff only scans one block
int A_size = 1000;
int A[1000];
int A_copy[1000];
int repeats[1000];
int repeats_index[1000];

int num_threads = 1000;
int num_blocks = 0;

void set_blocks(long int input_size)
{
	num_blocks = 0;
	while(num_threads*num_blocks < input_size){
		num_blocks++;
	}
}


void initialize_A()
{
	for (int i = 0; i < A_size; i++) {
		int random_integer = rand() % 100;
		A[i] = random_integer;
		A_copy[i] = random_integer;
	}
	return;
}

__device__ int exclusive_scan_warp(int* input_array)
{
	int tid = threadIdx.x;
	int lane = tid & 31;

	if(lane >= 1)  input_array[tid] = input_array[tid-1] + input_array[tid];
	if(lane >= 2)  input_array[tid] = input_array[tid-2] + input_array[tid];
	if(lane >= 4)  input_array[tid] = input_array[tid-4] + input_array[tid];
	if(lane >= 8)  input_array[tid] = input_array[tid-8] + input_array[tid];
	if(lane >= 16) input_array[tid] = input_array[tid-16] + input_array[tid];

	return (lane > 0) ? input_array[tid-1] : 0;
}

__global__ void exclusive_scan_block(int* input_array)
{
	int tid = threadIdx.x;
	int lane = tid & 31;
	int wid = tid >> 5;

	int value = exclusive_scan_warp(input_array);

	if(lane == 31) input_array[wid] = input_array[tid];
	__syncthreads();
	if(wid == 0) exclusive_scan_warp(input_array);
	__syncthreads();
	if(wid > 0) value = input_array[wid-1] + value;
	__syncthreads();

	input_array[tid] = value;
}

extern void exclusive_scan_addition(int* input_array, int input_size)
{
	set_blocks(input_size);
	int size = input_size*sizeof(int);
	int* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);
    exclusive_scan_block<<<num_blocks, num_threads>>>(d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(input_array, d_A, input_size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
    return;
}

void find_repeats(int* input_array, int input_size)
{
	for(int i = 0; i<input_size-1; i++){
		if(input_array[i] == input_array[i+1])
			repeats[i] = 1;
		else
			repeats[i] = 0;
	}

}

void find_repeats_index(int* repeat_array, int* scanned_array, int input_size)
{
	for(int i = 0; i < input_size; i++){
		if(repeat_array[i] == 1)
			repeats_index[scanned_array[i]] = i;
	}

}

void exclusive_scan_additionTest1()
{
	printf("Running exclusive_scan_additionTest1()\n -------------------------- \n");  
	int test_array[5], expected_output[5];
	test_array[0] = 1;
	test_array[1] = 4;
	test_array[2] = 6;
	test_array[3] = 8;
	test_array[4] = 2;
	expected_output[0] = 0;
	expected_output[1] = 1;
	expected_output[2] = 5;
	expected_output[3] = 11;
	expected_output[4] = 19;
	exclusive_scan_addition(test_array, 5);
	for(int i = 0; i < 5; i++){
		printf("test_array[%d] = %d expected %d \n", i, test_array[i], expected_output[i]);
	}
	printf(" -------------------------- \n");
}

void exclusive_scan_additionTest2()
{
	printf("Running exclusive_scan_additionTest2()\n -------------------------- \n");  
	initialize_A();
	exclusive_scan_addition(A, A_size);
	for (int j=0; j<=A_size-1; j++) {
	    if (j > 0)
	        A_copy[j] = A_copy[j] + A_copy[j-1];
	}
	A_copy[0] = 0;
	for(int i = 0; i < 20; i++){
		printf("test_array[%d] = %d expected %d \n", i, A[i], A_copy[i]);
	}
	printf(" -------------------------- \n");
}

void find_repeatsTest()
{
	printf("Running find_repeatsTest()\n -------------------------- \n");  
	int test_array[10], expected_output[10];
	test_array[0] = 1;
	test_array[1] = 2;
	test_array[2] = 2;
	test_array[3] = 1;
	test_array[4] = 1;
	test_array[5] = 1;
	test_array[6] = 3;
	test_array[7] = 5;
	test_array[8] = 3;
	test_array[9] = 3;
	expected_output[0] = 0;
	expected_output[1] = 1;
	expected_output[2] = 0;
	expected_output[3] = 1;
	expected_output[4] = 1;
	expected_output[5] = 0;
	expected_output[6] = 0;
	expected_output[7] = 0;
	expected_output[8] = 1;
	expected_output[9] = 0;
	find_repeats(test_array, 10);
	for(int i = 0; i < 10; i++){
		printf("test_array[%d] = %d expected %d \n", i, repeats[i], expected_output[i]);
	}
	printf(" -------------------------- \n");
}

void find_repeats_indexTest()
{
	printf("Running find_repeats_indexTest()\n -------------------------- \n");  
	int test_array[10], expected_output[4];
	test_array[0] = 1;
	test_array[1] = 2;
	test_array[2] = 2;
	test_array[3] = 1;
	test_array[4] = 1;
	test_array[5] = 1;
	test_array[6] = 3;
	test_array[7] = 5;
	test_array[8] = 3;
	test_array[9] = 3;
	expected_output[0] = 1;
	expected_output[1] = 3;
	expected_output[2] = 4;
	expected_output[3] = 8;
	find_repeats(test_array, 10); //repeats array now has repeats
	exclusive_scan_addition(test_array, 10); //test_array now has exclusive scan add of repeats
	find_repeats_index(repeats, test_array, 10);
	for(int i = 0; i < 4; i++){
		printf("index_array[%d] = %d expected %d \n", i, repeats_index[i], expected_output[i]);
	}
	printf(" -------------------------- \n");
}

int main()
{
	exclusive_scan_additionTest1();
	exclusive_scan_additionTest2();
	find_repeatsTest();
	find_repeats_indexTest();
	return 0;
}