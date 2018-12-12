#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

int A_size = 1000000;
int A[1000000];
int A_copy[1000000];


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
	int tid = threadIdx.x + threadIdx.y;
	printf("%d %d %d\n", threadIdx.x, threadIdx.y, tid);
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
	int size_root = int(sqrt(input_size));
	int size = input_size*sizeof(int);
	int* d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, input_array, size, cudaMemcpyHostToDevice);
    exclusive_scan_block<<<dim3(1,1), dim3(size_root, size_root)>>>(d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(input_array, d_A, input_size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
    return;
}

void find_repeats(int* input_array, int input_size)
{
	vector<int> B, C;
	for(int i = 0; i < input_size-1; i++){
		if(input_array[i] == input_array[i+1]) B.push_back(i);
	}
	if(B[0] != 0) C.push_back(input_array[0]);
	for(int j = 0; j < B.size()-1; j++){
		if(B[j] != B[j+1]) C.push_back(input_array[B[j]]);
	}	
}

bool exclusive_scan_additionTest()
{
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
		// if(test_array[i] - expected_output[i] != 0) return true;
	}
	return false;
}

int main()
{
	if(exclusive_scan_additionTest()){
		printf("exclusive_scan_additionTest() failed\n");
		return 0;
	}
	initialize_A();
	exclusive_scan_addition(A_copy, A_size);
	printf("It ran and possibly worked\n");
	// for(int i = 0; i < A_size; i ++){
	// 	printf("%d \n", A_copy[i]);
	// }
	return 0;
}