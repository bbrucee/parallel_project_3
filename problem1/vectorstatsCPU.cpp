#include <stdio.h>
#include <math.h>
#include <limits>


double find_maxCPU(double* input_array, int input_size)
{
    if(input_size == 0)
        return 0;
    double max_value =  std::numeric_limits<double>::min();
    for(int i = 0; i<input_size; i++){
        if(input_array[i] > max_value)
            max_value = input_array[i];
    }
    return max_value;
}
    
double find_minCPU(double* input_array, int input_size)
{
    if(input_size == 0)
        return 0;
    double min_value = std::numeric_limits<double>::max();
    for(int i = 0; i<input_size; i++){
        if(input_array[i] < min_value)
            min_value = input_array[i];
    }
    return min_value;

}

double find_meanCPU(double* input_array, int input_size)
{
    if(input_size == 0)
        return 0;
    double mean_value = 0;
    for(int i = 0; i<input_size; i++){
        mean_value += input_array[i];
    }
    mean_value = mean_value/input_size;
    return mean_value;
}
    
double find_stdCPU(double* input_array, int input_size)
{
    if(input_size == 0)
        return 0;
    double mean_value = 0;
    double square_sum = 0;
    for(int i = 0; i<input_size; i++){
        mean_value += input_array[i];
        square_sum += input_array[i]*input_array[i];
    }
    mean_value = mean_value/input_size;
    square_sum = sqrt((square_sum/input_size) - (mean_value*mean_value));
    return square_sum;
}

int vectorstatsCPUtest()
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
    vectorstatsCPUtest();
    return 0;
}