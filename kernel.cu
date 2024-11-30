#include <iostream>

#include "helpers/matrix_helper.h"
#include "helpers/Timer.h"

constexpr int matrix_size = 128 * 128;

void multiply_cpu(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c)
{
    for (unsigned long long i = 0; i < matrix_size; ++i)
    {
        for (unsigned long long j = 0; j < matrix_size; ++j)
        {
            c[i * j] = a[i * j] * b[i * j];
        }
    }
}

void main()
{
    // Helper objects
    matrix_helper matrix_helper;
    timer timer;

    // Create matrices in the host memory (CPU) - this will take a bit
    std::cout << "Creating random matrices: ";
    timer.start();
    const std::vector<double> a = matrix_helper.createRandomMatrix(matrix_size, matrix_size);
    const std::vector<double> b = matrix_helper.createRandomMatrix(matrix_size, matrix_size);
    std::vector<double> c = matrix_helper.createEmptyMatrix(matrix_size, matrix_size);
    timer.stop();

    // Timing on CPU
    std::cout << "CPU calculation: ";
    timer.start();
    multiply_cpu(a, b, c);
    timer.stop();
}
