#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helpers/helper.h"
#include "helpers/Timer.h"

// Host Matrix Addition.
void matrix_add_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

// Device Matrix Addition.
// Todo: add cuda kernel with name matrix_add_gpu

// Define our workload. You can play around with the matrix size to change the execution time.
constexpr int matrix_size = 32 * 512;
constexpr int matrix_elements = matrix_size * matrix_size;

int main()
{
    // Helper so we can track our execution time.
    timer timer;

    // Check which host & device we are using.
    helper::print_host_info();
    helper::print_device_info();

    // Allocate matrices on the host.
    std::cout << "\n" << "Creating random matrices: ";
    timer.start();
    std::vector<float> host_a(matrix_elements);
    std::vector<float> host_b(matrix_elements);
    std::vector<float> host_c(matrix_elements);
    // Fill matrices with random values. This will take a bit depending on the matrix_size.
    // host_c stays empty as it will be filled by the calculation.
    helper::random_fill(host_a, matrix_elements);
    helper::random_fill(host_b, matrix_elements);
    timer.stop();

    // Measure the execution speed on the host.
    std::cout << "Host Speed: ";
    timer.start();
    matrix_add_cpu(host_a, host_b, host_c, matrix_elements);
    timer.stop();

    // Todo: Allocate matrices on the device.

    // Todo: Copy the host data to device.

    // Determine the distribution on the device. Should not use more than 256 threads per block.
    constexpr int threads_per_block = 256;
    int blocks_per_grid = matrix_elements / threads_per_block;

    std::cout << "Device Speed: ";
    // Todo: Measure the execution speed on the device.

    // Todo: Copy the result from the device to the host.

    // Todo: Free the reserved memory on the device.

    return 0;
}
