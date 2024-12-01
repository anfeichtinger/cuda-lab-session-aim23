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
__global__ void matrix_add_gpu(const float* a, const float* b, float* c, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

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

    // Allocate matrices on the device.
    std::cout << "Device Speed with overhead: ";
    timer.start();
    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, matrix_elements * sizeof(float));
    cudaMalloc(&device_b, matrix_elements * sizeof(float));
    cudaMalloc(&device_c, matrix_elements * sizeof(float));

    // Copy the actual data to device.
    cudaMemcpy(device_a, &host_a, matrix_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, &host_b, matrix_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Determine the distribution on the device. Should not use more than 256 threads per block.
    int threads_per_block = 256;
    int blocks_per_grid = matrix_elements / threads_per_block;

    // The execution speed on the device with memory copying overhead.
    matrix_add_gpu<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, matrix_elements);
    cudaDeviceSynchronize();
    cudaMemcpy(&host_c, device_c, matrix_elements, cudaMemcpyDeviceToHost);
    timer.stop();

    // The raw execution speed on the device.
    std::cout << "Device Speed: ";
    timer.start();
    matrix_add_gpu<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, matrix_elements);
    cudaDeviceSynchronize();
    timer.stop();

    // Freeing the reserved memory on the device.
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
