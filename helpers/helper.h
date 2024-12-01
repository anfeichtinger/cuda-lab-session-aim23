#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <windows.h>
#include <intrin.h>

class helper
{
public:
    // Fills a square matrix of the given size with random float values.
    static void random_fill(std::vector<float>& vec, const int size)
    {
        for (int i = 0; i < size; ++i)
        {
            vec[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Print which device we are currently using. You can ignore this for now.
    static void print_host_info()
    {
        std::cout << "Using host: ";

        int cpu_info[4] = {-1};
        char cpu_brand_string[0x40];
        __cpuid(cpu_info, 0x80000002);
        memcpy(cpu_brand_string, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000003);
        memcpy(cpu_brand_string + 16, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000004);
        memcpy(cpu_brand_string + 32, cpu_info, sizeof(cpu_info));

        std::cout << std::string(cpu_brand_string) << '\n';
    }

    // Print which device we are currently using. You can ignore this for now.
    static void print_device_info()
    {
        std::cout << "Using device: ";

        int device_id;
        cudaDeviceProp device_prop;

        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&device_prop, device_id);

        std::cout << device_prop.name << '\n';
    }
};
