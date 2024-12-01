#pragma once
#include <chrono>
#include <iostream>

class timer
{
public:
    // Starts the timer
    void start()
    {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    // Stops the timer and prints the elapsed time
    void stop() const
    {
        const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time_;
        std::cout << elapsed.count() << " seconds" << '\n';
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};
