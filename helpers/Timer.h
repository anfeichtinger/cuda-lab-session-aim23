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
        const auto stop_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time_).count();

        if (duration >= 1000)
        {
            std::cout << duration / 1000.0 << " seconds" << "\n";
        }
        else
        {
            std::cout << duration << " milliseconds" << "\n";
        }
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};
