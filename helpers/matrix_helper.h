#pragma once
#include <iostream>
#include <vector>

class matrix_helper
{
public:
    // Constructor to initialize the random seed
    matrix_helper()
    {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    // Creates an empty matrix of the given size
    std::vector<double> createEmptyMatrix(const unsigned long long rows, const unsigned long long cols)
    {
        return std::vector<double>(rows * cols);
    }

    // Creates a random matrix of the given size with values in the range [min, max]
    std::vector<double> createRandomMatrix(const unsigned long long rows, const unsigned long long cols)
    {
        std::vector<double> matrix(rows * cols);
        for (double& val : matrix)
        {
            val = random_value(0.0, 1.0); // Use a fast random function
        }
        return matrix;
    }

private:
    // Generates a random value in the range [min, max]
    static double random_value(const double min, const double max)
    {
        return min + static_cast<double>(std::rand()) / RAND_MAX * (max - min);
    }
};
