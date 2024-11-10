#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

// Define Matrix and Array types
using Matrix = std::vector<std::vector<double>>;
using Array = std::vector<double>;

// Linear function for feedforward layer with bias
Matrix Linear(const Matrix &input, const Matrix &W, double b)
{
    size_t M = input.size(); // Number of rows in input
    size_t N = W[0].size();  // Number of columns in W (output size)
    size_t P = W.size();     // Number of rows in W (input size)

    Matrix output(M, std::vector<double>(N, b)); // Initialize output with bias
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < P; k++)
                output[i][j] += input[i][k] * W[k][j];
    return output;
}

// FeedForward function, returning a single Matrix
Matrix FeedForward(const Matrix &input,
                   const Matrix &W1, const Array &b1,
                   const Matrix &W2, const Array &b2,
                   bool use_bias)
{
    size_t M = input.size();
    size_t d_ff = W1[0].size();    // Number of columns in W1 (output dimension of first layer)
    size_t d_model = W2[0].size(); // Number of columns in W2 (output dimension of second layer)

    // Apply the first linear transformation and add bias
    Matrix h = Linear(input, W1, use_bias ? b1[0] : 0.0);

    // Apply ReLU activation function
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < d_ff; j++)
            h[i][j] = std::max(0.0, h[i][j]);

    // Apply the second linear transformation and add bias
    Matrix output = Linear(h, W2, use_bias ? b2[0] : 0.0);

    return output; // Returning a single Matrix
}

// Utility function to create a random matrix
Matrix createRandomMatrix(size_t rows, size_t cols)
{
    Matrix mat(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            mat[i][j] = (rand() % 100) / 100.0; // Random number between 0 and 1
    return mat;
}

// Utility function to create a random vector
Array createRandomVector(size_t size)
{
    Array vec(size);
    for (size_t i = 0; i < size; i++)
        vec[i] = (rand() % 100) / 100.0; // Random number between 0 and 1
    return vec;
}

// int main()
// {
//     // Define weights and biases
//     Matrix W1 = {{0.6, 0.1, 0.2, 0.7},
//                  {0.4, 0.3, 0.7, 0.3},
//                  {0.8, 0.4, 0.9, 0.4}};

//     Matrix W2 = {
//         {0.1, 0.3, 0.5},
//         {0.5, 0.5, 0.6},
//         {0.6, 0.1, 0.8},
//         {0.7, 0.8, 0.1}};

//     // Biases must match the number of output neurons for W1 and W2
//     Array b1 = {0.3, 0.3, 0.3, 0.3}; // Biases for W1
//     Array b2 = {0.2, 0.2, 0.2};      // Biases for W2

//     // Input matrices
//     Matrix input = {
//         {0.5, 0.7, 0.2},
//         {0.4, 0.2, 0.1},
//         {0.9, 0.4, 0.8}};

//     // Call FeedForward
//     bool use_bias = true; // Set to true to use biases
//     auto output = FeedForward(input, W1, b1, W2, b2, use_bias);

//     // Print output
//     for (const auto &mat : output)
//     {
//         for (const auto &val : mat)
//             cout << val << " ";
//         cout << endl;
//     }

//     return 0;
// }
