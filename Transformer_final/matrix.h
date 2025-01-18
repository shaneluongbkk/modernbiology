#include "main.h"
#pragma once

// Allocate a 2D matrix dynamically
float **allocateMatrix(int rows, int cols)
{
    float **matrix = new float *[rows];
    for (int i = 0; i < rows; ++i)
    {
        matrix[i] = new float[cols]();
    }
    return matrix;
}

// Deallocate a 2D matrix
void deallocateMatrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Randomize a matrix
void randomizeMatrix(float **matrix, int rows, int cols)
{
    float stddev = sqrt(2.0f / rows);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = dist(gen);
        }
    }
}

// Multiply two matrices
float **multiplyMatrices(float **A, int A_rows, int A_cols, float **B, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Allocate the result matrix
    float **result = allocateMatrix(A_rows, B_cols);

    // Matrix multiplication
    for (int i = 0; i < A_rows; ++i)
    {
        for (int j = 0; j < B_cols; ++j)
        {
            for (int k = 0; k < A_cols; ++k)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// Print a matrix
void printMatrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            fo << matrix[i][j] << " ";
        }
        fo << endl;
    }
}

// divide a matrix by a scalar
void divideMatrixByScalar(float **matrix, int rows, int cols, float scalar)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] /= scalar;
        }
    }
}