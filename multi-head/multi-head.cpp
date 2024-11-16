#include <iostream>
#include <cstdlib>
#include <stdexcept>

using namespace std;

const int EMBEDDING_DIM = 16;
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;

// Allocate a 2D matrix dynamically
float** allocateMatrix(int rows, int cols) {
    float** matrix = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new float[cols]();
    }
    return matrix;
}

// Deallocate a 2D matrix
void deallocateMatrix(float** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Randomize a matrix
void randomizeMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Multiply two matrices
float** multiplyMatrices(float** A, int A_rows, int A_cols, float** B, int B_rows, int B_cols) {
    if (A_cols != B_rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Allocate the result matrix
    float** result = allocateMatrix(A_rows, B_cols);

    // Matrix multiplication
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// Print a matrix
void printMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Generate multihead QKV matrices
void generateMultiheadQKV(float** input_embedding, int input_rows, int input_cols, 
                          float*** Q_heads, float*** K_heads, float*** V_heads) {
    float*** W_Q_heads = new float**[NUM_HEADS];
    float*** W_K_heads = new float**[NUM_HEADS];
    float*** W_V_heads = new float**[NUM_HEADS];

    for (int h = 0; h < NUM_HEADS; ++h) {
        // Allocate W_Q, W_K, W_V matrices
        W_Q_heads[h] = allocateMatrix(input_cols, HEAD_DIM);
        W_K_heads[h] = allocateMatrix(input_cols, HEAD_DIM);
        W_V_heads[h] = allocateMatrix(input_cols, HEAD_DIM);

        // Randomize weight matrices
        randomizeMatrix(W_Q_heads[h], input_cols, HEAD_DIM);
        randomizeMatrix(W_K_heads[h], input_cols, HEAD_DIM);
        randomizeMatrix(W_V_heads[h], input_cols, HEAD_DIM);

        // Allocate Q, K, V matrices
        Q_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_Q_heads[h], input_cols, HEAD_DIM);
        K_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_K_heads[h], input_cols, HEAD_DIM);
        V_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_V_heads[h], input_cols, HEAD_DIM);

        // Deallocate weight matrices after use
        deallocateMatrix(W_Q_heads[h], input_cols);
        deallocateMatrix(W_K_heads[h], input_cols);
        deallocateMatrix(W_V_heads[h], input_cols);
    }

    // Deallocate the head matrices
    delete[] W_Q_heads;
    delete[] W_K_heads;
    delete[] W_V_heads;
}

int main() {
    srand(42);

    // Create and randomize the input embedding matrix
    int input_rows = 10;
    float** input_embedding = allocateMatrix(input_rows, EMBEDDING_DIM);
    randomizeMatrix(input_embedding, input_rows, EMBEDDING_DIM);

    // Store Q, K, V heads
    float*** Q_heads = new float**[NUM_HEADS];
    float*** K_heads = new float**[NUM_HEADS];
    float*** V_heads = new float**[NUM_HEADS];

    // Generate the Q, K, V heads from the input embedding
    generateMultiheadQKV(input_embedding, input_rows, EMBEDDING_DIM, Q_heads, K_heads, V_heads);

    // Print Q, K, V heads
    cout << "Q_heads:" << endl;
    for (int h = 0; h < NUM_HEADS; ++h) {
        cout << "Q_head " << h + 1 << ":" << endl;
        printMatrix(Q_heads[h], input_rows, HEAD_DIM);
        cout << endl;
    }

    cout << "K_heads:" << endl;
    for (int h = 0; h < NUM_HEADS; ++h) {
        cout << "K_head " << h + 1 << ":" << endl;
        printMatrix(K_heads[h], input_rows, HEAD_DIM);
        cout << endl;
    }

    cout << "V_heads:" << endl;
    for (int h = 0; h < NUM_HEADS; ++h) {
        cout << "V_head " << h + 1 << ":" << endl;
        printMatrix(V_heads[h], input_rows, HEAD_DIM);
        cout << endl;
    }

    // Clean up dynamically allocated memory
    deallocateMatrix(input_embedding, input_rows);

    for (int h = 0; h < NUM_HEADS; ++h) {
        deallocateMatrix(Q_heads[h], input_rows);
        deallocateMatrix(K_heads[h], input_rows);
        deallocateMatrix(V_heads[h], input_rows);
    }

    delete[] Q_heads;
    delete[] K_heads;
    delete[] V_heads;

    return 0;
}
