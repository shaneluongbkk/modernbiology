#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cmath>

using namespace std;

const int EMBEDDING_DIM = 8; // 512
const int MAX_POSITION = 1000;
const int NUM_HEADS = 1;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
const float LEARNING_RATE = 0.1f;

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

// Forward function: generates Q, K, V and computes attention scores
void forward(float** input_embedding, int input_rows, int input_cols, 
             float*** Q_heads, float*** K_heads, float*** V_heads, 
             float*** W_Q_heads, float*** W_K_heads, float*** W_V_heads, 
             float*** attention_scores) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        // Allocate weight matrices
        W_Q_heads[h] = allocateMatrix(input_cols, HEAD_DIM);
        W_K_heads[h] = allocateMatrix(input_cols, HEAD_DIM);
        W_V_heads[h] = allocateMatrix(input_cols, HEAD_DIM);

        // Randomize weight matrices
        randomizeMatrix(W_Q_heads[h], input_cols, HEAD_DIM);
        randomizeMatrix(W_K_heads[h], input_cols, HEAD_DIM);
        randomizeMatrix(W_V_heads[h], input_cols, HEAD_DIM);

        // Generate Q, K, V matrices
        Q_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_Q_heads[h], input_cols, HEAD_DIM);
        K_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_K_heads[h], input_cols, HEAD_DIM);
        V_heads[h] = multiplyMatrices(input_embedding, input_rows, input_cols, W_V_heads[h], input_cols, HEAD_DIM);

        // Compute attention scores: Q * K^T
        float** K_transpose = allocateMatrix(HEAD_DIM, input_rows);
        for (int i = 0; i < HEAD_DIM; ++i) {
            for (int j = 0; j < input_rows; ++j) {
                K_transpose[i][j] = K_heads[h][j][i];
            }
        }
        attention_scores[h] = multiplyMatrices(Q_heads[h], input_rows, HEAD_DIM, K_transpose, HEAD_DIM, input_rows);

        // Clean up temporary K_transpose
        deallocateMatrix(K_transpose, HEAD_DIM);
    }
}

// Backward function: computes gradients and updates weights
void backward(float** input_embedding, int input_rows, int input_cols, 
              float*** Q_heads, float*** K_heads, float*** V_heads, 
              float*** W_Q_heads, float*** W_K_heads, float*** W_V_heads, 
              float*** attention_scores, float*** d_attention_scores, 
              float*** d_V) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        // Compute gradients with respect to Q and K
        float** d_Q = allocateMatrix(input_rows, HEAD_DIM);
        float** d_K = allocateMatrix(input_rows, HEAD_DIM);

        // Backpropagate through the attention score:
        // dQ = d_attention_scores * K
        // dK = (d_attention_scores)^T * Q
        for (int i = 0; i < input_rows; ++i) {
            for (int j = 0; j < input_rows; ++j) {
                for (int k = 0; k < HEAD_DIM; ++k) {
                    d_Q[i][k] += d_attention_scores[h][i][j] * K_heads[h][j][k];
                    d_K[i][k] += d_attention_scores[h][j][i] * Q_heads[h][j][k];
                }
            }
        }

        // Compute gradients for W_Q, W_K, and W_V
        float** d_W_Q = allocateMatrix(input_cols, HEAD_DIM);
        float** d_W_K = allocateMatrix(input_cols, HEAD_DIM);
        float** d_W_V = allocateMatrix(input_cols, HEAD_DIM);

        // dW_Q = input_embedding^T * d_Q
        for (int i = 0; i < input_cols; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j) {
                for (int k = 0; k < input_rows; ++k) {
                    d_W_Q[i][j] += input_embedding[k][i] * d_Q[k][j];
                }
            }
        }

        // dW_K = input_embedding^T * d_K
        for (int i = 0; i < input_cols; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j) {
                for (int k = 0; k < input_rows; ++k) {
                    d_W_K[i][j] += input_embedding[k][i] * d_K[k][j];
                }
            }
        }

        // dW_V = input_embedding^T * d_V[h]
        for (int i = 0; i < input_cols; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j) {
                for (int k = 0; k < input_rows; ++k) {
                    d_W_V[i][j] += input_embedding[k][i] * d_V[h][k][j];
                }
            }
        }

        // Update weights using the gradients
        for (int i = 0; i < input_cols; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j) {
                W_Q_heads[h][i][j] -= LEARNING_RATE * d_W_Q[i][j];
                W_K_heads[h][i][j] -= LEARNING_RATE * d_W_K[i][j];
                W_V_heads[h][i][j] -= LEARNING_RATE * d_W_V[i][j];
            }
        }

        // Clean up gradients
        deallocateMatrix(d_Q, input_rows);
        deallocateMatrix(d_K, input_rows);
        deallocateMatrix(d_W_Q, input_cols);
        deallocateMatrix(d_W_K, input_cols);
        deallocateMatrix(d_W_V, input_cols);
    }
}



int main() {
    srand(42);

    // Create and randomize the input embedding matrix
    int input_rows = 10;
    float** input_embedding = allocateMatrix(input_rows, EMBEDDING_DIM);
    randomizeMatrix(input_embedding, input_rows, EMBEDDING_DIM);

    // Arrays to store Q, K, V heads and their weight matrices
    float*** Q_heads = new float**[NUM_HEADS];
    float*** K_heads = new float**[NUM_HEADS];
    float*** V_heads = new float**[NUM_HEADS];
    float*** W_Q_heads = new float**[NUM_HEADS];
    float*** W_K_heads = new float**[NUM_HEADS];
    float*** W_V_heads = new float**[NUM_HEADS];
    
    float*** attention_scores = new float**[NUM_HEADS];
    float*** d_attention_scores = new float**[NUM_HEADS]; 
    float*** d_V = new float**[NUM_HEADS];

    forward(input_embedding, input_rows, EMBEDDING_DIM, Q_heads, K_heads, V_heads, 
            W_Q_heads, W_K_heads, W_V_heads, attention_scores);


    for (int h = 0; h < NUM_HEADS; ++h) {
        d_attention_scores[h] = allocateMatrix(input_rows, input_rows);
        randomizeMatrix(d_attention_scores[h], input_rows, input_rows); // Random values
    }

    // Backward pass
    for (int i = 0; i < NUM_HEADS; i++)
    {
        d_V[i] = allocateMatrix(input_rows, HEAD_DIM);
        randomizeMatrix(d_V[i], input_rows, HEAD_DIM);
    }
    
    
    cout << "W_Q_heads before backward\n";
    for (int i = 0; i < NUM_HEADS; i++)
    {
        cout << "W_Q_heads " << i << "th\n";
        printMatrix(W_Q_heads[i], EMBEDDING_DIM, HEAD_DIM);
    }
    backward(input_embedding, input_rows, EMBEDDING_DIM, Q_heads, K_heads, V_heads, 
             W_Q_heads, W_K_heads, W_V_heads, attention_scores, d_attention_scores, d_V);
    
    cout << "W_Q_heads after backward\n";
    for (int i = 0; i < NUM_HEADS; i++)
    {
        cout << "W_Q_heads " << i << "th\n";
        printMatrix(W_Q_heads[i], EMBEDDING_DIM, HEAD_DIM);
    }
    
    
    deallocateMatrix(input_embedding, input_rows);

    for (int h = 0; h < NUM_HEADS; ++h) {
        deallocateMatrix(Q_heads[h], input_rows);
        deallocateMatrix(K_heads[h], input_rows);
        deallocateMatrix(V_heads[h], input_rows);
        deallocateMatrix(W_Q_heads[h], EMBEDDING_DIM);
        deallocateMatrix(W_K_heads[h], EMBEDDING_DIM);
        deallocateMatrix(W_V_heads[h], EMBEDDING_DIM);
        deallocateMatrix(attention_scores[h], input_rows);
        deallocateMatrix(d_attention_scores[h], input_rows);
    }

    delete[] Q_heads;
    delete[] K_heads;
    delete[] V_heads;
    delete[] W_Q_heads;
    delete[] W_K_heads;
    delete[] W_V_heads;
    delete[] attention_scores;
    delete[] d_attention_scores;
    return 0;
}
