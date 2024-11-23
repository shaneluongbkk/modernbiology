#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

const int EMBEDDING_DIM = 200; // 512
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8; // Changed to match the second code
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
const float LEARNING_RATE = 0.1f;

// Error handling macro for CUDA calls
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Allocate a matrix (2D array) dynamically on the host
float** allocateMatrix(int rows, int cols) {
    float** mat = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new float[cols]();
    }
    return mat;
}

// Deallocate matrix on the host
void deallocateMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}

// Randomize a matrix with values between 0 and 1 on the host
void randomizeMatrix(float** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Print a matrix on the host
void printMatrix(float** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

void flattenMatrix(float** src, float* dest, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i * cols + j] = src[i][j];
        }
    }
}

void unflattenMatrix(const float* src, float** dest, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] = src[i * cols + j];
        }
    }
}

void copyMatrix(float** source, float** destination, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            destination[i][j] = source[i][j];
        }
    }
}

// Transpose matrix on the host
float** transposeMatrix(float** matrix, int rows, int cols) {
    float** transposed = allocateMatrix(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Softmax function for each row of the matrix on the host
void softmax(float** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += exp(mat[i][j]);
        }
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = exp(mat[i][j]) / sum;
        }
    }
}

// Create a causal mask matrix on the host
float** createCausalMask(int num_tokens) {
    float** mask_matrix = allocateMatrix(num_tokens, num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_tokens; ++j) {
            if (j > i)
                mask_matrix[i][j] = -numeric_limits<float>::infinity();
            else
                mask_matrix[i][j] = 0.0f;
        }
    }
    return mask_matrix;
}

// CUDA Kernel for matrix multiplication
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// Multiply two matrices (A * B) using CUDA
float** multiplyMatricesCUDA(float** A, int A_rows, int A_cols, float** B, int B_rows, int B_cols) {
    if (A_cols != B_rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    float* d_A, *d_B, *d_C;
    float** C = allocateMatrix(A_rows, B_cols);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, A_rows * A_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, B_rows * B_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, A_rows * B_cols * sizeof(float)));

    // Flatten matrices and copy to device
    float* flat_A = new float[A_rows * A_cols];
    float* flat_B = new float[B_rows * B_cols];
    flattenMatrix(A, flat_A, A_rows, A_cols);
    flattenMatrix(B, flat_B, B_rows, B_cols);
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, flat_B, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((B_cols + blockDim.x - 1) / blockDim.x, (A_rows + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    float* flat_C = new float[A_rows * B_cols];
    CUDA_CHECK(cudaMemcpy(flat_C, d_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost));
    unflattenMatrix(flat_C, C, A_rows, B_cols);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] flat_A;
    delete[] flat_B;
    delete[] flat_C;

    return C;
}
float** scaledAttention(float** Q, int Q_rows, int Q_cols, float** K, int K_rows, int K_cols) {
    // Use CUDA for matrix multiplication
    return multiplyMatricesCUDA(Q, Q_rows, Q_cols, K, K_rows, K_cols);
}

// Apply attention weights to the value matrix
float** applyAttentionWeights(float** attention_weights, int AW_rows, int AW_cols, float** V, int V_rows, int V_cols) {
    // Use CUDA for matrix multiplication
    return multiplyMatricesCUDA(attention_weights, AW_rows, AW_cols, V, V_rows, V_cols);
}

// Concatenate matrices along the last axis
float** concatenate_matrices(float*** matrices, int num_matrices, int rows, int total_cols) {
    float** result = allocateMatrix(rows, total_cols);
    int col_offset = 0;

    for (int m = 0; m < num_matrices; ++m) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < total_cols / num_matrices; ++j) {
                result[i][col_offset + j] = matrices[m][i][j];
            }
        }
        col_offset += total_cols / num_matrices;
    }

    return result;
}

// Generate multi-head Q, K, V matrices from input embedding
void generate_multihead_qkv(float** input_embedding, int num_tokens, int embedding_dim, float*** Q_heads, float*** K_heads, float*** V_heads) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        Q_heads[h] = multiplyMatricesCUDA(input_embedding, num_tokens, embedding_dim, Q_heads[h], embedding_dim, HEAD_DIM);
        K_heads[h] = multiplyMatricesCUDA(input_embedding, num_tokens, embedding_dim, K_heads[h], embedding_dim, HEAD_DIM);
        V_heads[h] = multiplyMatricesCUDA(input_embedding, num_tokens, embedding_dim, V_heads[h], embedding_dim, HEAD_DIM);
    }
}

// Scaled Dot-Product Attention (with masking)
float** scaleDotProductAttention(float*** Q, float*** K, float*** V, float** mask_matrix, int num_heads, int num_tokens) {
    float*** scores = new float**[num_heads];
    for (int h = 0; h < num_heads; ++h) {
        scores[h] = scaledAttention(Q[h], num_tokens, HEAD_DIM, K[h], num_tokens, HEAD_DIM);
    }

    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j < num_tokens; ++j) {
                scores[h][i][j] += mask_matrix[i][j];
            }
        }
        softmax(scores[h], num_tokens, num_tokens);
    }

    float** multihead_output = allocateMatrix(num_tokens, HEAD_DIM * num_heads);

    for (int h = 0; h < num_heads; ++h) {
        float** attention_output = applyAttentionWeights(scores[h], num_tokens, num_tokens, V[h], num_tokens, HEAD_DIM);
        if (h == 0) {
            copyMatrix(attention_output, multihead_output, num_tokens, HEAD_DIM * num_heads);
        } else {
            float** temp_concat = concatenate_matrices(&multihead_output, 1, num_tokens, (h + 1) * HEAD_DIM);
            deallocateMatrix(multihead_output, num_tokens);
            multihead_output = temp_concat;
            for (int i = 0; i < num_tokens; i++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    multihead_output[i][h * HEAD_DIM + j] = attention_output[i][j];
                }
            }
            deallocateMatrix(attention_output, num_tokens);
        }
        
    }

    return multihead_output;
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
        Q_heads[h] = multiplyMatricesCUDA(input_embedding, input_rows, input_cols, W_Q_heads[h], input_cols, HEAD_DIM);
        K_heads[h] = multiplyMatricesCUDA(input_embedding, input_rows, input_cols, W_K_heads[h], input_cols, HEAD_DIM);
        V_heads[h] = multiplyMatricesCUDA(input_embedding, input_rows, input_cols, W_V_heads[h], input_cols, HEAD_DIM);

        // Compute attention scores: Q * K^T
        float** K_transpose = transposeMatrix(K_heads[h], input_rows, HEAD_DIM);
        attention_scores[h] = multiplyMatricesCUDA(Q_heads[h], input_rows, HEAD_DIM, K_transpose, HEAD_DIM, input_rows);
        deallocateMatrix(K_transpose, HEAD_DIM);
    }
}
void backward(float** input_embedding, int input_rows, int input_cols, 
              float*** Q_heads, float*** K_heads, float*** V_heads, 
              float*** W_Q_heads, float*** W_K_heads, float*** W_V_heads, 
              float*** attention_scores, float*** d_attention_scores, 
              float*** d_V) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        
        // Allocate matrices for d_Q and d_K
        float** d_Q = allocateMatrix(input_rows, HEAD_DIM);
        float** d_K = allocateMatrix(input_rows, HEAD_DIM);

        // Compute d_Q = d_attention_scores * K
        d_Q = multiplyMatricesCUDA(d_attention_scores[h], input_rows, input_rows, K_heads[h], input_rows, HEAD_DIM);

        // Compute d_K = d_attention_scores^T * Q
        float** d_attention_scores_transpose = transposeMatrix(d_attention_scores[h], input_rows, input_rows);
        d_K = multiplyMatricesCUDA(d_attention_scores_transpose, input_rows, input_rows, Q_heads[h], input_rows, HEAD_DIM);

        // Compute gradients for W_Q, W_K, and W_V using matrix multiplication
        float** d_W_Q = allocateMatrix(input_cols, HEAD_DIM);
        float** d_W_K = allocateMatrix(input_cols, HEAD_DIM);
        float** d_W_V = allocateMatrix(input_cols, HEAD_DIM);

        // dW_Q = input_embedding^T * d_Q
        float** input_embedding_transpose = transposeMatrix(input_embedding, input_rows, input_cols);
        d_W_Q = multiplyMatricesCUDA(input_embedding_transpose, input_cols, input_rows, d_Q, input_rows, HEAD_DIM);

        // dW_K = input_embedding^T * d_K
        d_W_K = multiplyMatricesCUDA(input_embedding_transpose, input_cols, input_rows, d_K, input_rows, HEAD_DIM);
        // dW_V = input_embedding^T * d_V[h]
        d_W_V = multiplyMatricesCUDA(input_embedding_transpose, input_cols, input_rows, d_V[h], input_rows, HEAD_DIM);

        // Update weights using the gradients
        for (int i = 0; i < input_cols; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j) {
                W_Q_heads[h][i][j] -= LEARNING_RATE * d_W_Q[i][j];
                W_K_heads[h][i][j] -= LEARNING_RATE * d_W_K[i][j];
                W_V_heads[h][i][j] -= LEARNING_RATE * d_W_V[i][j];
            }
        }

        // Clean up allocated matrices
        deallocateMatrix(d_Q, input_rows);
        deallocateMatrix(d_K, input_rows);
        deallocateMatrix(d_W_Q, input_cols);
        deallocateMatrix(d_W_K, input_cols);
        deallocateMatrix(d_W_V, input_cols);
        deallocateMatrix(input_embedding_transpose, input_cols);
        deallocateMatrix(d_attention_scores_transpose, input_rows);
    }
}


int main() {
    freopen("output.txt", "w", stdout);
    int num_tokens = 5; // Example token count
    int embedding_dim = 200; // Dimensionality of embeddings
    float** input_embedding = allocateMatrix(num_tokens, embedding_dim);
    
    // Randomize input embedding for testing
    randomizeMatrix(input_embedding, num_tokens, embedding_dim);

    // Declare matrices for Q, K, V heads, and weights for multi-head attention
    float*** Q_heads = new float**[NUM_HEADS];
    float*** K_heads = new float**[NUM_HEADS];
    float*** V_heads = new float**[NUM_HEADS];
    float*** W_Q_heads = new float**[NUM_HEADS];
    float*** W_K_heads = new float**[NUM_HEADS];
    float*** W_V_heads = new float**[NUM_HEADS];
    float*** attention_scores = new float**[NUM_HEADS];

    // Generate multihead attention (forward pass)
    forward(input_embedding, num_tokens, embedding_dim, Q_heads, K_heads, V_heads, W_Q_heads, W_K_heads, W_V_heads, attention_scores);

    // Printing attention scores (optional)
    for (int h = 0; h < 1; ++h) {
        cout << "W_Q_heads " << h << "th\n";
        printMatrix(W_Q_heads[h], embedding_dim, HEAD_DIM);
    }

    // Simulating d_attention_scores for backpropagation (randomized for testing)
    float*** d_attention_scores = new float**[NUM_HEADS];
    for (int h = 0; h < NUM_HEADS; ++h) {
        d_attention_scores[h] = allocateMatrix(num_tokens, num_tokens);
        randomizeMatrix(d_attention_scores[h], num_tokens, num_tokens); // Randomizing for testing
    }

    // Simulating d_V for backpropagation (randomized for testing)
    float*** d_V = new float**[NUM_HEADS];
    for (int h = 0; h < NUM_HEADS; ++h) {
        d_V[h] = allocateMatrix(num_tokens, HEAD_DIM);  // Adjust the dimensions if necessary
        randomizeMatrix(d_V[h], num_tokens, HEAD_DIM); // Randomizing for testing
    }

    // Perform backward pass (gradient computation)
    backward(input_embedding, num_tokens, embedding_dim, 
             Q_heads, K_heads, V_heads, 
             W_Q_heads, W_K_heads, W_V_heads, 
             attention_scores, d_attention_scores, d_V);

    for (int i = 0; i < 1; i++)
    {
        cout << "W_Q_heads " << i << "th\n";
        printMatrix(W_Q_heads[i], EMBEDDING_DIM, HEAD_DIM);
    }
    // Clean up
    deallocateMatrix(input_embedding, num_tokens);
    for (int h = 0; h < NUM_HEADS; ++h) {
        deallocateMatrix(Q_heads[h], num_tokens);
        deallocateMatrix(K_heads[h], num_tokens);
        deallocateMatrix(V_heads[h], num_tokens);
        deallocateMatrix(W_Q_heads[h], embedding_dim);
        deallocateMatrix(W_K_heads[h], embedding_dim);
        deallocateMatrix(W_V_heads[h], embedding_dim);
        deallocateMatrix(attention_scores[h], num_tokens);
        deallocateMatrix(d_attention_scores[h], num_tokens);
        deallocateMatrix(d_V[h], num_tokens);
    }
    delete[] Q_heads;
    delete[] K_heads;
    delete[] V_heads;
    delete[] W_Q_heads;
    delete[] W_K_heads;
    delete[] W_V_heads;
    delete[] attention_scores;
    delete[] d_attention_scores;
    delete[] d_V;
    fclose(stdout);

    return 0;
}
