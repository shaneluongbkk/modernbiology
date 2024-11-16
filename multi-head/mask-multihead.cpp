#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <limits>

using namespace std;

const int EMBEDDING_DIM = 8; // 512
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;

// allocate a matrix (2D array) dynamically
double** allocateMatrix(int rows, int cols) {
    double** mat = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new double[cols]();
    }
    return mat;
}

// deallocate matrix
void deallocateMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}

// Randomize a matrix with values between 0 and 1
void randomizeMatrix(double** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}

// Multiply two matrices (A * B)
double** multiplyMatrices(double** A, int A_rows, int A_cols, double** B, int B_rows, int B_cols) {
    if (A_cols != B_rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    double** result = allocateMatrix(A_rows, B_cols);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// print a matrix
void printMatrix(double** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

// Simulate scaled dot-product attention
double** scaledAttention(double** Q, int Q_rows, int Q_cols, double** K, int K_rows, int K_cols) {
    double** result = allocateMatrix(Q_rows, K_cols);
    for (int i = 0; i < Q_rows; ++i) {
        for (int j = 0; j < K_cols; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < K_rows; ++k) {
                result[i][j] += Q[i][k] * K[k][j];
            }
        }
    }
    return result;
}

// Softmax function for each row of the matrix
void softmax(double** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += exp(mat[i][j]);
        }
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = exp(mat[i][j]) / sum;
        }
    }
}

// Apply attention weights to the value matrix
double** applyAttentionWeights(double** attention_weights, int AW_rows, int AW_cols, double** V, int V_rows, int V_cols) {
    double** result = allocateMatrix(AW_rows, V_cols);
    for (int i = 0; i < AW_rows; ++i) {
        for (int j = 0; j < V_cols; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < AW_cols; ++k) {
                result[i][j] += attention_weights[i][k] * V[k][j];
            }
        }
    }
    return result;
}

// Concatenate matrices along the last axis
double** concatenate_matrices(double*** matrices, int num_matrices, int rows, int total_cols) {
    double** result = allocateMatrix(rows, total_cols);
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

// Create a causal mask matrix
double** createCausalMask(int num_tokens) {
    double** mask_matrix = allocateMatrix(num_tokens, num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_tokens; ++j) {
            if (j > i)
                mask_matrix[i][j] = -numeric_limits<double>::infinity();
        }
    }
    return mask_matrix;
}

// Generate multi-head Q, K, V matrices from input embedding
void generate_multihead_qkv(double** input_embedding, int num_tokens, int embedding_dim, double*** Q_heads, double*** K_heads, double*** V_heads) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        Q_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, Q_heads[h], embedding_dim, HEAD_DIM);
        K_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, K_heads[h], embedding_dim, HEAD_DIM);
        V_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, V_heads[h], embedding_dim, HEAD_DIM);
    }
}

// Scaled Dot-Product Attention (with masking)
double** scaleDotProductAttention(double*** Q, double*** K, double*** V, double** mask_matrix, int num_heads, int num_tokens) {
    double*** scores = new double**[num_heads];
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


    double** multihead_output = allocateMatrix(num_tokens, HEAD_DIM * num_heads);
    
    for (int h = 0; h < num_heads; ++h) {
        double** attention_output = applyAttentionWeights(scores[h], num_tokens, num_tokens, V[h], num_tokens, HEAD_DIM);
        multihead_output = concatenate_matrices(&attention_output, 1, num_tokens, HEAD_DIM * num_heads);
    }

    return multihead_output;
}
