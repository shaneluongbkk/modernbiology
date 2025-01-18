#include "matrix.h"

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

// Simulate scaled dot-product attention with Q * K_T
float **scaledAttention(float **Q, int Q_rows, int Q_cols, float **K, int K_rows, int K_cols)
{
    // Q_rows x K_rows result matrix (since K_T has dimensions K_cols x K_rows)
    float **result = allocateMatrix(Q_rows, K_rows);
    for (int i = 0; i < Q_rows; ++i)
    {
        for (int j = 0; j < K_rows; ++j)
        { // Note: using K_rows here, not K_cols
            result[i][j] = 0.0;
            for (int k = 0; k < Q_cols; ++k)
            {                                      // Q_cols = K_cols
                result[i][j] += Q[i][k] * K[j][k]; // Access K as if transposed
            }
        }
    }
    return result;
}

// Softmax function for each row of the matrix
void softmax(float** mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += exp(mat[i][j]);
        }
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = exp(mat[i][j]) / sum;
        }
    }
}

// Apply attention weights to the value matrix
float** applyAttentionWeights(float** attention_weights, int AW_rows, int AW_cols, float** V, int V_rows, int V_cols) {
    float** result = allocateMatrix(AW_rows, V_cols);
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

// Create a causal mask matrix
float** createCausalMask(int num_tokens) {
    float** mask_matrix = allocateMatrix(num_tokens, num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_tokens; ++j) {
            if (j > i)
                mask_matrix[i][j] = -numeric_limits<float>::infinity();
        }
    }
    return mask_matrix;
}

// Generate multi-head Q, K, V matrices from input embedding
void generate_multihead_qkv(float** input_embedding, int num_tokens, int embedding_dim, float*** Q_heads, float*** K_heads, float*** V_heads) {
    for (int h = 0; h < NUM_HEADS; ++h) {
        Q_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, Q_heads[h], embedding_dim, HEAD_DIM);
        K_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, K_heads[h], embedding_dim, HEAD_DIM);
        V_heads[h] = multiplyMatrices(input_embedding, num_tokens, embedding_dim, V_heads[h], embedding_dim, HEAD_DIM);
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
        multihead_output = concatenate_matrices(&attention_output, 1, num_tokens, HEAD_DIM * num_heads);
    }

    return multihead_output;
}