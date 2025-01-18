#include "matrix.cpp"

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
        // dQ = d_attention_scores * K^T
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

