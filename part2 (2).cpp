#include <iostream>
#include <math.h>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double** matmul(double **x, double**y, int a, int b, int c, int samples) {
    double **res = createMatrix(samples, samples);
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = 0;
            for (int k = 0; k < b; ++k) {
                res[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return res;
}

double** createMatrix(int rows, int cols) {
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[cols]();
    }
    return matrix;
}

void deleteMatrix(double** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void softmax(double** mat, int rows, int cols, double** mask = nullptr) {
    for (int i = 0; i < rows; ++i) {
        double sumExp = 0.0;
        for (int j = 0; j < cols; ++j) {
            if (mask != nullptr && mask[i][j] == 0) {
                mat[i][j] = -1e9;  
            }
            mat[i][j] = std::exp(mat[i][j]);
            sumExp += mat[i][j];
        }
        for (int j = 0; j < cols; ++j) {
            mat[i][j] /= sumExp;
        }
    }
}

void backprop_softmax(double **dL_dS, double **S, int samples, int rows, int cols,) {
    double** jacobian = createMatrix(samples, samples);
    for (int i = 0; i < samples; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            if (i == j)
            {
                jacobian[i][j] = S[i][0] * (1 - S[i][0]);
            }
            else
            {
                jacobian[i][j] = -S[i][j] * S[j][0];
            }
        }
    }

    // Multiply Jacobian by the gradient of the loss with respect to softmax output
    return matmul(jacobian, dL_dS);
}


void matMul(double** A, double** B, double** result, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void applyMask(double** mat, double** mask, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (mask[i][j] == 0) {
                mat[i][j] = -1e9;  
            }
        }
    }
}

double** scaledAttention(double** query, double** key, int samples, int size, int dim, double** mask = nullptr) {
    double** result = createMatrix(samples, samples);
    double scale = 1.0 / std::sqrt(dim);
    
    matMul(query, key, result, samples, size, samples);
    
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < samples; ++j) {
            result[i][j] *= scale;
        }
    }

    if (mask != nullptr) {
        applyMask(result, mask, samples, samples);
    }

    return result;
}

void multiHeadAttention(double** Q, double** K, double** V, double** output, int samples, int size, int num_heads, double** mask = nullptr) {
    int head_size = size / num_heads;

    for (int h = 0; h < num_heads; ++h) {
        double** Q_head = new double*[samples];
        double** K_head = new double*[samples];
        double** V_head = new double*[samples];

        for (int i = 0; i < samples; ++i) {
            Q_head[i] = new double[head_size];
            K_head[i] = new double[head_size];
            V_head[i] = new double[head_size];
            for (int j = 0; j < head_size; ++j) {
                Q_head[i][j] = Q[i][h * head_size + j];
                K_head[i][j] = K[i][h * head_size + j];
                V_head[i][j] = V[i][h * head_size + j];
            }
        }

        double** attention_output = scaledAttention(Q_head, K_head, samples, head_size, head_size, mask);

        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < size; ++j) {
                output[i][j] += attention_output[i][j % head_size];
            }
        }

        for (int i = 0; i < samples; ++i) {
            delete[] Q_head[i];
            delete[] K_head[i];
            delete[] V_head[i];
            delete[] attention_output[i];
        }
        delete[] Q_head;
        delete[] K_head;
        delete[] V_head;
        delete[] attention_output;
    }
}

void backpropagationMultiHead(double** Q, double** K, double** V, double** grad_output, double** grad_Q, double** grad_K, double** grad_V, int samples, int size, int num_heads, double learning_rate, double** mask = nullptr) {
    int head_size = size / num_heads;

    for (int h = 0; h < num_heads; ++h) {
        double** Q_head = new double*[samples];
        double** K_head = new double*[samples];
        double** V_head = new double*[samples];
        double** grad_attention = new double*[samples];

        for (int i = 0; i < samples; ++i) {
            Q_head[i] = new double[head_size];
            K_head[i] = new double[head_size];
            V_head[i] = new double[head_size];
            grad_attention[i] = new double[samples];
            for (int j = 0; j < head_size; ++j) {
                Q_head[i][j] = Q[i][h * head_size + j];
                K_head[i][j] = K[i][h * head_size + j];
                V_head[i][j] = V[i][h * head_size + j];
            }
        }

        double** grad_output_head = new double*[samples];
        for (int i = 0; i < samples; ++i) {
            grad_output_head[i] = new double[head_size];
        }

        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                grad_attention[i][j] = grad_output[i][j] * sigmoid(grad_output[i][j]) * (1.0 - sigmoid(grad_output[i][j])); // Tính đạo hàm
            }
        }

        for (int i = 0; i < samples; ++i) {
            delete[] Q_head[i];
            delete[] K_head[i];
            delete[] V_head[i];
            delete[] grad_attention[i];
        }
        delete[] Q_head;
        delete[] K_head;
        delete[] V_head;
        delete[] grad_attention;
        delete[] grad_output_head;
    }
}