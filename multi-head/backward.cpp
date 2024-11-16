#ifndef MULTIHEAD
#define MULTIHEAD
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <ctime>

using namespace std;

const int EMBEDDING_DIM = 200;
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;

class HEAD {
public:
    int rows, cols;
    double** data;

    HEAD(int r = 0, int c = 0) : rows(r), cols(c) {
        allocateMemory();
        initializeToZero();
    }

    HEAD(const HEAD& other) : rows(other.rows), cols(other.cols) {
        allocateMemory();
        copyData(other.data);
    }

    ~HEAD() { freeMemory(); }

    HEAD& operator=(const HEAD& other) {
        if (this != &other) {
            freeMemory();
            rows = other.rows;
            cols = other.cols;
            allocateMemory();
            copyData(other.data);
        }
        return *this;
    }

    HEAD operator+(const HEAD& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw invalid_argument("HEAD dimensions do not match for addition.");
        }
        HEAD result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    HEAD operator*(const HEAD& other) const {
        if (cols != other.rows) {
            throw invalid_argument("HEAD dimensions do not match for multiplication.");
        }
        HEAD result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                for (int k = 0; k < cols; ++k)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }

    HEAD transpose() const {
        HEAD result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[j][i] = data[i][j];
        return result;
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }

private:
    void allocateMemory() {
        data = new double* [rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new double[cols];
        }
    }

    void freeMemory() {
        if (data) {
            for (int i = 0; i < rows; ++i) {
                delete[] data[i];
            }
            delete[] data;
            data = nullptr;
        }
    }

    void initializeToZero() {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = 0.0;
    }

    void copyData(double** src) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = src[i][j];
    }
};

void randomizeHEAD(HEAD& matrix) {
    for (int i = 0; i < matrix.rows; i++)
        for (int j = 0; j < matrix.cols; j++)
            matrix.data[i][j] = static_cast<double>(rand()) / RAND_MAX;
}

class MultiHeadAttention {
public:
    HEAD* Q_heads;
    HEAD* K_heads;
    HEAD* V_heads;
    HEAD* W_Q_heads;
    HEAD* W_K_heads;
    HEAD* W_V_heads;

    MultiHeadAttention() {
        Q_heads = new HEAD[NUM_HEADS];
        K_heads = new HEAD[NUM_HEADS];
        V_heads = new HEAD[NUM_HEADS];
        W_Q_heads = new HEAD[NUM_HEADS];
        W_K_heads = new HEAD[NUM_HEADS];
        W_V_heads = new HEAD[NUM_HEADS];

        for (int h = 0; h < NUM_HEADS; ++h) {
            Q_heads[h] = HEAD(MAX_POSITION, HEAD_DIM);
            K_heads[h] = HEAD(MAX_POSITION, HEAD_DIM);
            V_heads[h] = HEAD(MAX_POSITION, HEAD_DIM);
            W_Q_heads[h] = HEAD(EMBEDDING_DIM, HEAD_DIM);
            W_K_heads[h] = HEAD(EMBEDDING_DIM, HEAD_DIM);
            W_V_heads[h] = HEAD(EMBEDDING_DIM, HEAD_DIM);
        }
    }

    ~MultiHeadAttention() {
        delete[] Q_heads;
        delete[] K_heads;
        delete[] V_heads;
        delete[] W_Q_heads;
        delete[] W_K_heads;
        delete[] W_V_heads;
    }

    void forward(const HEAD& input_embedding) {
        generate_multihead_qkv(input_embedding);
    }

    void backward(HEAD* delta_attention_score, HEAD* delta_V_heads, const HEAD& input_embedding) {
        HEAD* delta_Q_heads = new HEAD[NUM_HEADS];
        HEAD* delta_K_heads = new HEAD[NUM_HEADS];

        //derivative
        for (int h = 0; h < NUM_HEADS; ++h) {
            delta_Q_heads[h] = delta_attention_score[h] * K_heads[h];
            delta_K_heads[h] = delta_attention_score[h].transpose() * Q_heads[h];
        }

        for (int h = 0; h < NUM_HEADS; ++h) {
            W_Q_heads[h] = input_embedding.transpose() * delta_Q_heads[h];
            W_K_heads[h] = input_embedding.transpose() * delta_K_heads[h];
            W_V_heads[h] = input_embedding.transpose() * delta_V_heads[h];
        }

        delete[] delta_Q_heads;
        delete[] delta_K_heads;
    }

private:
    void generate_multihead_qkv(const HEAD& input_embedding) {
        for (int h = 0; h < NUM_HEADS; ++h) {
            randomizeHEAD(W_Q_heads[h]);
            randomizeHEAD(W_K_heads[h]);
            randomizeHEAD(W_V_heads[h]);

            Q_heads[h] = input_embedding * W_Q_heads[h];
            K_heads[h] = input_embedding * W_K_heads[h];
            V_heads[h] = input_embedding * W_V_heads[h];
        }
    }
};

#endif
