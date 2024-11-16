#include <iostream>
#include <cstdlib>
#include <stdexcept>

using namespace std;

const int EMBEDDING_DIM = 8;
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;

class HEAD {
public:
    int rows, cols;
    float** data;

    // Constructor
    HEAD(int r, int c) : rows(r), cols(c) {
        data = new float*[rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new float[cols]();
        }
    }

    // Copy constructor
    HEAD(const HEAD& other) : rows(other.rows), cols(other.cols) {
        data = new float*[rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new float[cols];
            for (int j = 0; j < cols; ++j) {
                data[i][j] = other.data[i][j];
            }
        }
    }

    // Destructor
    ~HEAD() {
        for (int i = 0; i < rows; ++i) {
            delete[] data[i];
        }
        delete[] data;
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
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    void randomizeHEAD() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
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
};

// MultiHead Attention generation
void generate_multihead_qkv(const HEAD& input_embedding, HEAD* Q_heads[], HEAD* K_heads[], HEAD* V_heads[]) {
    HEAD* W_Q_heads[NUM_HEADS];
    HEAD* W_K_heads[NUM_HEADS];
    HEAD* W_V_heads[NUM_HEADS];

    for (int h = 0; h < NUM_HEADS; ++h) {
        W_Q_heads[h] = new HEAD(EMBEDDING_DIM, HEAD_DIM);
        W_K_heads[h] = new HEAD(EMBEDDING_DIM, HEAD_DIM);
        W_V_heads[h] = new HEAD(EMBEDDING_DIM, HEAD_DIM);

        W_Q_heads[h]->randomizeHEAD();
        W_K_heads[h]->randomizeHEAD();
        W_V_heads[h]->randomizeHEAD();

        Q_heads[h] = new HEAD(input_embedding * (*W_Q_heads[h]));
        K_heads[h] = new HEAD(input_embedding * (*W_K_heads[h]));
        V_heads[h] = new HEAD(input_embedding * (*W_V_heads[h]));

        delete W_Q_heads[h];
        delete W_K_heads[h];
        delete W_V_heads[h];
    }
}

// Clean up dynamically allocated memory
void cleanup(HEAD* heads[], int size) {
    for (int i = 0; i < size; ++i) {
        delete heads[i];
    }
}

// int main() {
//     srand(42);

//     // Tạo và khởi tạo ngẫu nhiên ma trận input_embedding
//     HEAD input_embedding(10, EMBEDDING_DIM);
//     input_embedding.randomizeHEAD();

//     // Tạo các mảng để lưu trữ các HEAD cho Q, K, V
//     HEAD* Q_heads[NUM_HEADS];
//     HEAD* K_heads[NUM_HEADS];
//     HEAD* V_heads[NUM_HEADS];

//     // Sinh các HEADs cho Q, K, V từ input_embedding
//     generate_multihead_qkv(input_embedding, Q_heads, K_heads, V_heads);

//     // In ra các HEADs Q, K, V
//     cout << "Q_heads:" << endl;
//     for (int h = 0; h < NUM_HEADS; ++h) {
//         cout << "Q_head " << h + 1 << ":" << endl;
//         Q_heads[h]->print();
//         cout << endl;
//     }

//     cout << "K_heads:" << endl;
//     for (int h = 0; h < NUM_HEADS; ++h) {
//         cout << "K_head " << h + 1 << ":" << endl;
//         K_heads[h]->print();
//         cout << endl;
//     }

//     cout << "V_heads:" << endl;
//     for (int h = 0; h < NUM_HEADS; ++h) {
//         cout << "V_head " << h + 1 << ":" << endl;
//         V_heads[h]->print();
//         cout << endl;
//     }

//     // Giải phóng bộ nhớ
//     cleanup(Q_heads, NUM_HEADS);
//     cleanup(K_heads, NUM_HEADS);
//     cleanup(V_heads, NUM_HEADS);

//     return 0;
// }

