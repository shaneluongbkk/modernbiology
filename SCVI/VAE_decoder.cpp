#include <iostream>
#include <cmath>
#include <random>


void sigmoid(double* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 1.0 / (1.0 + exp(-x[i]));
    }
}

void softmax(double* x, int size) {
    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += exp(x[i]);
    }
    for (int i = 0; i < size; ++i) {
        x[i] = exp(x[i]) / sum_exp;
    }
}

void merge_matrix(double** s, double** z, double** result, int rows_s, int rows_z, int cols) {
    for (int i = 0; i < rows_s; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = s[i][j];
        }
    }

    for (int i = 0; i < rows_z; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[rows_s + i][j] = z[i][j];
        }
    }
}

// Create random_mat with provided normal distribution
double** random_matrix(int rows, int cols, double mean = 0.0, double stddev = 1.0) {
    double** mat = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new double[cols];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = d(gen);
        }
    }
    return mat;
}

void matmul(double** mat, double* vec, double* result, int mat_rows, int mat_cols) {
    for (int i = 0; i < mat_rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < mat_cols; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}


class Decoder {
public:
// random W and b
    Decoder(int latent_size, int output_size) {
        W = random_matrix(latent_size, output_size);
        b = new double[output_size];
        for (int i = 0; i < output_size; ++i) {
            b[i] = 0.0;
        }
    }

    // free disk
    ~Decoder() {
        for (int i = 0; i < latent_size; ++i) {
            delete[] W[i];
        }
        delete[] W;
        delete[] b;
    }

    void forward(double* z, double* x_recon, bool use_softmax = false) {

        matmul(W, z, x_recon, latent_size, output_size);
        for (int i = 0; i < output_size; ++i) {
            x_recon[i] += b[i];
        }

        //activate function
        if (use_softmax) {
            softmax(x_recon, output_size);
        } else {
            for (int i = 0; i < output_size; ++i) {
                sigmoid(x_recon, output_size);
            }
        }
    }

private:
    double** W;
    double* b;
    int latent_size;
    int output_size;
};

int main() {
    int latent_size = 5;
    int output_size = 10;

    Decoder decoder(latent_size, output_size);

    double z[latent_size] = {0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);
    for (int i = 0; i < latent_size; ++i) {
        z[i] = d(gen);
        std::cout << z[i] << "/t";
    }

    double x_recon_sigmoid[output_size];
    double x_recon_softmax[output_size];

    decoder.forward(z, x_recon_sigmoid, false);

    for (int i = 0; i < output_size; ++i) {
        std::cout << x_recon_sigmoid[i] << " ";
    }

    decoder.forward(z, x_recon_softmax, true);

    for (int i = 0; i < output_size; ++i) {
        std::cout << x_recon_softmax[i] << " ";
    }

    return 0;
}
