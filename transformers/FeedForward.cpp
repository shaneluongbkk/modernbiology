#include <cmath>    // log
using namespace std;

const int d_model = 200, d_ff = 800, nHeads = 8;
const float learning_rate = 0.1;

struct FeedForward {
private:
    float hidden[nHeads][d_ff];
    float d_output[nHeads][d_model];
    float d_hidden[nHeads][d_ff];
public:
    float **input;  // (nHeads, d_model)
    float **output; // (nHeads, d_model)
    float **W_in;   // (d_model, d_ff)
    float **W_out;  // (d_ff, d_model)
    float *b1, *b2; // (nHeads)

    FeedForward(float **input, float **output, float **W_in, float **W_out, float *b1, float *b2)
    : input(input), output(output), W_in(W_in), W_out(W_out), b1(b1), b2(b2) {}
    void Forward();
    void Backward(float **target, float **d_input);   // target = (nHeads, d_model), d_input = (nHeads, d_model)
};

void FeedForward::Forward() {
    // Prep for second linear
    for (int i = 0; i < nHeads; ++i) {
        for (int j = 0; j < d_model; ++j) output[i][j] = b2[i];
    }

    for (int i = 0; i < nHeads; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            // first linear
            hidden[i][j] = b1[i];
            for (int k = 0; k < d_model; ++k) hidden[i][j] += input[i][k] * W_in[k][j];

            // ReLU
            if (hidden[i][j] < 0) hidden[i][j] = 0;

            // second linear
            for (int k = 0; k < d_model; ++k) output[i][k] += hidden[i][j] * W_out[j][k];
        }
    }
}

void FeedForward::Backward(float **target, float **d_input) {
    // d_output = target / (output * total)
    // d_b2[i] = d_output[i].sum()
    int total = nHeads * d_model;
    for (int i = 0; i < nHeads; ++i) {
        float d_b2 = 0;
        for (int j = 0; j < d_model; ++j) {
            d_output[i][j] = target[i][j] / (output[i][j] * total);
            d_b2 += d_output[i][j];
        }
        b2[i] -= learning_rate * d_b2;
    }

    // d_W_out = Transpose(hidden) * d_output
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float d_W_out = 0;
            for (int k = 0; k < nHeads; ++k) d_W_out += hidden[k][i] * d_output[k][j];
            W_out[i][j] -= learning_rate * d_W_out;
        }
    }

    // d_hidden = (hidden > 0) x (d_output * Transpose(W_out))
    // d_b1[i] = d_hidden[i].sum()
    for (int i = 0; i < nHeads; ++i) {
        float d_b1 = 0;
        for (int j = 0; j < d_ff; ++j) {
            d_hidden[i][j] = 0;
            if (hidden[i][j] <= 0) continue;
            for (int k = 0; k < d_model; ++k) d_hidden[i][j] += d_output[i][k] * W_out[j][k];
            d_b1 += d_hidden[i][j];
        }
        b1[i] -= learning_rate * d_b1;
    }

    // d_W_in = Transpose(input) * d_hidden
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            float d_W_in = 0;
            for (int k = 0; k < nHeads; ++k) d_W_in += d_hidden[k][j] * input[k][i];
            W_in[i][j] -= learning_rate * d_W_in;
        }
    }

    // d_input = d_hidden * Transpose(W_in)
    for (int i = 0; i < nHeads; ++i) {
        for (int j = 0; j < d_model; ++j) {
            d_input[i][j] = 0;
            for (int k = 0; k < d_ff; ++k) d_input[i][j] += d_hidden[i][k] * W_in[j][k];
        }
    }
}


// #include <random>
// #include <iostream>

// void makeMatrix(float **&matrix, int row, int col) {
//     matrix = new float*[row];
//     for (int i = 0; i < row; ++i) matrix[i] = new float[col];
// }

// void makeMatrix(float **&matrix, int row, int col, uniform_real_distribution<float> &dis, mt19937 &gen) {
//     matrix = new float*[row];
//     for (int i = 0; i < row; ++i) {
//         matrix[i] = new float[col];
//         for (int j = 0; j < col; ++j) matrix[i][j] = dis(gen);
//     }
// }

// void deleteMatrix(float **&matrix, int row) {
//     for (int i = 0; i < row; ++i) delete[] matrix[i];
//     delete[] matrix;
// }

// void makeArray(float *&array, int size, uniform_real_distribution<float> &dis, mt19937 &gen) {
//     array = new float[size];
//     for (int i = 0; i < size; ++i) array[i] = dis(gen);
// }

// void deleteArray(float *&array) { delete[] array; }

// void printArray(float *array, int size) {
//     for (int i = 0; i < size; ++i) cout << array[i] << ' ';
//     cout << '\n';
// }

// void printMatrix(float **matrix, int row, int col) {
//     for (int i = 0; i < row; ++i) printArray(matrix[i], col);
//     cout << '\n';
// }

// int main() {
//     // freopen("Output.txt", "w", stdout);
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution<float> dis;
//     cout << fixed;
//     cout.precision(2);

//     float **input, **output, **W_in, **W_out, *b1, *b2, **target, **d_input;
//     makeMatrix(input, nHeads, d_model, dis, gen);
//     makeMatrix(output, nHeads, d_model);
//     makeMatrix(W_in, d_model, d_ff, dis, gen);
//     makeMatrix(W_out, d_ff, d_model, dis, gen);
//     makeArray(b1, nHeads, dis, gen);
//     makeArray(b2, nHeads, dis, gen);
//     makeMatrix(target, nHeads, d_model, dis, gen);
//     makeMatrix(d_input, nHeads, d_model);

//     cout << "Input:\n";
//     printMatrix(input, nHeads, d_model);
//     cout << "Target:\n";
//     printMatrix(target, nHeads, d_model);
//     cout << "W_in:\n";
//     printMatrix(W_in, d_model, d_ff);
//     cout << "W_out:\n";
//     printMatrix(W_out, d_ff, d_model);
//     cout << "Biases:\n";
//     printArray(b1, nHeads);
//     printArray(b2, nHeads);

//     FeedForward ff(input, output, W_in, W_out, b1, b2);
//     ff.Forward();
//     cout << "\n##############################################\n"
//          << "                 After forward:\n"
//          << "##############################################\n\n"
//          << "Output:\n";
//     printMatrix(output, nHeads, d_model);

//     ff.Backward(target, d_input);
//     cout << "##############################################\n"
//          << "                After backward:\n"
//          << "##############################################\n\n"
//          << "W_in:\n";
//     printMatrix(W_in, d_model, d_ff);
//     cout << "W_out:\n";
//     printMatrix(W_out, d_ff, d_model);
//     cout << "Biases:\n";
//     printArray(b1, nHeads);
//     printArray(b2, nHeads);
//     cout << "\nd_input:\n";
//     printMatrix(d_input, nHeads, d_model);

//     deleteMatrix(input, nHeads);
//     deleteMatrix(output, nHeads);
//     deleteMatrix(W_in, d_model);
//     deleteMatrix(W_out, d_ff);
//     deleteArray(b1);
//     deleteArray(b2);
//     deleteMatrix(target, nHeads);
//     deleteMatrix(d_input, nHeads);
// }