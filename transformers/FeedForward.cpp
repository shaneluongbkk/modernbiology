#include <cmath>    // log
using namespace std;

const int d_model = 200, d_ff = 800, num_tokens = 8;
const float learning_rate = 0.1;

struct FeedForward {
private:
    float hidden[num_tokens][d_ff];
    float d_output[num_tokens][d_model];
    float d_hidden[num_tokens][d_ff];
public:
    float **input;  // (num_tokens, d_model)
    float **output; // (num_tokens, d_model)
    float **W_in;   // (d_model, d_ff)
    float **W_out;  // (d_ff, d_model)
    float *b1;      // (d_ff)
    float *b2;      // (d_model)

    FeedForward(float **input, float **output, float **W_in, float **W_out, float *b1, float *b2)
    : input(input), output(output), W_in(W_in), W_out(W_out), b1(b1), b2(b2) {}
    void Forward();
    void Backward(float **target, float **d_input);   // target = (num_tokens, d_model), d_input = (num_tokens, d_model)
};

void FeedForward::Forward() {
    // Prep for second linear
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < d_model; ++j) output[i][j] = b2[j];
    }

    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            // first linear
            hidden[i][j] = b1[j];
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
    // b2[j] -= d_output[*][j].sum() * learning_rate
    int total = num_tokens * d_model;
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < d_model; ++j) {
            d_output[i][j] = target[i][j] / (output[i][j] * total);
            b2[j] -= d_output[i][j] * learning_rate;
        }
    }

    // d_W_out = Transpose(hidden) * d_output
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float d_W_out = 0;
            for (int k = 0; k < num_tokens; ++k) d_W_out += hidden[k][i] * d_output[k][j];
            W_out[i][j] -= learning_rate * d_W_out;
        }
    }

    // d_hidden = (hidden > 0) x (d_output * Transpose(W_out))
    // b1[j] -= d_hidden[*][i].sum() * learning_rate
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            d_hidden[i][j] = 0;
            if (hidden[i][j] <= 0) continue;
            for (int k = 0; k < d_model; ++k) d_hidden[i][j] += d_output[i][k] * W_out[j][k];
            b1[j] -= d_hidden[i][j] * learning_rate;
        }
    }

    // d_W_in = Transpose(input) * d_hidden
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            float d_W_in = 0;
            for (int k = 0; k < num_tokens; ++k) d_W_in += d_hidden[k][j] * input[k][i];
            W_in[i][j] -= learning_rate * d_W_in;
        }
    }

    // d_input = d_hidden * Transpose(W_in)
    for (int i = 0; i < num_tokens; ++i) {
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
//     makeMatrix(input, num_tokens, d_model, dis, gen);
//     makeMatrix(output, num_tokens, d_model);
//     makeMatrix(W_in, d_model, d_ff, dis, gen);
//     makeMatrix(W_out, d_ff, d_model, dis, gen);
//     makeArray(b1, d_ff, dis, gen);
//     makeArray(b2, d_model, dis, gen);
//     makeMatrix(target, num_tokens, d_model, dis, gen);
//     makeMatrix(d_input, num_tokens, d_model);

//     cout << "Input:\n";
//     printMatrix(input, num_tokens, d_model);
//     cout << "Target:\n";
//     printMatrix(target, num_tokens, d_model);
//     cout << "W_in:\n";
//     printMatrix(W_in, d_model, d_ff);
//     cout << "W_out:\n";
//     printMatrix(W_out, d_ff, d_model);
//     cout << "Biases:\n";
//     printArray(b1, num_tokens);
//     printArray(b2, num_tokens);

//     FeedForward ff(input, output, W_in, W_out, b1, b2);
//     ff.Forward();
//     cout << "\n##############################################\n"
//          << "                 After forward:\n"
//          << "##############################################\n\n"
//          << "Output:\n";
//     printMatrix(output, num_tokens, d_model);

//     ff.Backward(target, d_input);
//     cout << "##############################################\n"
//          << "                After backward:\n"
//          << "##############################################\n\n"
//          << "W_in:\n";
//     printMatrix(W_in, d_model, d_ff);
//     cout << "W_out:\n";
//     printMatrix(W_out, d_ff, d_model);
//     cout << "Biases:\n";
//     printArray(b1, num_tokens);
//     printArray(b2, num_tokens);
//     cout << "\nd_input:\n";
//     printMatrix(d_input, num_tokens, d_model);

//     deleteMatrix(input, num_tokens);
//     deleteMatrix(output, num_tokens);
//     deleteMatrix(W_in, d_model);
//     deleteMatrix(W_out, d_ff);
//     deleteArray(b1);
//     deleteArray(b2);
//     deleteMatrix(target, num_tokens);
//     deleteMatrix(d_input, num_tokens);
// }