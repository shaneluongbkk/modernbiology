#include "FeedForward.h"
#include <cmath>    // log
#include <iostream> // cout
using namespace std;

FeedForward::FeedForward(double **input, double **output, double **W_in, double **W_out, double b1, double b2)
: input(input), output(output), W_in(W_in), W_out(W_out), b1(b1), b2(b2) {
    for (size_t i = 0; i < Nsamples; ++i) hidden[i] = new double[d_ff];
}

FeedForward::~FeedForward() { for (size_t i = 0; i < Nsamples; ++i) delete[] hidden[i]; }

void FeedForward::Forward() {
    // Prep for second linear
    for (size_t i = 0; i < Nsamples; ++i) {
        for (size_t j = 0; j < d_model; ++j) output[i][j] = b2;
    }

    for (size_t i = 0; i < Nsamples; ++i) {
        for (size_t j = 0; j < d_ff; ++j) {
            // first linear
            hidden[i][j] = b1;
            for (size_t k = 0; k < d_model; ++k) hidden[i][j] += input[i][k] * W_in[k][j];

            // ReLU
            if (hidden[i][j] < 0) hidden[i][j] = 0;

            // second linear
            for (size_t k = 0; k < d_model; ++k) output[i][k] += hidden[i][j] * W_out[j][k];
        }
    }
}

void FeedForward::Backward(const double **target) {
    // loss = (target * log(output)).sum()
    double loss = 0;
    for (size_t i = 0; i < Nsamples; ++i) {
        for (int j = 0; j < d_model; ++j) loss += target[i][j] * log(output[i][j]);
    }
    cout << "Loss: " << loss << '\n';

    // d_output = target / (output * total)
    // b2 -= learning_rate * d_b2.sum()
    double d_output[Nsamples][d_model];
    size_t total = Nsamples * d_model;
    double d_b2 = 0;
    for (size_t i = 0; i < Nsamples; ++i) {
        for (int j = 0; j < d_model; ++j) {
            d_output[i][j] = target[i][j] / (output[i][j] * total);
            d_b2 += d_output[i][j];
        }
    }
    b2 -= learning_rate * d_b2;

    // W_out -= learning_rate * Transpose(hidden) * d_output
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            double d_W_out = 0;
            for (int k = 0; k < Nsamples; ++k) {
                d_W_out += hidden[k][i] * d_output[k][j];
            }
            W_out[i][j] -= learning_rate * d_W_out;
        }
    }

    // d_hidden = (d_output * Transpose(W_out)) x (hidden > 0)
    // b1 -= learning_rate * d_hidden.sum()
    double d_hidden[Nsamples][d_ff];
    double d_b1 = 0;
    for (size_t i = 0; i < Nsamples; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            d_hidden[i][j] = 0;
            if (hidden[i][j] == 0) continue;
            for (int k = 0; k < d_model; ++k) d_hidden[i][j] += d_output[i][k] * W_out[j][k];
            d_b1 += d_hidden[i][j];
        }
    }
    b1 -= learning_rate * d_b1;

    // W_in -= learning_rate * Transpose(input) * d_hidden
    for (size_t i = 0; i < Nsamples; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            double d_W_in = 0;
            for (int k = 0; k < Nsamples; ++k) d_W_in += d_hidden[k][j] * input[k][i];
            W_in[i][j] -= learning_rate * d_W_in;
        }
    }
}