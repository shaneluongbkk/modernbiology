#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include <stddef.h>

const int d_model = 3, d_ff = 4;
const size_t Nsamples = 3;
const double learning_rate = 0.05;

struct FeedForward {
private:
    double *hidden[Nsamples];   // (d_ff)
public:
    double **input;     // (Nsamples, d_model)
    double **output;    // (Nsamples, d_model)
    double **W_in;      // (d_model, d_ff)
    double **W_out;     // (d_ff, d_model)
    double b1, b2;

    FeedForward(double **input, double **output, double **W_in, double **W_out, double b1, double b2);
    ~FeedForward();
    void Forward();
    void Backward(const double **target);    // target = (Nsamples, d_model)
};

#endif