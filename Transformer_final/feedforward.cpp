#include "main.h"

struct FeedForward
{
private:
    float hidden[NUM_HEADS][D_FF];
    float d_output[NUM_HEADS][D_MODEL];
    float d_hidden[NUM_HEADS][D_FF];

public:
    float **input;  // (NUM_HEADS, D_MODEL)
    float **output; // (NUM_HEADS, D_MODEL)
    float **W_in;   // (D_MODEL, D_FF)
    float **W_out;  // (D_FF, D_MODEL)
    float *b1, *b2; // (NUM_HEADS)

    FeedForward(float **input, float **output, float **W_in, float **W_out, float *b1, float *b2)
        : input(input), output(output), W_in(W_in), W_out(W_out), b1(b1), b2(b2) {}
    void Forward();
    void Backward(float **target, float **d_input); // target = (NUM_HEADS, D_MODEL), d_input = (NUM_HEADS, D_MODEL)
};

void FeedForward::Forward()
{
    // Prep for second linear
    for (int i = 0; i < NUM_HEADS; ++i)
    {
        for (int j = 0; j < D_MODEL; ++j)
            output[i][j] = b2[i];
    }

    for (int i = 0; i < NUM_HEADS; ++i)
    {
        for (int j = 0; j < D_FF; ++j)
        {
            // first linear
            hidden[i][j] = b1[i];
            for (int k = 0; k < D_MODEL; ++k)
                hidden[i][j] += input[i][k] * W_in[k][j];

            // ReLU
            if (hidden[i][j] < 0)
                hidden[i][j] = 0;

            // second linear
            for (int k = 0; k < D_MODEL; ++k)
                output[i][k] += hidden[i][j] * W_out[j][k];
        }
    }
}

void FeedForward::Backward(float **target, float **d_input)
{
    // d_output = target / (output * total)
    // d_b2[i] = d_output[i].sum()
    int total = NUM_HEADS * D_MODEL;
    for (int i = 0; i < NUM_HEADS; ++i)
    {
        float d_b2 = 0;
        for (int j = 0; j < D_MODEL; ++j)
        {
            d_output[i][j] = target[i][j] / (output[i][j] * total);
            d_b2 += d_output[i][j];
        }
        b2[i] -= LEARNING_RATE * d_b2;
    }

    // d_W_out = Transpose(hidden) * d_output
    for (int i = 0; i < D_FF; ++i)
    {
        for (int j = 0; j < D_MODEL; ++j)
        {
            float d_W_out = 0;
            for (int k = 0; k < NUM_HEADS; ++k)
                d_W_out += hidden[k][i] * d_output[k][j];
            W_out[i][j] -= LEARNING_RATE * d_W_out;
        }
    }

    // d_hidden = (hidden > 0) x (d_output * Transpose(W_out))
    // d_b1[i] = d_hidden[i].sum()
    for (int i = 0; i < NUM_HEADS; ++i)
    {
        float d_b1 = 0;
        for (int j = 0; j < D_FF; ++j)
        {
            d_hidden[i][j] = 0;
            if (hidden[i][j] <= 0)
                continue;
            for (int k = 0; k < D_MODEL; ++k)
                d_hidden[i][j] += d_output[i][k] * W_out[j][k];
            d_b1 += d_hidden[i][j];
        }
        b1[i] -= LEARNING_RATE * d_b1;
    }

    // d_W_in = Transpose(input) * d_hidden
    for (int i = 0; i < D_MODEL; ++i)
    {
        for (int j = 0; j < D_FF; ++j)
        {
            float d_W_in = 0;
            for (int k = 0; k < NUM_HEADS; ++k)
                d_W_in += d_hidden[k][j] * input[k][i];
            W_in[i][j] -= LEARNING_RATE * d_W_in;
        }
    }

    // d_input = d_hidden * Transpose(W_in)
    for (int i = 0; i < NUM_HEADS; ++i)
    {
        for (int j = 0; j < D_MODEL; ++j)
        {
            d_input[i][j] = 0;
            for (int k = 0; k < D_FF; ++k)
                d_input[i][j] += d_hidden[i][k] * W_in[j][k];
        }
    }
}