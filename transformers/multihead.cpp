#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <limits>

using namespace std;

const int EMBEDDING_DIM = 200; // 512
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
using Matrix = std::vector<std::vector<double>>;

class HEAD
{
public:
    int rows, cols;
    vector<vector<double>> data;

    HEAD(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0)) {}

    HEAD operator+(const HEAD &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw invalid_argument("HEAD dimensions do not match for addition.");
        }
        HEAD result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    HEAD operator*(const HEAD &other) const
    {
        if (cols != other.rows)
        {
            throw invalid_argument("HEAD dimensions do not match for multiplication.");
        }
        HEAD result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < other.cols; ++j)
            {
                for (int k = 0; k < cols; ++k)
                {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    void randomizeHEAD()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                data[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }

    Matrix toMatrix() const
    {
        Matrix mat(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat[i][j] = static_cast<double>(data[i][j]);
            }
        }
        return mat;
    }

    HEAD transpose() const
    {
        HEAD result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[j][i] = data[i][j];
        return result;
    }

    void print() const
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
};

// Hàm in ma trận
void printMatrix(const Matrix &mat)
{
    for (const auto &row : mat)
    {
        for (const auto &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Helper function to simulate scaled dot-product attention
Matrix scaledAttention(const Matrix &Q, const Matrix &K)
{
    Matrix result(Q.size(), vector<double>(K[0].size(), 0.0));
    for (int i = 0; i < Q.size(); ++i)
    {
        for (int j = 0; j < K[0].size(); ++j)
        {
            for (int k = 0; k < K.size(); ++k)
            {
                result[i][j] += Q[i][k] * K[k][j];
            }
        }
    }
    return result;
}

// Softmax implementation
void softmax(Matrix &mat)
{
    for (auto &row : mat)
    {
        double sum = 0.0;
        for (double val : row)
        {
            sum += exp(val);
        }
        for (double &val : row)
        {
            val = exp(val) / sum;
        }
    }
}

// Apply attention weights to value matrix
Matrix applyAttentionWeights(const Matrix &attention_weights, const Matrix &V)
{
    Matrix result(attention_weights.size(), vector<double>(V[0].size(), 0.0));
    for (int i = 0; i < attention_weights.size(); ++i)
    {
        for (int j = 0; j < V[0].size(); ++j)
        {
            for (int k = 0; k < V.size(); ++k)
            {
                result[i][j] += attention_weights[i][k] * V[k][j];
            }
        }
    }
    return result;
}

// Concatenate matrices along the last axis (this is a simplified version)
Matrix concatenate_matrices(const vector<Matrix> &matrices)
{
    int total_cols = 0;
    int rows = matrices[0].size();
    for (const auto &mat : matrices)
    {
        total_cols += mat[0].size();
    }
    Matrix result(rows, vector<double>(total_cols));
    int col_offset = 0;
    for (const auto &mat : matrices)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < mat[0].size(); ++j)
            {
                result[i][col_offset + j] = mat[i][j];
            }
        }
        col_offset += mat[0].size();
    }
    return result;
}

// Create causal mask
HEAD createCausalMask(int num_tokens)
{
    HEAD mask_matrix(num_tokens, num_tokens);
    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < num_tokens; j++)
        {
            if (j > i)
                mask_matrix.data[i][j] = numeric_limits<double>::infinity();
        }
    }
    return mask_matrix;
}

// MultiHead Attention generation
void generate_multihead_qkv(const HEAD &input_embedding, vector<HEAD> &Q_heads, vector<HEAD> &K_heads, vector<HEAD> &V_heads)
{
    vector<HEAD> W_Q_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));
    vector<HEAD> W_K_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));
    vector<HEAD> W_V_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));

    for (int h = 0; h < NUM_HEADS; ++h)
    {
        W_Q_heads[h].randomizeHEAD();
        W_K_heads[h].randomizeHEAD();
        W_V_heads[h].randomizeHEAD();

        Q_heads[h] = input_embedding * W_Q_heads[h];
        K_heads[h] = input_embedding * W_K_heads[h];
        V_heads[h] = input_embedding * W_V_heads[h];
    }
}

// Scaled Dot-Product Attention (Fixed to accept single HEAD for mask)
Matrix scaleDotProductAttention(const vector<HEAD> &Q, const vector<HEAD> &K, const vector<HEAD> &V, const HEAD &mask_matrix)
{
    vector<Matrix> scores(NUM_HEADS);

    for (int h = 0; h < NUM_HEADS; h++)
    {
        scores[h] = scaledAttention(Q[h].toMatrix(), K[h].toMatrix());
        printMatrix(scores[h]);
    }

    for (size_t k = 0; k < scores.size(); k++)
    {
        for (int i = 0; i < scores[k].size(); i++)
        {
            for (int j = 0; j < scores[k][i].size(); j++)
            {
                scores[k][i][j] += mask_matrix.data[i][j];
            }
            printMatrix(scores[k]);
        }
    }

    for (int h = 0; h < NUM_HEADS; h++)
    {
        softmax(scores[h]);
    }

    vector<Matrix> outputs(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        outputs[h] = applyAttentionWeights(scores[h], V[h].toMatrix());
    }

    Matrix multihead_output = concatenate_matrices(outputs);
    return multihead_output;
}

// int main()
// {
//     const int NUM_TOKENS = 10;

//     HEAD input_embedding(NUM_TOKENS, EMBEDDING_DIM);
//     input_embedding.randomizeHEAD();
//     input_embedding.print();

//     vector<HEAD> Q_heads(NUM_HEADS, HEAD(NUM_TOKENS, HEAD_DIM));
//     vector<HEAD> K_heads(NUM_HEADS, HEAD(NUM_TOKENS, HEAD_DIM));
//     vector<HEAD> V_heads(NUM_HEADS, HEAD(NUM_TOKENS, HEAD_DIM));

//     generate_multihead_qkv(input_embedding, Q_heads, K_heads, V_heads);

//     HEAD mask = createCausalMask(NUM_TOKENS);

//     Matrix attention_output = scaleDotProductAttention(Q_heads, K_heads, V_heads, mask);

//     cout << "Attention Output:" << endl;
//     printMatrix(attention_output);

//     return 0;
// }
