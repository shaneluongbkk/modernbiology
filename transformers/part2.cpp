#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
using namespace std;

using Matrix = std::vector<std::vector<double>>;

void printMatrix(const Matrix &mat)
{
    for (const auto &row : mat)
    {
        for (double val : row)
        {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << std::endl;
    }
}

Matrix createMatrix(int rows, int cols, double defaultValue = 0.0)
{
    return Matrix(rows, std::vector<double>(cols, defaultValue));
}

/* Return the scaled attention for query and key matrix pair
 * @param query
 * @param key
 */
Matrix scaledAttention(const Matrix &query, const Matrix &key)
{
    // Query and Key has same matrix size
    // Attention will be scored based on Q . K^T
    int samples = query.size(); // sample count
    int size = query[0].size(); // input size
    int dim = key.size();       // key dimension for scaling

    Matrix result = createMatrix(samples, samples);
    double scale = 1.0 / std::sqrt(dim);

    for (int i = 0; i < samples; ++i)
    {
        // Calculating scores for each query
        for (int j = 0; j < samples; ++j)
        {
            for (int k = 0; k < size; ++k)
                result[i][j] += query[i][k] * key[j][k];
            result[i][j] *= scale;
        }
    }
    return result;
}

void softmax(Matrix &mat)
{
    for (auto &row : mat)
    {
        double sumExp = 0.0;
        for (double &val : row)
        {
            val = std::exp(val);
            sumExp += val;
        }
        for (double &val : row)
        {
            val /= sumExp;
        }
    }
}

Matrix applyAttentionWeights(const Matrix &attention, const Matrix &value)
{
    // For each input, we get one Weighted per Value => Find sum over all values of a single input
    // Combine all input weights into one matrix
    int samples = attention.size();
    int input_size = attention[0].size();
    int output_size = value[0].size(); // output size;

    Matrix output = createMatrix(samples, output_size);

    for (int i = 0; i < samples; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            for (int k = 0; k < input_size; k++)
            {
                output[i][j] += attention[i][k] * value[i][j];
            }
        }
    }
    return output;
}
Matrix concatenate_matrices(const vector<Matrix> &matrices)
{
    if (matrices.empty())
        return Matrix(); // Return empty matrix if no matrices are provided

    int total_cols = 0;
    int rows = matrices[0].size();

    // Calculate the total number of columns
    for (const auto &mat : matrices)
    {
        if (mat.size() != rows)
        {
            throw runtime_error("All matrices must have the same number of rows for concatenation.");
        }
        total_cols += mat[0].size(); // Assuming all matrices have the same number of columns
    }

    Matrix concatenated(rows, vector<double>(total_cols));

    int col_offset = 0;
    for (const auto &mat : matrices)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < mat[i].size(); ++j)
            {
                concatenated[i][col_offset + j] = mat[i][j];
            }
        }
        col_offset += mat[0].size();
    }
    return concatenated;
}

// Function to project the concatenated output (e.g., linear transformation)
Matrix project_output(const Matrix &concatenated, int output_dim)
{
    // Create a projection matrix of size (concatenated.cols x output_dim)
    Matrix projection = createMatrix(concatenated[0].size(), output_dim, 0.0);
    // Here you would fill the projection matrix with your weights
    // For simplicity, let's assume it's initialized randomly or with predefined values

    Matrix output = createMatrix(concatenated.size(), output_dim);
    for (size_t i = 0; i < concatenated.size(); ++i)
    {
        for (size_t j = 0; j < output_dim; ++j)
        {
            for (size_t k = 0; k < concatenated[i].size(); ++k)
            {
                output[i][j] += concatenated[i][k] * projection[k][j]; // Matrix multiplication
            }
        }
    }
    return output;
}

// int main() {
//     int dim = 4;

//     Matrix query = {
//         {1, 0, 1, 0},
//         {0, 1, 0, 1},
//         {1, 1, 1, 1},
//         {0, 0, 0, 1}
//     };

//     Matrix key = {
//         {1, 0, 1, 0},
//         {0, 1, 0, 1},
//         {1, 1, 1, 1},
//         {0, 0, 0, 1}
//     };

//     Matrix value = {
//         {1, 2, 3, 4},
//         {2, 3, 4, 5},
//         {3, 4, 5, 6},
//         {4, 5, 6, 7}
//     };

//     // Bước 1: Tính attention scores đã scale
//     Matrix attention = scaledAttention(query, key);
//     std::cout << "Scaled Attention Scores:\n";
//     printMatrix(attention);

//     // Bước 2: Áp dụng hàm Softmax
//     softmax(attention);
//     std::cout << "\nAttention Weights (sau Softmax):\n";
//     printMatrix(attention);

//     // Bước 3: Kết hợp trọng số chú ý với ma trận giá trị
//     Matrix output = applyAttentionWeights(attention, value);
//     std::cout << "\nOutput cuối cùng của Self-Attention:\n";
//     printMatrix(output);

//     return 0;
// }