#include "Input_embedding_glove.cpp"
#include "multihead.cpp"
// #include "part2.cpp"
#include "FeedForward.cpp"
#include "AddnNorm.cpp"
#include <iostream>

// #define ENCODER
#define DECODER

using namespace std;

void Linear() {}

void EncoderLayer(const Matrix &input_embedding)
{
    int num_tokens = input_embedding.size();
    // Initialize vectors to store the resulting Q, K, V heads
    vector<HEAD> Q_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> K_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> V_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));

    // Generate Q, K, V heads for multi-head attention
    HEAD input_head = HEAD(input_embedding.size(), input_embedding[0].size());
    input_head.data = input_embedding;
    generate_multihead_qkv(input_head, Q_heads, K_heads, V_heads);

    // Calculating scaled attention for each head
    vector<Matrix> attention_scores(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        attention_scores[h] = scaledAttention(Q_heads[h].toMatrix(), K_heads[h].toMatrix());
        softmax(attention_scores[h]); // Apply softmax to attention scores
    }

    // apply attention weights to value heads
    vector<Matrix> outputs(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        outputs[h] = applyAttentionWeights(attention_scores[h], V_heads[h].toMatrix());
    }

    // Concatenate heads and project
    Matrix multihead_output = concatenate_matrices(outputs); // Concatenate outputs from all heads
    // cout << "Multi-head attention output with dimensions: " << multihead_output.size() << "x" << multihead_output[0].size() << endl;
    // printMatrix(multihead_output); // Assuming you have a printMatrix function to display the output

    // cout << "---------------------" << endl;
    Matrix orig_matrix = input_embedding;
    AddnNorm(multihead_output, orig_matrix);

    size_t d_model = 200;
    size_t d_ff = 200;
    size_t M = 1; // Batch size
    size_t N = 1; // Number of input tensors

    // Create random matrices and biases
    Matrix W1 = createRandomMatrix(d_model, d_ff);
    Matrix W2 = createRandomMatrix(d_ff, d_model);
    Array b1 = createRandomVector(N);
    Array b2 = createRandomVector(N);

    // Create random input tensor
    // std::vector<Matrix> input(N, createRandomMatrix(M, d_model));

    // Get FeedForward output
    auto feedforward_output = FeedForward(multihead_output, W1, b1, W2, b2, true);

    // Apply Add and Norm
    AddnNorm(feedforward_output, multihead_output);
    cout << "After processing Encoder Layer" << endl;
    printMatrix(feedforward_output);
    // return feedforward_output;
}

void DecoderLayer(const Matrix &input_embedding)
{
    int num_tokens = input_embedding.size();
    // Initialize vectors to store the resulting Q, K, V heads
    vector<HEAD> Q_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> K_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> V_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));

    // Generate Q, K, V heads for multi-head attention
    HEAD input_head = HEAD(input_embedding.size(), input_embedding[0].size());
    input_head.data = input_embedding;
    generate_multihead_qkv(input_head, Q_heads, K_heads, V_heads);

    HEAD mask = createCausalMask(num_tokens);

    // Concatenate heads and project
    Matrix multihead_output = scaleDotProductAttention(Q_heads, K_heads, V_heads, mask);
    // cout << "Multi-head attention output with dimensions: " << multihead_output.size() << "x" << multihead_output[0].size() << endl;
    // printMatrix(multihead_output); // Assuming you have a printMatrix function to display the output

    // cout << "---------------------" << endl;
    Matrix orig_matrix = input_embedding;
    AddnNorm(multihead_output, orig_matrix);

    size_t d_model = 200;
    size_t d_ff = 200;
    size_t M = 1; // Batch size
    size_t N = 1; // Number of input tensors

    // Create random matrices and biases
    Matrix W1 = createRandomMatrix(d_model, d_ff);
    Matrix W2 = createRandomMatrix(d_ff, d_model);
    Array b1 = createRandomVector(N);
    Array b2 = createRandomVector(N);

    // Create random input tensor

    auto feedforward_output = FeedForward(multihead_output, W1, b1, W2, b2, true);
    AddnNorm(feedforward_output, multihead_output);
    cout << "After processing Decoder Layer" << endl;
    printMatrix(feedforward_output);

    Matrix W = createRandomMatrix(d_model, d_model);
    Array b = createRandomVector(N);
    auto output = Linear(feedforward_output, W, b[0]);
    softmax(output);
}

// int main()
// {
//     // freopen("input.txt", "r", stdin);
//     // freopen("output.txt", "w", stdout);
//     unordered_map<string, vector<float>> embeddings;
//     load_embeddings("glove.100.200d.txt", embeddings); // Load file

//     string sentence;
//     cout << "Input a sentence: ";
//     getline(cin, sentence);

//     // Words to vec
//     vector<string> words = split_sentence(sentence);
//     size_t num_tokens = words.size();

//     float **wde = new float *[num_tokens];
//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         wde[i] = new float[EMBED_SIZE]; // Cấp memory cấp bộ nhớ
//     }

//     // Initialize input_embedding with the appropriate dimensions
//     HEAD input_embedding(num_tokens, EMBEDDING_DIM);

//     // Populate input_embedding with embeddings + positional encoding for each word
//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         // Get word embedding
//         vector<float> embedding = get_embedding(words[i], embeddings);
//         copy(embedding.begin(), embedding.end(), wde[i]);

//         // Get positional encoding for this position
//         vector<float> position_encoding = load_vector_position(i, EMBED_SIZE);

//         // Combine word embedding with positional encoding
//         if (embedding.size() == EMBEDDING_DIM)
//         {
//             for (size_t j = 0; j < EMBEDDING_DIM; ++j)
//             {
//                 // Add positional encoding to the word embedding
//                 input_embedding.data[i][j] = embedding[j] + position_encoding[j];
//             }
//         }
//     }

//     Matrix current_input = input_embedding.toMatrix();

//     // Stack 6 encoder layers
//     for (int i = 0; i < 6; ++i)
//     {
//         // printMatrix(current_input);
//         cout << "Processing Encoder Layer " << i + 1 << endl;
//         EncoderLayer(current_input);
//         // printMatrix(current_input);
//     }

//     // Output the final result after 6 encoder layers
//     cout << "Final encoder output (after 6 layers):" << endl;
//     printMatrix(current_input); // Assuming you have a function to print matrices

//     return 0;
// }

int main()
{
    unordered_map<string, vector<float>> embeddings;
    load_embeddings("glove.100.200d.txt", embeddings);

    string sentence;
    cout << "Input a sentence: ";
    getline(cin, sentence);

    vector<string> words = split_sentence(sentence);
    size_t num_tokens = words.size();

    if (num_tokens == 0)
    {
        cout << "Error: Empty input sentence" << endl;
        return 1;
    }

    // Initialize input embedding
    HEAD input_embedding(num_tokens, EMBEDDING_DIM);

    // Debug information
    // cout << "Processing " << num_tokens << " tokens" << endl;

    for (size_t i = 0; i < num_tokens; ++i)
    {
        // Get word embedding
        vector<float> embedding = get_embedding(words[i], embeddings);

        if (embedding.empty() || embedding.size() != EMBEDDING_DIM)
        {
            cout << "Warning: Missing or invalid embedding for word: " << words[i] << endl;
            // Use zero vector as fallback
            embedding = vector<float>(EMBEDDING_DIM, 0.0f);
        }

        // Normalize the word embedding to prevent extreme values
        float norm = 0.0f;
        for (float val : embedding)
        {
            norm += val * val;
        }
        norm = sqrt(norm);
        if (norm > 0)
        {
            for (float &val : embedding)
            {
                val /= norm;
            }
        }

        // Get positional encoding
        vector<float> position_encoding = load_vector_position(i, EMBEDDING_DIM);

        // Scale down positional encoding to prevent overflow
        float scale_factor = 0.1f; // Adjust this value if needed
        for (float &val : position_encoding)
        {
            val *= scale_factor;
        }

        // Combine embeddings and check for invalid values
        for (size_t j = 0; j < EMBEDDING_DIM; ++j)
        {
            float combined_value = embedding[j] + position_encoding[j];

            // Check for invalid values
            if (std::isnan(combined_value) || std::isinf(combined_value))
            {
                cout << "Warning: Invalid value detected at position " << i << ", dimension " << j << endl;
                combined_value = 0.0f;
            }

            input_embedding.data[i][j] = combined_value;
        }

        // Debug: Print first few values for this token
        // cout << "Token '" << words[i] << "' first 5 values: ";
        // for (size_t j = 0; j < std::min(size_t(5), static_cast<size_t>(EMBEDDING_DIM)); ++j)
        // {
        //     cout << input_embedding.data[i][j] << " ";
        // }
        // cout << endl;
    }

    Matrix current_input = input_embedding.toMatrix();

    // Process through encoder layers with value checking
    for (int i = 0; i < 6; ++i)
    {
#ifdef ENCODER
        cout << "Processing Encoder Layer " << i + 1 << endl;
#endif
#ifdef DECODER
        cout << "Processing Decoder Layer " << i + 1 << endl;
#endif

        // Check input matrix for invalid values
        for (size_t r = 0; r < current_input.size(); ++r)
        {
            for (size_t c = 0; c < current_input[r].size(); ++c)
            {
                if (std::isnan(current_input[r][c]) || std::isinf(current_input[r][c]))
                {
                    cout << "Warning: Invalid value detected in input matrix at (" << r << "," << c << ")" << endl;
                    current_input[r][c] = 0.0f;
                }
            }
        }

#ifdef ENCODER
        EncoderLayer(current_input);
#endif
#ifdef DECODER
        DecoderLayer(current_input);
#endif
        // Verify output after encoder layer
        // cout << "Layer " << i + 1 << " output first few values: ";
        // if (!current_input.empty() && !current_input[0].empty())
        // {
        //     for (size_t j = 0; j < min(size_t(5), current_input[0].size()); ++j)
        //     {
        //         cout << current_input[0][j] << " ";
        //     }
        // }
        // cout << endl;
    }

#ifdef ENCODER
    cout << "Final encoder output (after 6 layers):" << endl;
    printMatrix(current_input);
#endif
#ifdef DECODER
    cout << "Final decoder output (after 6 layers):" << endl;
    printMatrix(current_input);
#endif
    return 0;
}
