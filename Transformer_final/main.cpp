#include "embedding.h"
#include "multihead.h"

int main()
{
    if (!fo.is_open())
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    unordered_map<string, float *> embeddings;
    load_embeddings("glove.100.200d.txt", embeddings); // Load embeddings from file

    // Input sentence
    string sentence;
    cout << "Enter a sentence: ";
    getline(cin, sentence);

    // Words2vec
    size_t num_tokens;
    char **words = split_sentence(sentence, num_tokens); // Convert the sentence to an array of words

    float **wde = allocateMatrix(num_tokens, EMBED_SIZE);          // Allocate memory for word embeddings
    float **combined_ebd = allocateMatrix(num_tokens, EMBED_SIZE); // Allocate memory for combined embeddings

    for (size_t i = 0; i < num_tokens; ++i)
    {
        float *embedding = get_embedding(words[i], embeddings);
        copy(embedding, embedding + EMBED_SIZE, wde[i]); // Copy embedding into the 2D array

        float *vector_position = load_vector_position(i, EMBED_SIZE); // Get positional vector
        float *combined = combined_ebd[i];                            // Pointer to the combined embedding

        for (size_t j = 0; j < EMBED_SIZE; j++)
        {
            combined[j] = vector_position[j] + embedding[j]; // Combine position and word embedding
        }
        delete[] vector_position;
    }

    // Write embedding matrix to file
    fo << "Embedding layer\n";
    printMatrix(wde, num_tokens, EMBED_SIZE);

    // Write final combined vector to file (X)
    fo << "Final vector after add positional encoding\n";
    printMatrix(combined_ebd, num_tokens, EMBED_SIZE);

    float ***W_Q_heads = new float **[NUM_HEADS];
    float ***W_K_heads = new float **[NUM_HEADS];
    float ***W_V_heads = new float **[NUM_HEADS];

    float ***Q_heads = new float **[NUM_HEADS];
    float ***K_heads = new float **[NUM_HEADS];
    float ***V_heads = new float **[NUM_HEADS];

    float ***heads_output = new float **[NUM_HEADS];

    for (int i = 0; i < NUM_HEADS; ++i)
    {
        W_Q_heads[i] = allocateMatrix(EMBED_SIZE, HEAD_DIM);
        W_K_heads[i] = allocateMatrix(EMBED_SIZE, HEAD_DIM);
        W_V_heads[i] = allocateMatrix(EMBED_SIZE, HEAD_DIM);

        randomizeMatrix(W_Q_heads[i], EMBED_SIZE, HEAD_DIM);
        randomizeMatrix(W_K_heads[i], EMBED_SIZE, HEAD_DIM);
        randomizeMatrix(W_V_heads[i], EMBED_SIZE, HEAD_DIM);

        //  Q = X * W_Q; K = X * W_K; V = X * W_V
        Q_heads[i] = multiplyMatrices(combined_ebd, num_tokens, EMBED_SIZE, W_Q_heads[i], EMBED_SIZE, HEAD_DIM);
        K_heads[i] = multiplyMatrices(combined_ebd, num_tokens, EMBED_SIZE, W_K_heads[i], EMBED_SIZE, HEAD_DIM);
        V_heads[i] = multiplyMatrices(combined_ebd, num_tokens, EMBED_SIZE, W_V_heads[i], EMBED_SIZE, HEAD_DIM);

        // Q * K^T
        float **attention_score = scaledAttention(Q_heads[i], num_tokens, HEAD_DIM, K_heads[i], num_tokens, HEAD_DIM);
        // Q * K^T / sqrt(d_k)
        divideMatrixByScalar(attention_score, num_tokens, num_tokens, sqrt(HEAD_DIM));
        // A = softmax(Q * K^T / sqrt(d_k))
        softmax(attention_score, num_tokens, num_tokens);
        printMatrix(attention_score, num_tokens, num_tokens);
        // A * V
        heads_output[i] = applyAttentionWeights(attention_score, num_tokens, num_tokens, V_heads[i], num_tokens, HEAD_DIM);
        // printMatrix(heads_output[i], num_tokens, HEAD_DIM);
        deallocateMatrix(attention_score, num_tokens);
    }
    float **multihead_output = concatenate_matrices(heads_output, NUM_HEADS, num_tokens, HEAD_DIM * NUM_HEADS);

    // Write multi-head output to file
    fo << "Multi-head layer output\n";
    printMatrix(multihead_output, num_tokens, HEAD_DIM * NUM_HEADS);

    // Close the file stream
    fo.close();

    // Free allocated memory
    for (int i = 0; i < NUM_HEADS; ++i) {
        deallocateMatrix(W_Q_heads[i], EMBED_SIZE);
        deallocateMatrix(W_K_heads[i], EMBED_SIZE);
        deallocateMatrix(W_V_heads[i], EMBED_SIZE);

        deallocateMatrix(Q_heads[i], num_tokens);
        deallocateMatrix(K_heads[i], num_tokens);
        deallocateMatrix(V_heads[i], num_tokens);

        deallocateMatrix(heads_output[i], num_tokens);
    }
    delete[] W_Q_heads;
    delete[] W_K_heads;
    delete[] W_V_heads;
    delete[] Q_heads;
    delete[] K_heads;
    delete[] V_heads;
    delete[] heads_output;

    deallocateMatrix(wde, num_tokens);
    deallocateMatrix(combined_ebd, num_tokens);
    deallocateMatrix(multihead_output, num_tokens);


    // Free words array
    for (size_t i = 0; i < num_tokens; ++i)
    {
        delete[] words[i];
    }
    delete[] words;

    // Free embeddings
    for (auto &entry : embeddings)
    {
        delete[] entry.second; // Free each embedding vector
    }

    return 0;
}
