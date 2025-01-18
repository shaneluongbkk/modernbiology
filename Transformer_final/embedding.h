#include "matrix.h"

// Positional encoding with pointer
float* load_vector_position(int position, int dimension) {
    float* encoding = new float[dimension];   // Dynamically allocate memory for the position vector
    for (int i = 0; i < dimension; i++) {
        if (i % 2 == 0)
            encoding[i] = sin(position / pow(10000, i / (float)dimension));
        else
            encoding[i] = cos(position / pow(10000, (i - 1) / (float)dimension));
    }
    return encoding;
}

// Read embedding file
void load_embeddings(const string &filename, unordered_map<string, float*> &embeddings) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }

    while (getline(file, line)) {
        istringstream iss(line);
        string word;
        iss >> word;
        float* vec = new float[EMBED_SIZE];  // Dynamically allocate memory for the embedding vector
        float value;
        int i = 0;
        while (iss >> value && i < EMBED_SIZE) {
            vec[i++] = value;
        }
        embeddings[word] = vec;
    }
    file.close();
}

// Get embedding for a word
float* get_embedding(const string &word, const unordered_map<string, float*> &embeddings) {
    auto it = embeddings.find(word);
    if (it != embeddings.end()) {
        return it->second;
    } else {
        cerr << "Word not found: " << word << endl;
        float* empty_vector = new float[EMBED_SIZE]();  // Return a zero vector if word not found
        return empty_vector;
    }
}

// Split sentence into words
char** split_sentence(const string &sentence, size_t &num_tokens) {
    istringstream iss(sentence);
    string word;
    num_tokens = 0;
    char** words = new char*[100];  // Preallocate a reasonable size (100 words max)

    while (iss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);  // Lowercase the word
        words[num_tokens] = new char[word.size() + 1];  // Allocate memory for each word
        strcpy(words[num_tokens], word.c_str());  // Copy word into allocated space
        num_tokens++;
    }
    return words;
}