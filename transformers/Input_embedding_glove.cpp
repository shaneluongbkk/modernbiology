#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#define EMBED_SIZE 200
using namespace std;

vector<float> load_vector_position(int position, int dimension)
{ // khai bao
    // double encoding[dimension] = {0.0};  // create array for position and assign all equal 0;
    vector<float> encoding(dimension, 0.0);
    for (int i = 0; i < dimension; i++)
    { // check from 0 to number of dimention
        if (i % 2 == 0)
            encoding[i] = sin(position / pow(10000, i / dimension)); // funtion from paper, if the dimention is even we use sin;
        else
            encoding[i] = cos(position / pow(10000, (i - 1) / dimension)); // else if the dimention is odd we use cos;
    }
    return encoding; // resutl;
}

// Read embedding file
void load_embeddings(const string &filename, unordered_map<string, vector<float>> &embeddings)
{
    ifstream file(filename);
    string line;

    if (!file.is_open())
    {
        cerr << "Error opening file!" << endl;
        return;
    }

    while (getline(file, line))
    {
        istringstream iss(line);
        string word;
        vector<float> vec;

        iss >> word;
        float value;
        while (iss >> value)
        {
            vec.push_back(value);
        }

        embeddings[word] = vec;
    }
    file.close();
}

// Taking vector of word
vector<float> get_embedding(const string &word, const unordered_map<string, vector<float>> &embeddings)
{
    auto it = embeddings.find(word);
    if (it != embeddings.end())
    {
        return it->second;
    }
    else
    {
        cerr << "Word not found: " << word << endl;
        return vector<float>(EMBED_SIZE, 0.0f);
    }
}

// Split sentence into words
vector<string> split_sentence(const string &sentence)
{
    istringstream iss(sentence);
    vector<string> words;
    string word;

    while (iss >> word)
    {
        // Lowercase
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        words.push_back(word);
    }
    return words;
}

// int main()
// {
//     unordered_map<string, vector<float>> embeddings;
//     load_embeddings("glove.100.200d.txt", embeddings); // load file

//     // Input sentence
//     string sentence;
//     cout << "Nhập một câu: ";
//     getline(cin, sentence);

//     // Words2vec
//     vector<string> words = split_sentence(sentence);

//     size_t num_tokens = words.size();

//     float **wde = new float *[num_tokens];
//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         wde[i] = new float[EMBED_SIZE]; // Cấp memory
//     }

//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         vector<float> embedding = get_embedding(words[i], embeddings);
//         copy(embedding.begin(), embedding.end(), wde[i]); // Mảng 2 chiều chứa vec
//     }
//     // Test
//     cout << "Embedding matrix\n";
//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         for (size_t j = 0; j < EMBED_SIZE; ++j)
//         {
//             cout << wde[i][j] << " ";
//         }
//         cout << endl;
//     }

//     // free disk
//     for (size_t i = 0; i < num_tokens; ++i)
//     {
//         delete[] wde[i];
//     }
//     delete[] wde;
//     return 0;
// }
