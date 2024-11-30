#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>
// Define size of embedding vector and output file path
#define EMBED_SIZE 300
const std::string DICT_PATH = "output.txt";

// Count number of words in dict
long line_count_dict(){
    std::ifstream dict_file(DICT_PATH);
    if (!dict_file) {
        std::cerr << "Không thể mở file.\n";
        return 0;
    }

    int count = 0;
    std::string line;
    while (std::getline(dict_file, line)) {
        count++;
    }
return count;
}

// Creating one hot vector from a distributed one
float* one_hot_coding(float* input){
    float max = input[0];
    long index = 0;
    for (int i = 0; i < line_count_dict(); i++) {
        if (input[i] > max) {
            max = input[i];
            index = i;
        }
    }
    float* one_hot = new float[line_count_dict()];
    for (int i=0; i<line_count_dict(); i++){
        if (i!=index){ one_hot[i]=0;}
        else one_hot[i]=1;
    }
return one_hot;
}     
// Cross-entropy in which "input" is the output of the previous layer, 
// "one_hot" is the one hot coding vector from test file
float cross_entropy_loss(float* input, float* one_hot, int num_tokens, int embed_size) {
    float loss = 0.0;
    for (int i = 0; i < num_tokens; ++i) {
            if (input[i] > 0) {  
                loss -= one_hot[i] * std::log(input[i]);
            }
        }
    return loss / num_tokens; 
}

// Free disk
void free_matrix(float** matrix, int num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}