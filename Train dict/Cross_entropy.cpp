#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>
#define EMBED_SIZE 300

float** generate_random_input(int num_tokens) {
    float** input = new float*[num_tokens];
    for (int i = 0; i < num_tokens; ++i) {
        input[i] = new float[EMBED_SIZE];
        for (int j = 0; j < EMBED_SIZE; ++j) {
            input[i][j] = static_cast<float>(rand()) / RAND_MAX;  // Random float between 0 and 1
        }
    }
    return input;
}
float cross_entropy_loss(float** input, float** target, int num_tokens, int embed_size) {
    float loss = 0.0;
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < embed_size; ++j) {
            if (input[i][j] > 0) {  
                loss -= target[i][j] * std::log(input[i][j]);
            }
        }
    }
    return loss / num_tokens; 
}

// Hàm giải phóng bộ nhớ
void free_matrix(float** matrix, int num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}