#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

using namespace std;

#define EMBED_SIZE 200
#define D_MODEL 200
#define D_FF 800
#define EMBEDDING_DIM 8
#define MAX_POSITION 1000
#define NUM_HEADS 8
#define HEAD_DIM (EMBEDDING_DIM / NUM_HEADS)
#define LEARNING_RATE 0.1

ofstream fo("output.txt");