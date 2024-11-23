#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include<algorithm>
#include<cmath>

#define EMBED_SIZE 300                               
using namespace std;


// postional ebd
/*vector<float> load_vector_position(int position, int dimension){     // khai bao
       // double encoding[dimension] = {0.0};  // create array for position and assign all equal 0; 
       vector<float> encoding(dimension,0.0);
        for(int i =0; i<dimension;i++){    // check from 0 to number of dimention
            if(i % 2 ==0) encoding[i] = sin( position / pow(10000,i/dimension) ); // funtion from paper, if the dimention is even we use sin;
            else encoding[i] =  cos( position / pow(10000,(i-1)/dimension) );   // else if the dimention is odd we use cos;
        }
    return encoding; // resutl;
}
*/
/*vector<float> load_vector_position(int position, int dimension){   pointer lỏ
    vector<float> encoding(dimension,0.0);
    float* ptr = encoding.data();
    for(int i=0;i<encoding.size();i++){
         if(i % 2 ==0) ptr[i] = sin( position / pow(10000,i/dimension) ); // funtion from paper, if the dimention is even we use sin;
            else ptr[i] =  cos( position / pow(10000,(i-1)/dimension) );  
}
return encoding;
}
*/
// trò này hơi .... lấy pointer trỏ vào vector....

float* load_vector_position(int position, int dimension){  //pointer xịn
        float* encoding = new float[dimension];   // thay đổi vector thành pointer
        for(int i=0;i< dimension;i++){
            if(i%2==0) encoding[i] = sin( position / pow(10000,i/dimension) );
            else encoding[i] = cos( position / pow(10000,(i-1)/dimension) ); 
        }
    return encoding;
}

//Read embedding file
void load_embeddings(const string &filename, unordered_map<string, vector<float>> &embeddings) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }    // check file đầu vào, nếu file lỗi ==> lỗi mở file 

    while (getline(file, line)) {
        istringstream iss(line);
        string word;
        vector<float> vec;
        iss >> word;
        float value;
        while (iss >> value) { 
            vec.push_back(value);
        }

        embeddings[word] = vec;
    }
    file.close();
}

//Taking vector of word
vector<float> get_embedding(const string &word, const unordered_map<string, vector<float>> &embeddings) {
    auto it = embeddings.find(word);
    if (it != embeddings.end()) {
        return it->second;
    } else {
        cerr << "Word not found: " << word << endl;
        return vector<float>(EMBED_SIZE, 0.0f);
    }
}

// Split sentence into words
vector<string> split_sentence(const string &sentence) {
    istringstream iss(sentence);
    vector<string> words;
    string word;

    while (iss >> word) {
        // Lowercase
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        words.push_back(word);
    }
    return words;
}


int main() {
    unordered_map<string, vector<float>> embeddings;
   load_embeddings("mini_test.txt", embeddings); //load file
   //load_embeddings("mini_test.txt",embeddings); // load file minitest

    //Input sentence
    string sentence;
    cout << "Nhập một câu: ";
    getline(cin, sentence);

    //Words2vec
    vector<string> words = split_sentence(sentence); // convert the sentence to vector
    
    size_t num_tokens = words.size();

    float** wde = new float*[num_tokens];
    for (size_t i = 0; i < num_tokens; ++i) {
        wde[i] = new float[EMBED_SIZE];  // Cấp memory cấp bộ nhớ 
    }

    vector<vector<float>> combined_ebd;  // khai báo một vecto 2 chiều để lưu tổng vị trí và từ nhúng
    for (size_t i = 0; i < num_tokens; ++i) {
        vector<float> embedding = get_embedding(words[i], embeddings);

        copy(embedding.begin(), embedding.end(), wde[i]);  //Mảng 2 chiều chứa vec.

        float* vector_position = load_vector_position(i, EMBED_SIZE); // lấy vector vị trí của token trong câu. i là token EMBED_SIZE số chiều của một vec
        vector<float> combined(EMBED_SIZE,0.0f);  // dùng để lưu 

        for(size_t j =0 ;j<EMBED_SIZE;j++){
                 combined[j] = vector_position[j] + embedding[j];  // vector tổng của vector vị trí và embeding
    }
        combined_ebd.push_back(combined); // đẩy toàn bộ mọi tọa độ của vector tổng vào cmbed
    }
    
// Test
    cout << "Embedding matrix\n";
    for (size_t i = 0; i < num_tokens; ++i) {
        for (size_t j = 0; j < EMBED_SIZE; ++j) {
            cout << wde[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
// Test for sum vector
    cout <<"Final vector" <<  endl;
    for(size_t i = 0 ;i < num_tokens;i++){
        for(size_t j =0;j<EMBED_SIZE;j++){
            cout << combined_ebd[i][j];
        }
        cout << endl;
    }

    //free disk
    for (size_t i = 0; i < num_tokens; ++i) {
        delete[] wde[i]; 
    }
    delete[] wde;
    return 0;
}
