#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

// Giải nén train-envi.tgz và tạo output.txt trước khi chạy code
double randomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int main() {
    // Khởi tạo seed cho hàm random -> để hàm random thực sự ngẫu nhiên (?! cái này chat gpt)
    srand(static_cast<unsigned>(time(0)));

    std::string outputFileName = "output.txt";

    std::ifstream inputFile("train.en");

    if (!inputFile) {
        std::cerr << "Không thể mở file input!" << std::endl;
        return 1;
    }

    std::unordered_set<std::string> uniqueWords;

    std::string word;
    while (inputFile >> word) {
        uniqueWords.insert(word);
    }
    inputFile.close();
    std::ifstream inputFile2("train.vi");
    while (inputFile2 >> word) {
        uniqueWords.insert(word);
    }
    inputFile2.close();

    std::ofstream outputFile(outputFileName);
    if (!outputFile) {
        std::cerr << "Không thể mở file output!" << std::endl;
        return 1;
    }


    for (const auto& uniqueWord : uniqueWords) {

        outputFile << uniqueWord;

        // Tạo vector 200 phần tử ngẫu nhiên và ghi vào file
        for (int i = 0; i < 200; ++i) {
            double randomValue = randomDouble();
            outputFile << " " << std::fixed << std::setprecision(6) << randomValue;
        }

        outputFile << "\n";
    }

    outputFile.close();

    std::cout << "Đã hoàn thành!" << std::endl;
    return 0;
}
