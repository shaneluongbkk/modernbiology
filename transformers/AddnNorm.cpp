#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

using Matrix = double**;
using Array = double*;

class AddnNorm{
private:
    Matrix w;
    Matrix &prev, &orig, &orig_grad;
    Matrix dropout_mask; // Mask for dropout
    Array gamma, beta, mu, sigma;
    int num_words;
    int N; // size
    const double dropout_rate=0.4;

    void matmul(const Array& A, const Matrix& B, Array& C){
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i] += A[j]*B[i][j]; 
        return;
    }

public:
    AddnNorm(Matrix &prev, Matrix &orig, Matrix &orig_grad, int num_words, int N) : prev(prev), orig(orig), orig_grad(orig_grad), num_words(num_words), N(N) {
        gamma=new double[num_words];
        beta=new double[num_words];
        mu=new double[num_words];
        sigma=new double[num_words];
        
        for(int i=0;i<num_words;i++) gamma[i]=1.0,beta[i]=0.0;

        w=new double*[num_words];
        dropout_mask=new double*[num_words];
        for(int i=0;i<num_words;i++){
            w[i]=new double[N], dropout_mask[i]=new double[N];
            fill(w[i],w[i]+N,0.0);
            fill(dropout_mask[i],dropout_mask[i]+N,1.0); // Default mask is all active
        }
    }

    ~AddnNorm() {
        delete[] gamma, beta, mu, sigma;
        for(int i=0;i<num_words;i++) delete[] w[i], delete[] dropout_mask[i];
        delete[] w, delete[] dropout_mask;
    }

    Matrix forward(bool is_training=true){
        // Residual neural network: He et al. (2016)
        // Layer normalization: Ba et al. (2016)
        // This function performs the LayerNorm(x + Sublayer(x)) operation directly on target
        // target and orig are assumed to be vectors of size N x 512

        for (int i = 0; i < num_words; ++i){
            mu[i] = 0; sigma[i] = 0;

            for (int j = 0; j < N; ++j){
                w[i][j] = prev[i][j] + orig[i][j]; // residual connections
                mu[i] += w[i][j]; // E[X]
                sigma[i] += w[i][j]*w[i][j]; // E[X^2]
            }

            mu[i] /= N;
            sigma[i] = sqrt(sigma[i]/N - mu[i]*mu[i] + 1e-8); 
            // sigma^2(X) = Var(X) = E[X^2] - E^2[X]
            // epsilon = 1e-8

            double a = gamma[i]/sigma[i];
            double b = beta[i] - mu[i]*a;

            for (int j = 0; j < N; ++j){
                w[i][j] = a*w[i][j] + b;
                // w[i][j] = gamma[i]/sigma[i]*(w[i][j] - mu[i]) + beta[i];

                // Apply dropout during training
                if(is_training){
                    dropout_mask[i][j]=static_cast<double>(rand())/RAND_MAX>dropout_rate? 1.0 : 0.0;
                    w[i][j]*=dropout_mask[i][j];
                }
                else {
                    dropout_mask[i][j]=1.0; // All neurons are active during testing
                }
                cout<<'\n';
            }

            // Scale active neurons during training
            if(is_training){
                for (int j = 0; j < N; ++j){
                    w[i][j]/=(1-dropout_rate);
                }
            }
        }

        return w;
    }

    Matrix backward(Matrix& dy, const double lrate = 0.1){
        // Maybe refer to this file for detailed explanations
        // https://drive.google.com/file/d/1ldvjiYQn7e9bTNZwEDMFm4yuKJMhah09/view?usp=sharing
        Matrix dz=new double*[num_words];
        for(int i=0;i<num_words;i++){
            dz[i]=new double[N];
            fill(dz[i],dz[i]+N,0.0);
        }
        
        Matrix Jacobi=new double*[N];
        for(int i=0;i<N;i++){
            Jacobi[i]=new double[N];
        }

        for (int i = 0; i < num_words; ++i){
            double dgamma = 0, dbeta = 0, dmu = 0, dsigma = 0;

            for (int j = 0; j < N; ++j){
                dgamma += dy[i][j]*w[i][j];
                dbeta += dy[i][j];

                for (int k = 0; k < N; ++k)
                    Jacobi[j][k] = -1 - w[i][j]*w[i][k];
                
                Jacobi[j][j] += N;
            }

            dgamma /= N; dbeta /= N;
            
            // mat_mul
            matmul(dy[i], Jacobi, dz[i]);

            double a = gamma[i]/(N*sigma[i]);
            for (int j = 0; j < N; ++j) {
                dz[i][j] *= a;
            }

            // update parameters
            gamma[i] -= lrate*dgamma;
            beta[i] -= lrate*dbeta;

            // update orig_grad
            for (int j = 0; j < N; ++j) orig_grad[i][j] += dz[i][j];
        }

        for(int i=0;i<num_words;i++) delete[] Jacobi[i];
        delete[] Jacobi;

        // this gradient is added to both the previous layer and the original
        return dz;
    }

    Array getGamma() {return gamma;}
    Array getBeta() {return beta;}
};

Matrix randomizeMatrix(int rows,int cols){
    Matrix random=new double*[rows];
    for(int i=0;i<rows;i++){
        random[i]=new double[cols];
        for(int j=0;j<cols;j++) {
            random[i][j]=static_cast<double>(rand())/RAND_MAX;
        }
    }
    return random;
}

// Function to allocate a matrix
Matrix allocateMatrix(int rows, int cols) {
    Matrix mat = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new double[cols];
    }
    return mat;
}

// Deallocate a matrix
void freeMatrix(Matrix mat, int rows){
    for(int i=0;i<rows;i++) delete[] mat[i];
    delete[] mat;
}

// Function to read a CSV file into a Matrix
Matrix readCSV(const string &filename, int &rows, int &cols) {
    ifstream file(filename);
    string line;
    rows = 0;
    cols = 0;

    // First pass to determine the number of rows and columns
    while (getline(file, line)) {
        if (rows == 0) {
            stringstream ss(line);
            string value;
            while (getline(ss, value, ',')) ++cols;
        }
        ++rows;
    }
    file.clear();
    file.seekg(0, ios::beg);

    // Allocate memory for the matrix
    Matrix data = allocateMatrix(rows, cols);

    // Read data into the matrix
    int i = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        int j = 0;
        while (getline(ss, value, ',')) {
            data[i][j++] = stod(value);
        }
        ++i;
    }
    file.close();
    return data;
}

// Function to write a Matrix to a CSV file
void writeCSV(const string &filename, Matrix data, int rows, int cols) {
    ofstream file(filename);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << fixed << setprecision(6) << data[i][j];
            if (j < cols - 1) file << ",";
        }
        file << '\n';
    }
    file.close();
}

// Function to read a CSV file into an Array
Array readArray(const string &filename, int &size) {
    ifstream file(filename);
    string line;
    getline(file, line);
    stringstream ss(line);
    string value;
    size = 0;

    // Count the number of values
    while (getline(ss, value, ',')) ++size;

    // Allocate memory for the array
    Array data = new double[size];

    // Fill the array
    ss.clear();
    ss.str(line);
    int i = 0;
    while (getline(ss, value, ',')) {
        data[i++] = stod(value);
    }
    file.close();
    return data;
}

// Function to write an Array to a CSV file
void writeArray(const string &filename, Array data, int size) {
    ofstream file(filename);
    for (int i = 0; i < size; ++i) {
        file << fixed << setprecision(6) << data[i];
        if (i < size - 1) file << '\n';
    }
    file.close();
}

int main(){
    srand(static_cast<unsigned>(time(NULL)));
    int rows=10,cols=512;

    Matrix prev = readCSV("prev.csv", rows, cols);
    Matrix orig = readCSV("orig.csv", rows, cols);

    Matrix orig_grad = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        orig_grad[i] = new double[cols]{0, 0, 0};
    }

    // Initialize AddnNorm instance
    AddnNorm addn(prev, orig, orig_grad, rows, cols);

    // Perform forward pass
    Matrix forward_result = addn.forward(true);

    // Generate random dY for backward pass
    Matrix dY = readCSV("dY.csv", rows, cols);

    // Perform backward pass
    Matrix backward_result = addn.backward(dY);

    // Write output to CSV files
    writeCSV("forward_output.csv", forward_result, rows, cols);
    writeCSV("backward_output.csv", backward_result, rows, cols);
    writeArray("gamma.csv", addn.getGamma(), rows);
    writeArray("beta.csv", addn.getBeta(), rows);

    // Free dynamically allocated memory
    freeMatrix(prev, rows);
    freeMatrix(orig, rows);
    freeMatrix(orig_grad, rows);
    freeMatrix(dY, rows);
    freeMatrix(forward_result, rows);
    freeMatrix(backward_result, rows);
}
