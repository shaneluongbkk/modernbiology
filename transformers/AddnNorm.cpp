#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <time.h>

using namespace std;

using Matrix = std::vector<std::vector<double>>;
using Array = std::vector<double>;

class AddnNorm{
private:
    Matrix w;
    Matrix &prev, &orig, &orig_grad;
    Array gamma, beta, mu, sigma;
    int num_words;
    const int N = 512; // size

    void matmul(const Array& A, const Matrix& B, Array& C){
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i] += A[j]*B[i][j]; 
        return;
    }

public:
    AddnNorm(Matrix &prev, Matrix &orig, Matrix &orig_grad) : prev(prev), orig(orig), orig_grad(orig_grad) {
        num_words = prev.size();
        gamma.resize(num_words, 1);
        beta.resize(num_words, 0);
        mu.reserve(num_words);
        sigma.reserve(num_words);
        w.resize(num_words, Array(N, 0));
    }

    Matrix forward(){
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

            for (int j = 0; j < N; ++j)
                w[i][j] = a*w[i][j] + b;
                // w[i][j] = gamma[i]/sigma[i]*(w[i][j] - mu[i]) + beta[i];
        }

        return w;
    }

    Matrix backward(Matrix& dy, const double lrate = 0.01){
        Matrix dz;
        dz.resize(num_words, Array(N, 0));
        vector<vector<double>> Jacobi(N, vector<double>(N, 0));

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
            for (int j = 0; j < N; ++j) dz[i][j] *= a;

            // update parameters
            gamma[i] -= lrate*dgamma;
            beta[i] -= lrate*dbeta;

            // update orig_grad
            for (int j = 0; j < N; ++j) orig_grad[i][j] += dz[i][j];
        }

        // this gradient is added to both the previous layer and the original
        return dz;
    }
};

// Matrix randomizeMatrix(int rows,int cols){
//     Matrix random(rows, vector<double>(cols));
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++) random[i][j]=static_cast<double>(rand())/RAND_MAX;
//     }
//     return random;
// }

// int main(){
//     srand(time(NULL));

//     vector<vector<double>> A = {{1, 2, 3}, {3, 4, 5}};
//     vector<vector<double>> orig = {{1, -1, 0.2}, {0.4, 2.3, 2.1}};
//     vector<vector<double>> orig_grad = {{0, 0, 0}, {0, 0, 0}};
//     vector<double> gamma = {2.2, 1}, beta = {1.1, 0.6};

//     AddnNorm addn(A, orig, orig_grad);
//     auto X = addn.forward();
//     cout << "Forward pass: \n";
//     for (auto& x : X){
//         for (auto& y : x) cout << y << " ";
//         cout << '\n';
//     }

//     Matrix dY = randomizeMatrix(2, 3);

//     cout << "dY: \n";
//     for (auto& x : dY){
//         for (auto& y : x) cout << y << " ";
//         cout << '\n';
//     }

//     auto Z = addn.backward(dY, 0.1);
//     cout << "Backward pass: \n";
//     for (auto& x : Z){
//         for (auto& y : x) cout << y << " ";
//         cout << '\n';
//     }

//     return 0;
// }
