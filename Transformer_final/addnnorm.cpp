#include "main.h"

using Matrix = double**;
using Array = double*;

class AddnNorm{
private:
    Matrix w;
    Matrix &prev, &orig, &orig_grad;
    Array gamma, beta, mu, sigma;
    int num_words;
    int N; // size

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
        for(int i=0;i<num_words;i++){
            w[i]=new double[N];
            fill(w[i],w[i]+N,0.0);
        }
    }

    ~AddnNorm() {
        delete[] gamma, beta, mu, sigma;
        for(int i=0;i<num_words;i++) delete[] w[i];
        delete[] w;
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
            for (int j = 0; j < N; ++j) dz[i][j] *= a;

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

// Deallocate a matrix
void freeMatrix(Matrix mat, int rows){
    for(int i=0;i<rows;i++) delete[] mat[i];
    delete[] mat;
}

// int main(){
//     srand(static_cast<unsigned>(time(NULL)));
//     int rows=2,cols=3;

//     Matrix A = new double*[rows];
//     A[0] = new double[cols]{1, 2, 3};
//     A[1] = new double[cols]{3, 4, 5};

//     Matrix orig = new double*[rows];
//     orig[0] = new double[cols]{1, -1, 0.2};
//     orig[1] = new double[cols]{0.4, 2.3, 2.1};

//     Matrix orig_grad = new double*[rows];
//     for (int i = 0; i < rows; ++i) {
//         orig_grad[i] = new double[cols]{0, 0, 0};
//     }

//     double gamma[] = {2.2, 1};
//     double beta[] = {1.1, 0.6};

//     AddnNorm addn(A, orig, orig_grad, rows, cols);
//     auto X = addn.forward();
//     cout << "Forward pass: \n";
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++) cout<<X[i][j]<<" ";
//         cout<<'\n';
//     }

//     Matrix dY = randomizeMatrix(2, 3);
//     cout << "dY: \n";
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++) cout<<dY[i][j]<<" ";
//         cout<<'\n';
//     }

//     auto Z = addn.backward(dY, 0.1);
//     cout << "Backward pass: \n";
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++) cout<<Z[i][j]<<" ";
//         cout<<'\n';
//     }

//     freeMatrix(A,rows);
//     freeMatrix(orig,rows);
//     freeMatrix(orig_grad,rows);
//     freeMatrix(dY,rows);
//     freeMatrix(Z,rows);
// }