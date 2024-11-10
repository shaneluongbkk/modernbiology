#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

using Matrix = vector<vector<double>>;

void AddnNorm(Matrix &target, Matrix &orig, double gamma = 1, double beta = 0)
{
    // Residual neural network: He et al. (2016)
    // Layer normalization: Ba et al. (2016)
    // This function performs the LayerNorm(x + Sublayer(x)) operation directly on target
    // target and orig are assumed to be vectors of size N x 512

    int sz = target.size();

    for (int i = 0; i < sz; ++i)
    {
        double mu = 0, sigma = 0;

        // Calculate mean (mu) and variance (sigma^2)
        for (int j = 0; j < 200; ++j)
        {
            target[i][j] += orig[i][j];           // residual connections
            mu += target[i][j];                   // E[X]
            sigma += target[i][j] * target[i][j]; // E[X^2]
        }

        mu /= 512;
        sigma = sqrt(sigma / 200 - mu * mu + 1e-10); // Var(X) = E[X^2] - E^2[X]

        // Apply normalization to each element
        for (int j = 0; j < 200; ++j)
            target[i][j] = gamma / sigma * (target[i][j] - mu) + beta;

        // Optional optimized version
        /*
        mu /= 512;
        sigma = gamma / sqrt(sigma / 512 - mu * mu + 1e-10);
        mu = beta - sigma * mu;

        for (int j = 0; j < 512; ++j)
            target[i][j] = sigma * target[i][j] + mu;
        */
    }
}
