import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def merge_matrix(s, z):
    return np.vstack((s, z))

def random_matrix(rows, cols, mean=0.0, stddev=1.0):
    return np.random.normal(mean, stddev, (rows, cols))

def matmul(mat, vec):
    return np.dot(mat, vec)

class Decoder:
    def __init__(self, latent_size, output_size):
        self.W = random_matrix(latent_size, output_size)
        self.b = np.zeros(output_size)
        self.latent_size = latent_size
        self.output_size = output_size

    def forward(self, z, use_softmax=False):
        x_recon = matmul(self.W, z) + self.b
        if use_softmax:
            return softmax(x_recon)
        else:
            return sigmoid(x_recon)

def main():
    latent_size = 5
    output_size = 10

    decoder = Decoder(latent_size, output_size)

    z = np.random.normal(0.0, 1.0, latent_size)
    print(z)

    x_recon_sigmoid = decoder.forward(z, use_softmax=False)
    print(x_recon_sigmoid)

    x_recon_softmax = decoder.forward(z, use_softmax=True)
    print(x_recon_softmax)

if __name__ == "__main__":
    main()