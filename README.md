# Deep Learning - Modern Biology Study Group - Group 3
This branch provides functions to calculate scaled attention and softmax, with backpropagation and masked technique.
# Scaled attention function
For Q, K collected from token input, we have attention = Q . K^T
To scale the attention for better distribution, we divide attention with the square root of the dimension of key vectors.
Therefore scaled attention = Q . K^T / sqrt(d_K)
# Softmax function:
The main purpose of applying softmax is to create a uniform distribution to guess the next letter.
Apply softmax on each token.
# Masked technique
To prevent attention and softmax from collecting information from future letters, a mask is created in order to prevent the latter letter from affecting the current letter's outcome.
The technique is simply covering letters that come after the main focus letter.

