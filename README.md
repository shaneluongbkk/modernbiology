# Deep Learning - Modern Biology Study Group 2
**Multihead - Encoder**
1. Divide into heads: Split the input embedding into NUM_HEADS heads, each head has dimension: NUM_TOKENS x DIM/NUM_HEADS
2. Initialize Weight Matrices: Define and initialize three weight matrices: W_Q, W_K, W_V (DIM x DIM/NUM_HEADS)
3. Compute Q,K,V by multiplying the input embedding with the weight matrices
4. Compute Q*K_T

**Masked - Multihead - Decoder**
1. Utilizing the function in Encoder
2. Create a causal masked matrix (a quare matrix, NUM_TOKENS x NUM_TOKENS), ensure tokens just consider the preceding token by let -inf
3. Calculate the scaled-dot product, apply the causal masked matrix and calculate softmax

**Backward**
1. The derivatives of Q*K_T




   
