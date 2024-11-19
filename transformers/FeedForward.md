# Documentation for `Feed Forward.cpp`

### Overview
The `FeedForward.cpp` file implements a FeedForward neural network layer, which is commonly used in deep learning architectures like Transformers. The layer includes forward propagation and backward propagation for training. It is designed to work with multiple attention heads (nHeads) and processes data through two linear layers with a ReLU activation in between.

## Constants and Variables
### Constants
- `d_model`: Size of the model input/output vector (200)
- `d_ff`: Size of the hidden layer (800)
- `nHeads`: Number of attention heads (8)
- `learning_rate`: Learning rate for weight and bias updates (0.1)
### Variables
- `input`, `output`: Input and output matrices with size of (`nHeads`, `d_model`)
- `W_in`, `W_out`: Weight matrices for the input and output transformations with size of (`d_model`, `d_ff`) and (`d_ff`, `d_model`), respectively
- `b1`, `b2`: Bias vectors with size of (`nHeads`)
- `hidden` **(private)**: Intermediate matrix to store the ReLU output
- `d_output`, `d_hidden` **(private)**: Gradient matrices for backpropagation.

## Constructor
```cpp
FeedForward(float **input, float **output, float **W_in, float **W_out, float *b1, float *b2);
```
Initializes the FeedForward layer with pointers to input/output matrices, weight matrices, and biases.

## Method
### Forward
```cpp
void FeedForward::Forward();
```
Computes the forward pass of the Feedforward layer using this formulation
$$\text{hidden} = \max(0, \text{input} \cdot W_{in} + b_1)\\
\text{output} = \text{hidden} \cdot W_{out} + b_2$$
The `hidden` and `output` layer are saved for backward pass.

### Backward
```cpp
void Backward(float **target, float **d_input);
```
Computes the backward pass given the target layer and save the gradient of loss with respect to the input.
#### Parameters
`target`, `d_input`: Target matrix and gradient of the loss with respect to the input, both with size of (`nHeads`, `d_model`)
#### Theories:
Let $Y$, $\hat{Y}$, $H$ and $X$ be the target matrix, output matrix, hidden matrix and input matrix, respectively. Recall that
$$H = \max(0, X \cdot W_{in} + b_1)\\
Y = H \cdot W_{out} + b_2$$
Assuming the loss function is Cross-Entropy Loss Function $L$
$${L(\hat{Y}, Y) = \frac{\sum \hat{Y_{ij}}\ln (Y_{ij})}{mn}}$$
- **Output layer**'s derivative:
$$ \frac{\partial Y}{\partial L} = \frac{\hat{Y}}{mnY}$$

- $b_2$'s derivative:
$$ \frac{\partial b_{2_{i}}}{\partial L} = \frac{\partial b_{2_{i}}}{\partial Y_i} \cdot \frac{\partial Y_i}{\partial L}
= 1^{(1 \times m)} \cdot \frac{\partial Y_i}{\partial L} $$
$$\boxed{= \sum \frac{\partial Y_{ij}}{\partial L}}$$

- Denote $M_{*n}$ be the $n^\text{th}$ column vector of some matrix $M$, then $W_{out}$'s derivative:
$$ \frac{\partial W_{2_{ij}}}{\partial L} = \frac{\partial W_{2_{ij}}}{\partial Y_{*j}} 
\cdot \frac{\partial Y_{*j}}{\partial L} = (H^T)_i \cdot \frac{\partial Y_{*j}}{\partial L} $$
$$\implies 
\boxed{\frac{\partial W_2}{\partial L} = H^T \cdot \frac{\partial Y}{\partial L}}$$

- Let $M=X \cdot W_{in} + b_1$, then $M$'s derivative:
$$ \frac{\partial M_{ij}}{\partial L} = \frac{\partial M_{ij}}{\partial Y_i} \cdot \frac{\partial Y_i}{\partial L} =
((M_{ij} \gt 0 )W_{2_j})^{T} \cdot \frac{\partial Y_i}{\partial L} = (H_{ij} \gt 0)(W_{2_j})^T \cdot \frac{\partial Y_i}{\partial L}
$$
$$\implies \frac{\partial M}{\partial L}
= (H > 0) \circ (\frac{\partial Y}{\partial L} \cdot W_2^T)$$

- $b_1$'s derivative:
$$ \frac{\partial b_{1_i}}{\partial L} = \frac{\partial b_{1_i}}{\partial M_i} \cdot \frac{\partial M_i}{\partial L} = 1^{(1 \times p)} \cdot \frac{\partial M}{\partial L} 
$$
$$\boxed{= \sum \frac{\partial M_{ij}}{\partial L}}$$

- $W_{in}$'s derivative:
$$ \frac{\partial W_{1_{ij}}}{\partial L} = \frac{\partial W_{1_{ij}}}{\partial M_{*j}} \cdot \frac{\partial M_{*j}}{\partial L} = (X^T)_i \cdot \frac{\partial M_{*j}}{\partial L}
$$
$$\implies \boxed{\frac{\partial W_1}{\partial L} = X^T \cdot \frac{\partial M}{\partial L}} $$

- **Input Layer**'s derivative:
$$ \frac{\partial X_{ij}}{\partial L} = \frac{\partial X_{ij}}{\partial M_i} \cdot \frac{\partial M_i}{\partial L} = (W_{1_j})^T \cdot \frac{\partial M_i}{\partial L} 
$$
$$\implies \boxed{\frac{\partial X}{\partial L} = \frac{\partial M}{\partial L} \cdot W_1^T}$$

After computing derivative of each variable, we subtract the variable with said derivative times `learning_rate` to update it.

## Main Function (commented out)
The file includes `main` function (commented out) for generating random matrices and testing the forward and backward passes, along with other utility functions:
- `makeMatrix` and `makeArray`: Allocates and optionally initializes a matrix/array.
- `printMatrix` and `printArray`: Outputs a matrix/array.
- `deleteMatrix` and `deleteArray`: Frees allocated memory.
- `main`: Demo for `FeedForward` class