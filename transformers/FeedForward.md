# Documentation for `FeedForward.cpp`

### Overview
The `FeedForward.cpp` file implements a FeedForward neural network layer, which is commonly used in deep learning architectures like Transformers. The layer includes forward propagation and backward propagation for training, and processes data through two linear layers with a ReLU activation in between.
## Constants and Variables
### Constants
- `d_model`: Size of the model input/output vector (200)
- `d_ff`: Size of the hidden layer (800)
- `num_tokens`: Number of tokens/words (8)
- `learning_rate`: Learning rate for weight and bias updates (0.1)
### Variables
- `input`, `output`: Input and output matrices with size of (`num_tokens`, `d_model`)
- `W_in`, `W_out`: Weight matrices for the input and output transformations with size of (`d_model`, `d_ff`) and (`d_ff`, `d_model`), respectively
- `b1`, `b2`: Bias vectors with size of (`d_ff`) and (`d_model`) respectively
- `hidden` **(private)**: Intermediate matrix to store the ReLU output
- `d_output`, `d_hidden` **(private)**: Gradient matrices for backpropagation.

Apart from `learning_rate`, all bound-checkings rely on the constants, as well as the sizes of the matrices/vectors matching with these constants. If their sizes do not match with the constants or change during runtime, or the constants change values during runtime, it may causes memory errors.

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
```math
 \text{hidden} = \max(0, \text{input} \cdot W_{in} + b_1)
```
```math
\text{output} = \text{hidden} \cdot W_{out} + b_2
```
The `hidden` and `output` layer are saved for backward pass.

### Backward
```cpp
void Backward(float **target, float **d_input);
```
Computes the backward pass given the `target` layer and save the gradient of loss with respect to input into `d_input`.
#### Parameters
`target`, `d_input`: Target matrix and gradient of loss with respect to input, both with size of (`num_tokens`, `d_model`)
#### Theories:
Let $`Y`$, $`\hat{Y}`$, $`H`$ and $`X`$ be the output matrix, target matrix, hidden matrix and input matrix, respectively. Recall that
```math
H = \max(0, X \cdot W_{in} + b_1)
```
```math
Y = H \cdot W_{out} + b_2
```
Let $`m = \text{num\_tokens}`$ and $`n = d_{model}`$. Assuming the loss function is Cross-Entropy Loss Function $L$
```math
{L(\hat{Y}, Y) = \frac{\sum \hat{Y_{ij}}\ln (Y_{ij})}{mn}}
```
- **Output layer**'s derivative:
```math
 \frac{\partial Y}{\partial L} = \frac{\hat{Y}}{mnY}
```

- Denote $M_{*n}$ be the $n^\text{th}$ column vector of some matrix $M$, then $b_2$'s derivative:
```math
 \frac{\partial b_{2_{i}}}{\partial L} = \frac{\partial b_{2_{i}}}{\partial Y_{*i}} \cdot \frac{\partial Y_{*i}}{\partial L}
= 1^{(1 \times m)} \cdot \frac{\partial Y_{*i}}{\partial L}
```
```math
\boxed{= \sum \frac{\partial Y_{ji}}{\partial L}}
```

- $W_{out}$'s derivative:
```math
\frac{\partial W_{2_{ij}}}{\partial L} = \frac{\partial W_{2_{ij}}}{\partial Y_{*j}} 
\cdot \frac{\partial Y_{*j}}{\partial L} = (H^T)_i \cdot \frac{\partial Y_{*j}}{\partial L}
```
```math
\implies 
\boxed{\frac{\partial W_2}{\partial L} = H^T \cdot \frac{\partial Y}{\partial L}}
```

- Let $M=X \cdot W_{in} + b_1$, then $M$'s derivative:
```math
\frac{\partial M_{ij}}{\partial L} = \frac{\partial M_{ij}}{\partial Y_i} \cdot \frac{\partial Y_i}{\partial L} =
((M_{ij} \gt 0 )W_{2_j})^{T} \cdot \frac{\partial Y_i}{\partial L} = (H_{ij} \gt 0)(W_{2_j})^T \cdot \frac{\partial Y_i}{\partial L}
```
```math
\implies \frac{\partial M}{\partial L}
= (H > 0) \circ (\frac{\partial Y}{\partial L} \cdot W_2^T)
```

- $b_1$'s derivative:
```math
\frac{\partial b_{1_i}}{\partial L} = \frac{\partial b_{1_i}}{\partial M_{*i}} \cdot \frac{\partial M_{*i}}{\partial L} = 1^{(1 \times m)} \cdot \frac{\partial M_{*i}}{\partial L} 
```
```math
\boxed{= \sum \frac{\partial M_{ji}}{\partial L}}
```

- $W_{in}$'s derivative:
```math
\frac{\partial W_{1_{ij}}}{\partial L} = \frac{\partial W_{1_{ij}}}{\partial M_{*j}} \cdot \frac{\partial M_{*j}}{\partial L} = (X^T)_i \cdot \frac{\partial M_{*j}}{\partial L}
```
```math
\implies \boxed{\frac{\partial W_1}{\partial L} = X^T \cdot \frac{\partial M}{\partial L}}
```

- **Input Layer**'s derivative:
```math
\frac{\partial X_{ij}}{\partial L} = \frac{\partial X_{ij}}{\partial M_i} \cdot \frac{\partial M_i}{\partial L} = (W_{1_j})^T \cdot \frac{\partial M_i}{\partial L} 
```
```math
\implies \boxed{\frac{\partial X}{\partial L} = \frac{\partial M}{\partial L} \cdot W_1^T}
```

After computing derivative of each variable, we subtract the variable with said derivative times `learning_rate` to update it.

## Main Function (commented out)
The file includes `main` function (commented out) for generating random matrices and testing the forward and backward passes, along with other utility functions:
- `makeMatrix` and `makeArray`: Allocates and optionally initializes a matrix/array.
- `printMatrix` and `printArray`: Outputs a matrix/array.
- `deleteMatrix` and `deleteArray`: Frees allocated memory.
- `main`: Demo for `FeedForward` class