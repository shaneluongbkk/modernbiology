# Add&Norm Layer Implementation

This README provides an overview of the Add&Norm Layer implementation, focusing on its functionality, the core AddnNorm class, and the forward and backward methods. This layer combines residual connections and layer normalization to process inputs in a neural network layer.

## Class Overview: `AddnNorm`

### Attributes

1. **Inputs and Outputs:**
- `prev`: Input matrix from the previous layer.
- `orig`: Original input to this layer.
- `orig_grad`: Gradient matrix for `orig` (used in backpropagation).

2. **Parameters:**
- `gamma`: Scale parameter for layer normalization.
- `beta`: Shift parameter for layer normalization.

3. **Intermediate variables:**
- `w`: Store the normalized and transformed output.
- `mu`: Mean of the inputs for normalization.
- `sigma`: Standard deviation of the inputs for normalization.

4. **Dimensions:**
- `num_words`: Number of input sequences (rows).
- `N`: Length of each sequence (columns), or the dimension of the model.

## Methods

### 1. `forward()`

Computes the **forward pass** for the Add&Norm layer:

$\text{output} = \text{LayerNorm}(\text{prev} + \text{orig})$

#### Steps:
- Residual connection: Adds `prev` and `orig`.

    $w_{i,j} = \text{prev}_{i,j} + \text{orig}_{i,j}$

- Computes the mean (mu) and standard deviation (sigma) of the inputs.

    $\mu_{i} = \frac{1}{N} \sum_{j=0}^{N-1} w_{i,j}$

    $\sigma^2_{i} = \frac{1}{N} \sum_{j=0}^{N-1} w_{i,j}^2 - \mu_{i}^2$

- Normalizes and scales the inputs using `gamma` and `beta`.

    $w_{i,j} = \gamma[i] \cdot \frac{w_{i,j} - \mu_{i}}{\sigma_{i}} + \beta_{i}$

**Returns:** The normalized matrix `w` (size: `num_words x N`).

### 2. `backward(Matrix& dy, double lrate)`

$dy$: Gradient matrix for original input, $lrate$: learning rate.

Implements the **backward pass** for gradient updates:

1. Calculates:
   -  Gradients for `gamma` and `beta`.

        $\frac{\partial L}{\partial \gamma_{i}} = \sum_{j=0}^{N-1} \text{dy}_{i,j} \cdot w_{i,j}$

        $\frac{\partial L}{\partial \beta_{i}} = \sum_{j=0}^{N-1} \text{dy}_{i,j}$

   - Jacobian matrix the normalization operation.

        $
        \mathcal{J}_{i,j} =
        \begin{cases} 
        -\frac{1 + (z_i^{(n)} - \mu)(z_j^{(n)} - \mu)}{\sigma^2} & \text{if } i \neq j \\[10pt]
        N - 1 - \frac{(z_i^{(n)} - \mu)^2}{\sigma^2} & \text{if } i = j
        \end{cases}
        $

        $
        \mathcal{J} = N I_N - 1_{N\times N} - w^{(n)} \otimes w^{(n)}
        $

        Where $\mathbf{I}_N$ is the identity matrix, $1_{N\times N}$ is the $N\times N$ matrix filled with 1s and $\otimes$ is the outer product. 

2. Updates `gamma` and `beta` using gradient descent.

    $\gamma_{i} -= \text{lrate} \times \frac{\partial \mathcal{L}}{\partial \gamma_{i}}$

    $\beta{i} -= \text{lrate} \times \frac{\partial \mathcal{L}}{\partial \beta_{i}}$

3. Calculates matrix backward `dz` and update to `orig_grid`
    $dz = \frac{\gamma}{N \sigma} \left( dy \times \mathcal{J} \right)$

    $\text{orig\_ grid}+=dz$


**Returns:** Gradient matrix `dz` for the inputs.

**Note:** For detailed equations and calculations, read the file backwards.pdf. You can view the file [here](./backwards.pdf).

## Usage

### **Initialization**

```cpp
Matrix prev = ...; // Input from previous layer
Matrix orig = ...; // Original input
Matrix orig_grad = ...; // Zero-initialized gradient matrix

AddnNorm addn(prev, orig, orig_grad, num_words, N);