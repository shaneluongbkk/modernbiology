# Add&Norm Layer Implementation

This README provides an overview of the Add&Norm Layer implementation, focusing on its functionality, the core AddnNorm class, and the forward and backward methods. This layer combines residual connections, layer normalization to process inputs in a neural network layer, as well as and dropout regularization to improve robustness and generalization in the layers.

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
- `dropout_mask`: Binary mask applied to neurons for dropout regularization.

4. **Dimensions:**
- `num_words`: Number of input sequences (rows).
- `N`: Length of each sequence (columns), or the dimension of the model.

5. **Dropout:**
- `dropout_rate`: Probability of dropping a neuron during training.

## Methods

### 1. `forward()`

Computes the **forward pass** for the Add&Norm layer:
```math
\text{output} = \text{LayerNorm}(\text{prev} + \text{orig})
```

#### Steps:
- Residual connection: Adds `prev` and `orig`.
```math
w_{i,j} = \text{prev}_{i,j} + \text{orig}_{i,j}
```

- Computes the mean (mu) and standard deviation (sigma) of the inputs.

    ```math
    \mu_{i} = \frac{1}{N} \sum_{j=0}^{N-1} w_{i,j}
    ```

    ```math
    \sigma^2_{i} = \frac{1}{N} \sum_{j=0}^{N-1} w_{i,j}^2 - \mu_{i}^2
    ```

- Normalizes and scales the inputs using `gamma` and `beta`.

    ```math
    w_{i,j} = \gamma[i] \cdot \frac{w_{i,j} - \mu_{i}}{\sigma_{i}} + \beta_{i}
    ```

- Dropout regularization (during training):

    A dropout mask is applied to deactivate neurons with a probability equal to `dropout_rate`, ensures that only a subset of neurons contribute during the forward pass.

```cpp
dropout_mask[i][j] = static_cast<double>(rand()) / RAND_MAX > dropout_rate ? 1.0 : 0.0;
w[i][j] *= dropout_mask[i][j];
// Scale active neurons to maintain their contribution
w[i][j] = w[i][j] / (1 - dropout_rate);
```

- During inference (`is_training=false`), all neurons remain active (`dropout_mask=1.0` for all).

**Returns:** The normalized matrix `w` (size: `num_words x N`).

### 2. `backward(Matrix& dy, double lrate)`

$dy$: Gradient matrix for original input, $lrate$: learning rate.

Implements the **backward pass** for gradient updates:

1. Calculates:
   -  Gradients for `gamma` and `beta`.

        ```math
        \frac{\partial L}{\partial \gamma_{i}} = \sum_{j=0}^{N-1} \text{dy}_{i,j} \cdot w_{i,j}
        ```

        ```math
        \frac{\partial L}{\partial \beta_{i}} = \sum_{j=0}^{N-1} \text{dy}_{i,j}
        ```

   - Jacobian matrix the normalization operation.

        ```math
        \mathcal{J}_{i,j} =
        \begin{cases} 
        -\frac{1 + (z_i^{(n)} - \mu)(z_j^{(n)} - \mu)}{\sigma^2} & \text{if } i \neq j \\[10pt]
        N - 1 - \frac{(z_i^{(n)} - \mu)^2}{\sigma^2} & \text{if } i = j
        \end{cases}
        ```

        ```math
        \mathcal{J} = N I_N - 1_{N\times N} - w^{(n)} \otimes w^{(n)}
        ```

        Where $\mathbf{I}_N$ is the identity matrix, $1_{N\times N}$ is the $N\times N$ matrix filled with 1s and $\otimes$ is the outer product. 

2. Updates `gamma` and `beta` using gradient descent.

    ```math
    \gamma_{i} -= \text{lrate} \times \frac{\partial \mathcal{L}}{\partial \gamma_{i}}
    ```

    ```math
    \beta{i} -= \text{lrate} \times \frac{\partial \mathcal{L}}{\partial \beta_{i}}
    ```

3. Calculates matrix backward `dz` and update to `orig_grid`
    ```math
    dz = \frac{\gamma}{N \sigma} \left( dy \times \mathcal{J} \right)
    ```

    ```math
    \text{orig\_ grid}+=dz
    ```


**Returns:** Gradient matrix `dz` for the inputs.

**Note:** For detailed equations and calculations, read the file backwards.pdf. You can view the file [here](./backwards.pdf).

## Usage

### **Initialization**

```cpp
Matrix prev = ...; // Input from previous layer
Matrix orig = ...; // Original input
Matrix orig_grad = ...; // Zero-initialized gradient matrix

AddnNorm addn(prev, orig, orig_grad, num_words, N);
```

### **Forward Pass with Dropout**
```cpp
Matrix output = addn.forward(true); // Training mode (dropout active)
Matrix output_test = addn.forward(false); // Testing mode (no dropout)
```

### **Backward Pass**
```cpp
Matrix dy = ...; // Gradient from the next layer
Matrix dz = addn.backward(dy); // Backpropagation with learning rate 0.1
```
