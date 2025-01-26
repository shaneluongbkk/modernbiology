# Single-cell Variational Inference (scVI) Implementation

scVI is a variational autoencoder that learns a probabilistic representation of gene expression data, enabling various downstream tasks such as dimensionality reduction, clustering, and differential expression analysis.

## Key Components

#### 1. `Encoder`
- **Input**: Gene expression matrix
- **Output**: Parameters of latent space distributions and library size distribution (z_mean, z_var, l_mean, l_var)
- **Role**: The Encoder compresses high-dimensional gene expression data into a low-dimensional latent space
- **Parameters**:
  - `input_dim`: Number of input genes
  - `n_layers`: Number of hidden layers (default: 1)
  - `n_hidden`: Number of hidden units (default: 128)
  - `n_latent`: Dimension of latent space (default: 10)
  - `dropout_rate`: Dropout probability (default: 0.1)

#### 2. `Decoder`
- **Input**: Latent representation
- **Output**: Parameters of ZINB distribution (px_scale, px_r, px_dropout)
- **Role**: The Decoder transforms data from the latent space (created by the Encoder) back into gene expression data
- **Parameters**:
  - `n_latent`: Dimension of latent space
  - `n_layers`: Number of hidden layers
  - `n_hidden`: Number of hidden units
  - `output_dim`: Number of genes
  - `dropout_rate`: Dropout probability

#### 3. `scVI`
Main model class that combines encoder and decoder
- **Key Methods**:
  - `forward()`: Processes input through encoder and decoder
  - `generate_data()`: Generates synthetic data from the model
- **Parameters**: Combines parameters from Encoder and Decoder

#### 4.1 `ZINBLoss`
Implements Zero-Inflated Negative Binomial loss function
A mixture model combining:
- Negative Binomial (NB) for count data
- Bernoulli for excess zeros

**Formula:**
```
P(X = k) =
  \pi + (1 - \pi) f_NB(0), if k = 0
  (1 - \pi) f_NB(k),       if k > 0
```
- \(\pi\): Dropout probability.

#### 4.2 `KL Loss`
KL Loss measures the difference between the learned latent distribution \( q(z|x) \) and a prior distribution \( p(z) \) (for latent space and library size)

**Formula:**
```
KL(q(z|x) || p(z)) = -\frac{1}{2} \sum_{j=1}^J \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)
```
where:
- \( \mu_j \): Mean of the learned latent distribution for dimension \( j \).
- \( \sigma_j^2 \): Variance of the learned latent distribution for dimension \( j \).
- \( J \): Dimension of the latent space.

#### 5. `scVITrainer`
Handles model training and validation
- **Key Methods**:
  - `train()`: Trains model for specified epochs
  - `validate()`: Performs validation
  - `compute_loss()`: Calculates total loss including KL divergence
- **Parameters**:
  - `model`: scVI model instance
  - `adata`: AnnData object containing gene expression data
  - `train_size`: Training data fraction (default: 0.9)
  - `batch_size`: Batch size (default: 128)
  - `lr`: Learning rate (default: 1e-3)
  - `max_epochs`: Maximum training epochs (default: 400)
  - `use_gpu`: Whether to use GPU (default: True)
  - `kl_weight`: Weight for KL divergence term (default: 1.0)

### Helper Functions

#### `prepare_data()`
Preprocesses single-cell RNA data:
- Filters cells and genes
- Computes quality metrics
- Normalizes data
- Identifies variable genes

#### `main()`
The entire workflow:
1. Prepares data
2. Initializes model
3. Creates trainer
4. Trains model

## Data Input Format

The model expects input data as an AnnData object containing:
- Raw count matrix in `.X`
- Gene names in `.var`
- Cell metadata in `.obs`
