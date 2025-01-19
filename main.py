import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
import scanpy as sc

class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers=1, n_hidden=128, n_latent=10, dropout_rate=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        
        layers = [nn.Linear(input_dim, n_hidden),
                 nn.BatchNorm1d(n_hidden),
                 nn.ReLU(),
                 nn.Dropout(p=dropout_rate)]
        
        for _ in range(n_layers-1):
            layers.extend([
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            
        self.encoder = nn.Sequential(*layers)
        
        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_var = nn.Linear(n_hidden, n_latent)
        
        self.l_mean = nn.Linear(n_hidden, 1)
        self.l_var = nn.Linear(n_hidden, 1)
        
    def forward(self, x):

        # Encode through the shared layers
        q = self.encoder(x)
        
        # Get latent space parameters
        z_mean = self.z_mean(q)
        z_var = torch.exp(self.z_var(q))  # Ensure positive variance
        
        # Get library size parameters
        l_mean = self.l_mean(q)
        l_var = torch.exp(self.l_var(q))  # Ensure positive variance
        
        return z_mean, z_var, l_mean, l_var

class Decoder(nn.Module):

    def __init__(self, n_latent, n_layers=1, n_hidden=128, output_dim=None, dropout_rate=0.1):
        super().__init__()
        
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        
        layers = [nn.Linear(n_latent, n_hidden),
                 nn.BatchNorm1d(n_hidden),
                 nn.ReLU(),
                 nn.Dropout(p=dropout_rate)]
        
        for _ in range(n_layers-1):
            layers.extend([
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            
        self.decoder = nn.Sequential(*layers)
        
        self.px_scale = nn.Linear(n_hidden, output_dim)  
        self.px_r = nn.Linear(n_hidden, output_dim)      
        self.px_dropout = nn.Linear(n_hidden, output_dim) 
        
    def forward(self, z):

        h = self.decoder(z)
        
        # Get ZINB parameters
        px_scale = F.softmax(self.px_scale(h), dim=-1)  # px_scale (torch.Tensor): Mean parameter of ZINB
        px_r = torch.exp(self.px_r(h))                  # px_r (torch.Tensor): Dispersion parameter of ZINB
        px_dropout = self.px_dropout(h)                 # px_dropout (torch.Tensor): Dropout parameter of ZINB
        
        return px_scale, px_r, px_dropout

class scVI(nn.Module):
    
    def __init__(self, input_dim, n_batches=0, n_hidden=128, n_latent=10, 
                 n_layers=1, dropout_rate=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_batches = n_batches
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        
        # Initialize encoder and decoder networks
        self.encoder = Encoder(input_dim=input_dim, n_layers=n_layers,
                             n_hidden=n_hidden, n_latent=n_latent,
                             dropout_rate=dropout_rate)
        
        self.decoder = Decoder(n_latent=n_latent, n_layers=n_layers,
                             n_hidden=n_hidden, output_dim=input_dim,
                             dropout_rate=dropout_rate)
    
    def sample_from_posterior(self, mean, var):
      
        return mean + var.sqrt() * torch.randn_like(mean)

    def forward(self, x, batch_index=None):

        z_mean, z_var, l_mean, l_var = self.encoder(x)

        z = self.sample_from_posterior(z_mean, z_var)
        library = self.sample_from_posterior(l_mean, l_var)

        px_scale, px_r, px_dropout = self.decoder(z)

        # Calculate the negative binomial mean
        library = library.exp().view(-1, 1)  # Transform from log space and add dimension
        px_rate = library * px_scale  # Element-wise multiplication

        return {
            'z_mean': z_mean,
            'z_var': z_var,
            'l_mean': l_mean,
            'l_var': l_var,
            'z': z,
            'library': library,
            'px_scale': px_scale,
            'px_r': px_r,
            'px_dropout': px_dropout,
            'px_rate': px_rate  
        }

    def generate_data(self, n_samples: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # Sample from standard normal for latent space
            z = torch.randn(n_samples, self.n_latent).to(next(self.parameters()).device)
            
            # Get ZINB parameters from decoder
            px_scale, px_r, px_dropout = self.decoder(z)
            
            library = torch.exp(torch.randn(n_samples, 1).to(next(self.parameters()).device))
            
            # Calculate mean parameter
            px_rate = library * px_scale
            
            # Generate data using ZINB distribution
            # First generate dropout mask
            dropout_mask = torch.bernoulli(torch.sigmoid(-px_dropout))
            
            # Then generate NB samples
            theta = px_r
            p = px_rate / (px_rate + theta)
            nb_samples = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=theta,
                probs=1-p  # NB distribution in PyTorch uses failure probability
            ).sample()
            
            # Apply dropout mask
            data = dropout_mask * nb_samples
            
            return data
    
class ZINBLoss:
    def __init__(self):
        super().__init__()

    def negative_binomial_loss(self, x: torch.Tensor, mu: torch.Tensor, 
                             theta: torch.Tensor, eps=1e-8) -> torch.Tensor:
        log_theta_mu_eps = torch.log(theta + mu + eps)
        
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        
        return res

    def __call__(self, x: torch.Tensor, px_scale: torch.Tensor, 
                 px_rate: torch.Tensor, px_r: torch.Tensor, 
                 px_dropout: torch.Tensor) -> torch.Tensor:

        nb_case = self.negative_binomial_loss(x, px_rate, px_r)

        zero_inflation = -F.softplus(px_dropout)
   
        zero_case = zero_inflation + self.negative_binomial_loss(torch.zeros_like(x), px_rate, px_r)
        final_case = torch.where(x < 1e-8, zero_case, nb_case + zero_inflation)
        
        return -torch.sum(final_case, dim=-1)

class scVITrainer:
    def __init__(
        self,
        model: "scVI",
        adata,
        train_size: float = 0.9,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_epochs: int = 400,
        use_gpu: bool = True,
        kl_weight: float = 1.0,
    ):
        self.model = model
        self.adata = adata
        self.train_size = train_size
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.kl_weight = kl_weight
        
        # Setup device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup loss functions
        self.zinb_loss = ZINBLoss()
        
        # Create data loaders
        self._setup_data_loaders()
        
    def _setup_data_loaders(self):
        """
        Prepare training and validation data loaders
        """
        # Convert data to torch tensors
        X = torch.FloatTensor(self.adata.X.toarray())
        
        # Split into train and validation
        n_train = int(self.train_size * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        train_data = X[train_idx]
        val_data = X[val_idx]
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False
        )

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Compute ZINB loss
        zinb_loss = self.zinb_loss(
            x,
            outputs["px_scale"],
            outputs["px_rate"],
            outputs["px_r"],
            outputs["px_dropout"]
        )
        
        # Compute KL divergence for latent space
        kl_divergence_z = -0.5 * torch.sum(
            1 
            + outputs["z_var"].log()
            - outputs["z_mean"].pow(2)
            - outputs["z_var"],
            dim=1
        )
        
        # Compute KL divergence for library size
        kl_divergence_l = -0.5 * torch.sum(
            1 
            + outputs["l_var"].log()
            - outputs["l_mean"].pow(2)
            - outputs["l_var"],
            dim=1
        )
        
        # Combine losses
        kl_local = kl_divergence_z + kl_divergence_l
        weighted_kl = self.kl_weight * kl_local
        
        total_loss = zinb_loss.mean() + weighted_kl.mean()
        
        return {
            "loss": total_loss,
            "zinb_loss": zinb_loss.mean(),
            "kl_local": kl_local.mean(),
            "kl_divergence_z": kl_divergence_z.mean(),
            "kl_divergence_l": kl_divergence_l.mean(),
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_losses = []
        
        for batch_idx, x in enumerate(self.train_loader):
            x = x.to(self.device)
        
            self.optimizer.zero_grad()
            outputs = self.model(x)
         
            loss_dict = self.compute_loss(x, outputs)
            loss = loss_dict["loss"]
          
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        return {
            k: torch.stack([d[k] for d in epoch_losses]).mean().item()
            for k in epoch_losses[0].keys()
        }

    def train(self, n_epochs: Optional[int] = None) -> Dict[str, list]:
      
        n_epochs = n_epochs or self.max_epochs
        history = {
            "train_loss": [],
            "val_loss": [],
            "kl_divergence": [],
            "zinb_loss": []
        }
        
        for epoch in range(n_epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Store metrics
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["kl_divergence"].append(train_metrics["kl_local"])
            history["zinb_loss"].append(train_metrics["zinb_loss"])
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{n_epochs}, "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                
        return history
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x in self.val_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                loss_dict = self.compute_loss(x, outputs)
                val_losses.append(loss_dict)
        
        return {
            k: torch.stack([d[k] for d in val_losses]).mean().item()
            for k in val_losses[0].keys()
        }


# Example usage:
def prepare_data():
  
    print("Loading and preprocessing data...")
    
    # Load example dataset (Paul et al., 2015)
    adata = sc.datasets.pbmc3k()
    
    # Basic preprocessing steps
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate quality metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Normalize total counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Logarithmize the data
    sc.pp.log1p(adata)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
    # Keep only highly variable genes
    adata = adata[:, adata.var.highly_variable]
    
    print(f"Final data shape: {adata.shape}")
    return adata

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load and preprocess data
    adata = prepare_data()
    
    # 2. Initialize model
    n_genes = adata.n_vars
    model = scVI(
        input_dim=n_genes,
        n_latent=10,
        n_hidden=128,
        n_layers=2,
        dropout_rate=0.1
    )
    
    # 3. Create trainer
    trainer = scVITrainer(
        model=model,
        adata=adata,
        train_size=0.9,
        batch_size=128,
        lr=1e-3,
        max_epochs=100,
        use_gpu=True
    )

    generative_data = model.generate_data(n_samples=1000)
    
    # 4. Train the model
    print("Training model...")
    history = trainer.train()
    return model, adata, history, generative_data

if __name__ == "__main__":
    model, adata, history, new_data = main()