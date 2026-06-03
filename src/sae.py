import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
 
 
class SAEOutput(NamedTuple):
    x_hat:      torch.Tensor   # reconstructed activation (batch, d_model)
    h:          torch.Tensor   # sparse feature activations (batch, d_hidden)
    loss:       torch.Tensor   # total loss (scalar)
    recon_loss: torch.Tensor   # MSE reconstruction loss (scalar)
    l1_loss:    torch.Tensor   # L1 sparsity loss (scalar)
 
 
class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for mechanistic interpretability.
 
    Args:
        d_model  : input dimension (768 for GPT-2 small)
        d_hidden : hidden dimension — use 8× expansion (6144)
        l1_coeff : weight of L1 sparsity penalty (λ). Controls the
                   sparsity–reconstruction tradeoff. Start at 2e-4.
    """
 
    def __init__(self, d_model: int, d_hidden: int, l1_coeff: float = 2e-4):
        super().__init__()
        self.d_model  = d_model
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
 
        # Encoder matrix:  (d_model, d_hidden)
        self.W_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
 
        # Decoder matrix: (d_model, d_hidden)
        # Column i = the "dictionary atom" for feature i
        self.W_dec = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
 
        # ── Initialise ────────────────────────────────────────────────────
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self.normalize_decoder_()  # enforce unit-norm columns from the start
 
 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, d_model)
        → h : (batch, d_hidden)   sparse, non-negative feature activations
        """
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
 
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (batch, d_hidden)
        → x_hat : (batch, d_model)   reconstructed activation
        """
        # W_dec is (d_model, d_hidden), so W_dec.T is (d_hidden, d_model)
        return h @ self.W_dec.T + self.b_dec
 
    def forward(self, x: torch.Tensor) -> SAEOutput:
        h     = self.encode(x)
        x_hat = self.decode(h)
 
        # Reconstruction loss: mean over batch, sum over d_model
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
 
        # Sparsity loss: L1 norm of hidden activations
        l1_loss = self.l1_coeff * h.abs().sum(dim=-1).mean()
 
        return SAEOutput(
            x_hat=x_hat,
            h=h,
            loss=recon_loss + l1_loss,
            recon_loss=recon_loss,
            l1_loss=l1_loss,
        )
 
 
    @torch.no_grad()
    def normalize_decoder_(self):
        """
        Normalise each column of W_dec to unit L2 norm (in-place).
        Must be called after every optimiser step.
        Without this, the model can cheat the L1 penalty.
        """
        col_norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= col_norms
 
 
    @torch.no_grad()
    def explained_variance(self, x: torch.Tensor) -> float:
        """Fraction of variance in x explained by the reconstruction."""
        x_hat = self.forward(x).x_hat
        total_var    = (x - x.mean(dim=0)).pow(2).sum()
        residual_var = (x - x_hat).pow(2).sum()
        return 1.0 - (residual_var / total_var).item()
 
    @torch.no_grad()
    def avg_l0(self, x: torch.Tensor) -> float:
        """Average number of active features per token (L0 norm)."""
        h = self.encode(x)
        return (h > 0).float().sum(dim=-1).mean().item()
 
    @torch.no_grad()
    def dead_feature_count(self, x: torch.Tensor) -> int:
        """Number of features that produce zero activation on this batch."""
        h = self.encode(x)
        return int((h.sum(dim=0) == 0).sum().item())
 
 
    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "d_model":    self.d_model,
            "d_hidden":   self.d_hidden,
            "l1_coeff":   self.l1_coeff,
        }, path)
 
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            d_model=ckpt["d_model"],
            d_hidden=ckpt["d_hidden"],
            l1_coeff=ckpt["l1_coeff"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)
 