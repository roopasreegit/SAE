 
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional
from tqdm import tqdm
 
from sae import SparseAutoencoder
 
 
 
def train_sae(
    activations:    torch.Tensor,
    d_hidden:       int   = 6144,
    l1_coeff:       float = 2e-4,
    lr:             float = 1e-4,
    batch_size:     int   = 4096,
    n_steps:        int   = 30_000,
    log_every:      int   = 500,
    save_dir:       str   = "results",
    device:         Optional[str] = None,
) -> SparseAutoencoder:
    """
    Train a SparseAutoencoder on pre-collected activations.
 
    Args:
        activations : (N, d_model) FloatTensor of GPT-2 residual-stream vectors
        d_hidden    : SAE hidden dimension (use 8 × d_model)
        l1_coeff    : L1 sparsity penalty weight λ
        lr          : AdamW learning rate
        batch_size  : activations per gradient step
        n_steps     : total training steps
        log_every   : logging frequency
        save_dir    : where to save the checkpoint and metrics
        device      : 'cuda' | 'cpu' | None (auto-detect)
 
    Returns:
        trained SparseAutoencoder
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    d_model = activations.shape[1]
    N       = activations.shape[0]
 
    print(f"\n{'─'*60}")
    print(f"Training SAE | d_model={d_model} d_hidden={d_hidden} λ={l1_coeff}")
    print(f"  {N:,} activations | batch={batch_size} | steps={n_steps:,} | device={device}")
    print(f"{'─'*60}")
 
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
 
    dataset = TensorDataset(activations)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
 
    steps_since_fired = torch.zeros(d_hidden, dtype=torch.long)
    DEAD_THRESHOLD    = 1_000  # steps without firing = dead feature
 
    metrics = {
        "step": [], "loss": [], "recon_loss": [], "l1_loss": [],
        "avg_l0": [], "var_explained": [], "dead_features": [],
    }
 
    step        = 0
    epoch       = 0
    log_loss    = log_recon = log_l1 = 0.0
    t0          = time.time()
 
    pbar = tqdm(total=n_steps, desc="Training SAE", unit="step")
 
    while step < n_steps:
        epoch += 1
        for (batch_acts,) in loader:
            if step >= n_steps:
                break
 
            batch_acts = batch_acts.to(device)
 
            optimizer.zero_grad()
            out = sae(batch_acts)
            out.loss.backward()
            optimizer.step()
 
            sae.normalize_decoder_()
 
            with torch.no_grad():
                fired = (out.h > 0).any(dim=0).cpu()   # (d_hidden,) bool
                steps_since_fired[fired]  = 0
                steps_since_fired[~fired] += 1
 
            log_loss  += out.loss.item()
            log_recon += out.recon_loss.item()
            log_l1    += out.l1_loss.item()
 
            step += 1
            pbar.update(1)
 
            if step % log_every == 0:
                n_dead   = int((steps_since_fired >= DEAD_THRESHOLD).sum().item())
                avg_l0   = sae.avg_l0(batch_acts)
                var_expl = sae.explained_variance(batch_acts)
 
                metrics["step"].append(step)
                metrics["loss"].append(log_loss / log_every)
                metrics["recon_loss"].append(log_recon / log_every)
                metrics["l1_loss"].append(log_l1 / log_every)
                metrics["avg_l0"].append(avg_l0)
                metrics["var_explained"].append(var_expl)
                metrics["dead_features"].append(n_dead)
 
                elapsed = time.time() - t0
                pbar.set_postfix({
                    "loss":  f"{log_loss/log_every:.3f}",
                    "L0":    f"{avg_l0:.1f}",
                    "R²":    f"{var_expl:.2f}",
                    "dead":  f"{n_dead}",
                })
                print(
                    f"\nStep {step:6d} | loss={log_loss/log_every:.4f} "
                    f"recon={log_recon/log_every:.4f} l1={log_l1/log_every:.5f} | "
                    f"L0={avg_l0:.1f}  R²={var_expl:.3f}  dead={n_dead}/{d_hidden} | "
                    f"{elapsed:.0f}s"
                )
                log_loss = log_recon = log_l1 = 0.0
 
    pbar.close()
 
    ckpt_path    = os.path.join(save_dir, "sae.pt")
    metrics_path = os.path.join(save_dir, "training_metrics.json")
 
    sae.save(ckpt_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\nSaved SAE → {ckpt_path}")
    print(f"Saved metrics → {metrics_path}")
 
    with torch.no_grad():
        sample = activations[:4096].to(device)
        final_l0   = sae.avg_l0(sample)
        final_r2   = sae.explained_variance(sample)
        final_dead = int((steps_since_fired >= DEAD_THRESHOLD).sum())
 
    print(f"\n{'─'*60}")
    print(f"Training complete")
    print(f"  Avg L0 (active features/token): {final_l0:.1f} / {d_hidden}")
    print(f"  Variance explained:             {final_r2:.1%}")
    print(f"  Dead features:                  {final_dead} / {d_hidden} "
          f"({100*final_dead/d_hidden:.1f}%)")
    print(f"{'─'*60}")
 
    return sae
 
 
 
def plot_training(metrics_path: str = "results/training_metrics.json",
                  save_path:    str = "results/training_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    with open(metrics_path) as f:
        m = json.load(f)
 
    steps = m["step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SAE Training Curves", fontsize=14)
 
    axes[0, 0].plot(steps, m["loss"],       color="#1565C0", linewidth=2)
    axes[0, 0].set_title("Total loss"); axes[0, 0].set_yscale("log")
 
    axes[0, 1].plot(steps, m["recon_loss"], color="#2E7D32", linewidth=2)
    axes[0, 1].set_title("Reconstruction loss (MSE)"); axes[0, 1].set_yscale("log")
 
    axes[1, 0].plot(steps, m["avg_l0"],       color="#D84315", linewidth=2)
    axes[1, 0].set_title("Average L0 (features active per token)")
    axes[1, 0].axhline(50, color="gray", linestyle="--", alpha=0.5, label="target ≈ 50")
    axes[1, 0].legend()
 
    axes[1, 1].plot(steps, m["var_explained"], color="#6A1B9A", linewidth=2)
    axes[1, 1].set_title("Variance explained (R²)")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
 
    for ax in axes.flat:
        ax.set_xlabel("Step"); ax.grid(alpha=0.25)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {save_path}")