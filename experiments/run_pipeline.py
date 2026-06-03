import sys
import os
import argparse
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import torch
 
 
 
CONFIG = {
    # Activation collection
    "model_name":    "gpt2",
    "layer_idx":     8,           # hook GPT-2 layer 8 (of 12)
    "n_tokens":      200_000,     # tokens to collect
    "context_len":   128,
    "collect_batch": 32,          # sequences per GPT-2 forward pass
 
    # SAE architecture
    "expansion":     8,           # d_hidden = d_model × expansion = 6144
    "l1_coeff":      2e-4,        # sparsity penalty λ
 
    # SAE training
    "lr":            1e-4,
    "batch_size":    4096,
    "n_steps":       30_000,
    "log_every":     500,
 
    # Analysis
    "top_k":         20,          # examples to save per feature
    "context_radius": 15,         # tokens of context around the activating token
    "n_features":    500,         # how many features to save in the JSON
 
    # Paths
    "data_dir":    "data",
    "results_dir": "results",
}
 
 
 
def step_collect(cfg, force=False, device=None):
    from activations import collect_activations
    print("\n" + "=" * 60)
    print("STEP 1 — Collecting activations")
    print("=" * 60)
    acts, toks = collect_activations(
        model_name=cfg["model_name"],
        layer_idx=cfg["layer_idx"],
        n_tokens=cfg["n_tokens"],
        context_len=cfg["context_len"],
        batch_size=cfg["collect_batch"],
        save_dir=cfg["data_dir"],
        device=device,
        overwrite=force,
    )
    return acts, toks
 
 
def step_train(cfg, activations, force=False, device=None):
    from train import train_sae
    print("\n" + "=" * 60)
    print("STEP 2 — Training SAE")
    print("=" * 60)
 
    ckpt_path = os.path.join(cfg["results_dir"], "sae.pt")
    if not force and os.path.exists(ckpt_path):
        from sae import SparseAutoencoder
        print(f"Loading existing SAE from {ckpt_path}")
        return SparseAutoencoder.load(ckpt_path, device=device or "cpu")
 
    d_hidden = activations.shape[1] * cfg["expansion"]
    sae = train_sae(
        activations=activations,
        d_hidden=d_hidden,
        l1_coeff=cfg["l1_coeff"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        n_steps=cfg["n_steps"],
        log_every=cfg["log_every"],
        save_dir=cfg["results_dir"],
        device=device,
    )
    return sae
 
 
def step_analyze(cfg, sae, activations, tokens, force=False, device=None):
    from analyze import analyze_features, print_top_features
    print("\n" + "=" * 60)
    print("STEP 3 — Analysing features")
    print("=" * 60)
 
    analysis_path = os.path.join(cfg["results_dir"], "feature_analysis.json")
    if not force and os.path.exists(analysis_path):
        import json
        print(f"Loading existing analysis from {analysis_path}")
        with open(analysis_path) as f:
            return json.load(f)
 
    analysis = analyze_features(
        sae=sae,
        activations=activations,
        tokens=tokens,
        tokenizer_name=cfg["model_name"],
        top_k=cfg["top_k"],
        context_radius=cfg["context_radius"],
        batch_size=cfg["batch_size"],
        save_dir=cfg["results_dir"],
        device=device,
        n_features_to_save=cfg["n_features"],
    )
    print_top_features(analysis, n=20)
    return analysis
 
 
def step_plot(cfg):
    from train import plot_training
    print("\n" + "=" * 60)
    print("STEP 4 — Plotting training curves")
    print("=" * 60)
    metrics_path = os.path.join(cfg["results_dir"], "training_metrics.json")
    if os.path.exists(metrics_path):
        plot_training(
            metrics_path=metrics_path,
            save_path=os.path.join(cfg["results_dir"], "training_curves.png"),
        )
    else:
        print("No training metrics found — skipping plot.")
 
 
 
def main():
    parser = argparse.ArgumentParser(description="SAE training pipeline")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip activation collection (load from cache)")
    parser.add_argument("--skip-train",   action="store_true",
                        help="Skip SAE training (load from checkpoint)")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Skip feature analysis (load from JSON)")
    parser.add_argument("--force",        action="store_true",
                        help="Re-run all steps even if outputs exist")
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Config: {CONFIG}")
 
    # Step 1: Collect activations
    acts, toks = step_collect(CONFIG, force=args.force and not args.skip_collect, device=device)
 
    # Step 2: Train SAE
    sae = step_train(CONFIG, acts, force=args.force and not args.skip_train, device=device)
 
    # Step 3: Analyse features
    if not args.skip_analyze:
        step_analyze(CONFIG, sae, acts, toks, force=args.force, device=device)
 
    # Step 4: Plot training curves
    step_plot(CONFIG)
 
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("Next step: python dashboard/app.py")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()