import os
import json
import torch
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer
 
from sae import SparseAutoencoder
 
 
 
def analyze_features(
    sae:           SparseAutoencoder,
    activations:   torch.Tensor,
    tokens:        torch.Tensor,
    tokenizer_name: str  = "gpt2",
    top_k:         int   = 20,
    context_radius: int  = 15,
    batch_size:    int   = 4096,
    save_dir:      str   = "results",
    device:        Optional[str] = None,
    n_features_to_save: int = 500,   
) -> dict:
    """
    Compute per-feature statistics and top-K activating examples.
 
    Args:
        sae              : trained SparseAutoencoder
        activations      : (N, d_model) — cached GPT-2 activations
        tokens           : (N,)          — corresponding token IDs
        tokenizer_name   : for decoding token IDs to strings
        top_k            : examples to save per feature
        context_radius   : tokens of context on each side of the peak
        batch_size       : activations processed at once
        save_dir         : where to write feature_analysis.json
        device           : compute device
        n_features_to_save: only save the N features that fire most often
 
    Returns:
        analysis dict (also written to disk)
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sae.eval().to(device)
 
    N        = activations.shape[0]
    d_hidden = sae.d_hidden
 
    print(f"\nAnalysing {N:,} tokens × {d_hidden:,} features | top_k={top_k}")
 
    topk_vals = torch.full((d_hidden, top_k), float("-inf"))
    topk_pos  = torch.zeros((d_hidden, top_k), dtype=torch.long)
 
    # Feature statistics
    feature_freq    = torch.zeros(d_hidden)  # number of times each feature fired
    feature_sum_act = torch.zeros(d_hidden)  # sum of activations (for mean)
 
    for start in tqdm(range(0, N, batch_size), desc="Finding top examples", unit="batch"):
        end        = min(start + batch_size, N)
        batch_acts = activations[start:end].to(device)
 
        with torch.no_grad():
            h = sae.encode(batch_acts)   # (B, d_hidden)
 
        h_cpu = h.cpu()
        B     = h_cpu.shape[0]
 
        fired_mask       = h_cpu > 0          # (B, d_hidden)
        feature_freq    += fired_mask.float().sum(dim=0)
        feature_sum_act += h_cpu.sum(dim=0)
        batch_pos = torch.arange(start, end, dtype=torch.long)  # (B,)
        batch_pos = batch_pos.unsqueeze(0).expand(d_hidden, -1)  # (d_hidden, B)
 
        all_vals = torch.cat([topk_vals, h_cpu.T], dim=1)   # (d_hidden, K+B)
        all_pos  = torch.cat([topk_pos,  batch_pos], dim=1)  # (d_hidden, K+B)
 
        topk_vals, keep_idx = torch.topk(all_vals, k=top_k, dim=1, sorted=True)
        topk_pos            = all_pos.gather(1, keep_idx)

    freq_float = feature_freq / N
 
    mean_act = torch.where(
        feature_freq > 0,
        feature_sum_act / feature_freq.clamp(min=1),
        torch.zeros_like(feature_freq),
    )
    max_act = topk_vals[:, 0].clamp(min=0)  # top-1 = global max
 
    min_freq = 0.001   
    max_freq = 0.50   
    interpretable_mask = (freq_float >= min_freq) & (freq_float <= max_freq)
    candidate_features = interpretable_mask.nonzero(as_tuple=True)[0]
 
    if len(candidate_features) > n_features_to_save:
        scores = max_act[candidate_features]
        _, top_idx = scores.topk(n_features_to_save)
        selected_features = candidate_features[top_idx].tolist()
    else:
        selected_features = candidate_features.tolist()
 
    print(f"  Total features:        {d_hidden:,}")
    print(f"  Interpretable range:   {len(candidate_features):,} "
          f"(fire {100*min_freq:.1f}%–{100*max_freq:.0f}% of tokens)")
    print(f"  Saving examples for:   {len(selected_features):,} features")
 
    top_examples = {}
 
    for feat_idx in tqdm(selected_features, desc="Building contexts", unit="feature"):
        examples = []
        for rank in range(top_k):
            act_val = topk_vals[feat_idx, rank].item()
            if act_val <= 0:
                break  
 
            pos = topk_pos[feat_idx, rank].item()
 
            ctx_start = max(0, pos - context_radius)
            ctx_end   = min(N, pos + context_radius + 1)
            ctx_token_ids   = tokens[ctx_start:ctx_end].tolist()
            act_pos_in_ctx  = pos - ctx_start  # index of peak token in context
 
            ctx_strings = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                for tid in ctx_token_ids
            ]
 
            examples.append({
                "activation":  round(act_val, 4),
                "context":     ctx_strings,
                "pos":         int(act_pos_in_ctx),  
            })
 
        top_examples[str(feat_idx)] = examples
 
    feature_stats = {}
    for feat_idx in selected_features:
        feature_stats[str(feat_idx)] = {
            "frequency": round(float(freq_float[feat_idx].item()), 6),
            "mean_act":  round(float(mean_act[feat_idx].item()),   4),
            "max_act":   round(float(max_act[feat_idx].item()),     4),
        }
 
    analysis = {
        "n_tokens":    N,
        "d_hidden":    d_hidden,
        "top_k":       top_k,
        "feature_stats":  feature_stats,
        "top_examples":   top_examples,
    }
    out_path = os.path.join(save_dir, "feature_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f)
 
    print(f"\nSaved feature analysis → {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    return analysis
 
 
 
def print_top_features(analysis: dict, n: int = 20):
    """Print the top N features by max activation with their peak token."""
    stats    = analysis["feature_stats"]
    examples = analysis["top_examples"]
 
    sorted_feats = sorted(
        stats.keys(),
        key=lambda k: stats[k]["max_act"],
        reverse=True,
    )[:n]
 
    print(f"\n{'─'*70}")
    print(f"{'Feature':>8}  {'Freq':>6}  {'MaxAct':>7}  {'Top token'}")
    print(f"{'─'*70}")
    for feat_str in sorted_feats:
        s     = stats[feat_str]
        exs   = examples.get(feat_str, [])
        if exs:
            peak_ctx = exs[0]["context"]
            peak_pos = exs[0]["pos"]
            peak_tok = repr(peak_ctx[peak_pos]) if peak_pos < len(peak_ctx) else "?"
        else:
            peak_tok = "?"
        print(f"{feat_str:>8}  {s['frequency']:>5.2%}  {s['max_act']:>7.2f}  {peak_tok}")
    print(f"{'─'*70}")