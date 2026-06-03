import os
import json
import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
 
 
 
def collect_activations(
    model_name:    str  = "gpt2",
    layer_idx:     int  = 8,
    n_tokens:      int  = 200_000,
    context_len:   int  = 128,
    batch_size:    int  = 32,         
    save_dir:      str  = "data",
    device:        Optional[str] = None,
    overwrite:     bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stream WikiText-2 through GPT-2 and cache residual-stream activations.
 
    Args:
        model_name  : HuggingFace model ID (default 'gpt2')
        layer_idx   : which transformer block to hook (0-indexed)
        n_tokens    : how many token activations to collect
        context_len : tokens per chunk fed to the model
        batch_size  : chunks processed in one forward pass
        save_dir    : directory for output .pt files
        device      : 'cuda', 'cpu', or None (auto-detect)
        overwrite   : if False and files exist, load and return them
 
    Returns:
        activations : (N, d_model) FloatTensor
        tokens      : (N,)          LongTensor
    """
    os.makedirs(save_dir, exist_ok=True)
    act_path   = os.path.join(save_dir, "activations.pt")
    token_path = os.path.join(save_dir, "tokens.pt")
    meta_path  = os.path.join(save_dir, "meta.json")
 
    if not overwrite and os.path.exists(act_path) and os.path.exists(token_path):
        print(f"Loading cached activations from {save_dir}/")
        activations = torch.load(act_path, map_location="cpu")
        tokens      = torch.load(token_path, map_location="cpu")
        print(f"  Loaded: activations {tuple(activations.shape)}, tokens {tuple(tokens.shape)}")
        return activations, tokens
 
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Collecting activations | model={model_name} | layer={layer_idx} | "
          f"n_tokens={n_tokens:,} | device={device}")
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()
    d_model = model.config.hidden_size  # 768 for GPT-2 small
 
    # Captures the output tensor of block `layer_idx` (residual stream)
    _buffer: list[torch.Tensor] = []
 
    def _hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        _buffer.append(hidden.detach().float().cpu())
 
    hook_handle = model.transformer.h[layer_idx].register_forward_hook(_hook)
 
    print("Loading WikiText-2 ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text    = "\n\n".join(t for t in dataset["text"] if t.strip())
 
    all_tokens = tokenizer.encode(text)
    print(f"  WikiText-2 tokenized: {len(all_tokens):,} tokens available")
 
    act_chunks:   list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    total_collected = 0
    n_chunks_needed = (n_tokens + context_len - 1) // context_len
 
    chunks = [
        all_tokens[i : i + context_len]
        for i in range(0, len(all_tokens) - context_len, context_len)
    ]
    while len(chunks) < n_chunks_needed:
        chunks.extend(chunks)
    chunks = chunks[:n_chunks_needed]
 
    for batch_start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Collecting activations",
        unit="batch",
    ):
        batch_chunks  = chunks[batch_start : batch_start + batch_size]
        input_ids     = torch.tensor(batch_chunks, dtype=torch.long, device=device)  # (B, L)
 
        with torch.no_grad():
            model(input_ids)  # triggers the hook
 
        # _buffer[-1]: (B, L, d_model)
        acts_batch = _buffer.pop()          # (B, L, d_model)
        toks_batch = input_ids.cpu()        # (B, L)
 
        # Flatten to (B*L, d_model) and (B*L,)
        B, L, D = acts_batch.shape
        act_chunks.append(acts_batch.reshape(B * L, D))
        token_chunks.append(toks_batch.reshape(B * L))
 
        total_collected += B * L
        if total_collected >= n_tokens:
            break
 
    hook_handle.remove()
 
   
    activations = torch.cat(act_chunks, dim=0)[:n_tokens]   # (N, d_model)
    tokens      = torch.cat(token_chunks, dim=0)[:n_tokens]  # (N,)
 
    print(f"\nCollected {activations.shape[0]:,} activations of dim {activations.shape[1]}")
    print(f"  Mean activation: {activations.mean().item():.4f}")
    print(f"  Std  activation: {activations.std().item():.4f}")
 
    torch.save(activations, act_path)
    torch.save(tokens,      token_path)
    with open(meta_path, "w") as f:
        json.dump({
            "model":       model_name,
            "layer_idx":   layer_idx,
            "n_tokens":    int(activations.shape[0]),
            "d_model":     int(activations.shape[1]),
            "context_len": context_len,
        }, f, indent=2)
 
    print(f"Saved → {act_path}, {token_path}")
    return activations, tokens