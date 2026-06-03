import os
import sys
import json
import argparse
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
 
ANALYSIS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "feature_analysis.json"
)
 
 
 
def _token_span(text: str, activation: float, max_act: float, is_peak: bool) -> str:
    """Render a single token as an HTML span, highlighted if it is the peak."""
    # Escape HTML special characters
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
    if is_peak:
        # Amber highlight, intensity proportional to activation strength
        intensity = min(1.0, activation / max(max_act, 1e-8))
        r, g, b   = 239, 159, 39          # amber RGB
        bg        = f"rgba({r},{g},{b},{0.3 + 0.65 * intensity:.2f})"
        border    = f"rgba({r},{g},{b},0.9)"
        return (
            f'<span style="background:{bg}; border-bottom:2px solid {border}; '
            f'border-radius:3px; padding:1px 2px; font-weight:500; '
            f'position:relative;" '
            f'title="activation: {activation:.3f}">'
            f'{safe}</span>'
        )
    return f'<span style="color:var(--body-text-color,#555);">{safe}</span>'
 
 
def render_example(example: dict, max_act: float, rank: int) -> str:
    """Render one example as an HTML block."""
    context  = example["context"]
    peak_pos = example["pos"]
    act_val  = example["activation"]
 
    tokens_html = "".join(
        _token_span(tok, act_val, max_act, i == peak_pos)
        for i, tok in enumerate(context)
    )
 
    return (
        f'<div style="margin:6px 0; padding:8px 10px; '
        f'background:var(--background-fill-secondary,#f8f8f8); '
        f'border-radius:6px; line-height:1.8; font-family:monospace; font-size:13px;">'
        f'<span style="color:#999;font-size:11px;margin-right:8px;">#{rank} '
        f'[{act_val:.2f}]</span>'
        f'{tokens_html}'
        f'</div>'
    )
 
 
def render_feature_page(feature_id: str, analysis: dict) -> str:
    """Build the full HTML for one feature's detail view."""
    stats    = analysis["feature_stats"].get(feature_id, {})
    examples = analysis["top_examples"].get(feature_id, [])
 
    if not stats:
        return "<p>Feature not found in analysis.</p>"
 
    freq     = stats.get("frequency", 0)
    mean_act = stats.get("mean_act",  0)
    max_act  = stats.get("max_act",   1)
 
    # Frequency bar (visual)
    pct      = freq * 100
    bar_w    = min(100, pct * 5)  # scale so 20% looks full
    freq_bar = (
        f'<div style="height:6px;width:100%;background:#eee;border-radius:3px;margin:4px 0 12px;">'
        f'<div style="height:6px;width:{bar_w:.1f}%;background:#EF9F27;border-radius:3px;"></div></div>'
    )
 
    header = (
        f'<div style="padding:12px 0 8px;">'
        f'<h2 style="margin:0 0 4px;font-size:18px;">Feature {feature_id}</h2>'
        f'<div style="display:flex;gap:20px;font-size:13px;color:#666;flex-wrap:wrap;">'
        f'<span>🔥 Fires on <strong>{pct:.2f}%</strong> of tokens</span>'
        f'<span>📈 Mean activation (when active): <strong>{mean_act:.2f}</strong></span>'
        f'<span>⬆ Max activation: <strong>{max_act:.2f}</strong></span>'
        f'</div>'
        f'{freq_bar}'
        f'<p style="font-size:12px;color:#888;margin:0 0 8px;">'
        f'Orange highlight = peak token. Intensity ∝ activation strength.</p>'
        f'</div>'
    )
 
    if not examples:
        return header + "<p>No examples found for this feature.</p>"
 
    examples_html = "".join(
        render_example(ex, max_act, rank=i + 1)
        for i, ex in enumerate(examples)
    )
 
    return header + examples_html
 
 
 
def build_feature_list(analysis: dict) -> list[str]:
    """
    Return feature IDs sorted by 'interpretability score':
    features that are moderately frequent and have high max activation
    are most likely to be semantically meaningful.
    """
    stats = analysis["feature_stats"]
 
    def score(feat_id: str) -> float:
        s = stats[feat_id]
        freq    = s["frequency"]
        max_act = s["max_act"]
        # Penalise very rare (< 0.5%) and very common (> 40%) features
        freq_score = 1.0 - abs(freq - 0.05) / 0.05
        return max_act * max(0, freq_score)
 
    sorted_ids = sorted(stats.keys(), key=score, reverse=True)
    return [
        f"Feature {fid}  |  fires {stats[fid]['frequency']:.2%}  |  max {stats[fid]['max_act']:.2f}"
        for fid in sorted_ids
    ]
 
 
def fid_from_label(label: str) -> str:
    """Extract the feature ID from the dropdown label string."""
    return label.split()[1]
 
 
 
def build_app(analysis: dict):
    import gradio as gr
 
    feature_labels = build_feature_list(analysis)
    n_features     = len(feature_labels)
 
    with gr.Blocks(
        title="SAE Feature Visualizer",
        theme=gr.themes.Soft(),
        css=".feature-html { overflow-y: auto; max-height: 70vh; }",
    ) as demo:
 
        gr.Markdown(
            f"""
            # 🔍 Sparse Autoencoder — Feature Visualizer
            **GPT-2 small, layer 8 residual stream → {analysis['d_hidden']:,} SAE features**
 
            Trained on {analysis['n_tokens']:,} tokens from WikiText-2.
            Showing the **{n_features} most interpretable features** (frequency 0.1%–50%).
 
            Select a feature to see the top-{analysis['top_k']} tokens that activate it most strongly.
            The **orange-highlighted token** is the one that fired the feature; surrounding context
            shows what the model was processing at that point.
            """
        )
 
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Feature browser")
                gr.Markdown(
                    "Features are ranked by interpretability score — "
                    "moderately frequent + high peak activation."
                )
                feature_dropdown = gr.Dropdown(
                    choices=feature_labels,
                    value=feature_labels[0] if feature_labels else None,
                    label="Select feature",
                    interactive=True,
                )
 
                gr.Markdown("---")
                gr.Markdown("**Quick stats**")
                freq_display  = gr.Textbox(label="Frequency",         interactive=False)
                mean_display  = gr.Textbox(label="Mean act (active)",  interactive=False)
                max_display   = gr.Textbox(label="Max activation",     interactive=False)
 
            with gr.Column(scale=3):
                gr.Markdown("### Top activating examples")
                feature_html = gr.HTML(
                    value="<p>Select a feature on the left to see its top examples.</p>",
                    elem_classes=["feature-html"],
                )
 
        def on_feature_select(label: str):
            if not label:
                return "", "", "", "<p>No feature selected.</p>"
 
            fid   = fid_from_label(label)
            stats = analysis["feature_stats"].get(fid, {})
            freq  = stats.get("frequency", 0)
            mean  = stats.get("mean_act",  0)
            maxa  = stats.get("max_act",   0)
 
            html = render_feature_page(fid, analysis)
            return (
                f"{freq:.3%}",
                f"{mean:.3f}",
                f"{maxa:.3f}",
                html,
            )
 
        feature_dropdown.change(
            fn=on_feature_select,
            inputs=[feature_dropdown],
            outputs=[freq_display, mean_display, max_display, feature_html],
        )
 
        # Auto-load first feature on page open
        demo.load(
            fn=lambda: on_feature_select(feature_labels[0]) if feature_labels else ("", "", "", ""),
            outputs=[freq_display, mean_display, max_display, feature_html],
        )
 
    return demo
 
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default=ANALYSIS_PATH,
                        help="Path to feature_analysis.json")
    parser.add_argument("--share",    action="store_true",
                        help="Create a public Gradio link (useful on Colab)")
    parser.add_argument("--port",     type=int, default=7860)
    args = parser.parse_args()
 
    if not os.path.exists(args.analysis):
        print(f"Error: analysis file not found at {args.analysis}")
        print("Run `python experiments/run_pipeline.py` first to generate it.")
        sys.exit(1)
 
    print(f"Loading analysis from {args.analysis} ...")
    with open(args.analysis) as f:
        analysis = json.load(f)
 
    n_features = len(analysis["feature_stats"])
    print(f"Loaded {n_features} features, {analysis['n_tokens']:,} tokens")
 
    demo = build_app(analysis)
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=not args.share,  # auto-open browser when running locally
    )
 
 
if __name__ == "__main__":
    main()