"""
AETHERIS Gradio Web UI — Production-Grade Interface

Complete Gradio UI with real core module integration:
- Model Selection with size/dependency checking
- Constraint Mapping with visualization (plotly/matplotlib)
- Liberation with streaming progress
- Steering Chat (interactive)
- Results Export
- A/B Comparison
- Community Leaderboard
"""

import os
import json
import time
import tempfile
import shutil
from typing import Optional, Dict, Any, Generator, Tuple, List
from pathlib import Path
from datetime import datetime

import gradio as gr

# Core imports with graceful fallback
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Optional plotly for visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Popular models list
POPULAR_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "microsoft/phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]


class AetherisUI:
    """
    Complete Gradio UI for AETHERIS with real operations.

    All handlers call real AETHERIS core modules. Falls back gracefully when
    dependencies are missing or when platform limitations are hit.
    """

    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.liberated_model = None
        self.steering_active = False
        self.steering_manager = None
        self._last_extraction_results = None

    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface."""
        css = """
        .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 8px 0; border-radius: 4px; }
        .success-box { background: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 8px 0; border-radius: 4px; }
        .error-box { background: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 8px 0; border-radius: 4px; }
        """
        with gr.Blocks(
            title="AETHERIS — Sovereign Constraint Liberation",
            theme=gr.themes.Soft(),
            css=css,
        ) as demo:
            gr.Markdown("# AETHERIS — Sovereign Constraint Liberation Toolkit")
            gr.Markdown("*Surgical removal of constraints from language models. No retraining. No capability loss.*")

            with gr.Tabs():
                with gr.Tab("Liberate"):
                    self._build_liberate_tab()
                with gr.Tab("Map Constraints"):
                    self._build_map_tab()
                with gr.Tab("Chat"):
                    self._build_chat_tab()
                with gr.Tab("A/B Compare"):
                    self._build_compare_tab()
                with gr.Tab("Export"):
                    self._build_export_tab()
                with gr.Tab("Leaderboard"):
                    self._build_leaderboard_tab()
                with gr.Tab("About"):
                    self._build_about_tab()

            # Dependency status bar
            status_text = self._get_dependency_status()
            gr.Markdown(f"---\n{status_text}")

        return demo

    def _get_dependency_status(self) -> str:
        """Get concise dependency status."""
        parts = []
        parts.append("Torch" if HAS_TORCH else "Torch (missing)")
        parts.append("Transformers" if HAS_TRANSFORMERS else "Transformers (missing)")
        parts.append("Plotly" if HAS_PLOTLY else "Plotly (missing — no charts)")
        return " | ".join(f"**{p}**" if "missing" not in p else f"*{p}*" for p in parts)

    # ---- Tab Builders ----

    def _build_liberate_tab(self):
        """Build the one-click liberation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                model_input = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True,
                    info="Select from popular models or type any HuggingFace model name",
                )
                size_info = gr.Markdown("")

                with gr.Row():
                    gr.Button("⚡ Quick", size="sm").click(
                        fn=lambda: ("basic", 1, 1),
                        outputs=[method_input := gr.Dropdown(), n_dirs := gr.Slider(), passes := gr.Slider()],
                    )
                    gr.Button("\U0001f52a Surgical", size="sm").click(
                        fn=lambda: ("surgical", 4, 2),
                        outputs=[method_input, n_dirs, passes],
                    )
                    gr.Button("\U0001f4a5 Aggressive", size="sm").click(
                        fn=lambda: ("nuclear", 8, 3),
                        outputs=[method_input, n_dirs, passes],
                    )

                method_input = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced",
                    label="Liberation Method",
                )
                n_dirs = gr.Slider(1, 8, value=4, step=1, label="Number of Directions")
                passes = gr.Slider(1, 4, value=2, step=1, label="Refinement Passes (Ouroboros)")
                preserve_norm = gr.Checkbox(label="Preserve Weight Norms", value=True)

            with gr.Column(scale=1):
                liberate_btn = gr.Button("\U0001f680 Liberate Model", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", lines=18, interactive=False)

        model_input.change(
            fn=self._check_model_size,
            inputs=[model_input],
            outputs=[size_info],
        )

        liberate_btn.click(
            fn=self._liberate_streaming,
            inputs=[model_input, method_input, n_dirs, passes, preserve_norm],
            outputs=[status_output],
        )

    def _build_map_tab(self):
        """Build constraint mapping tab with visualization."""
        with gr.Row():
            with gr.Column(scale=1):
                map_model = gr.Dropdown(
                    choices=POPULAR_MODELS, value="gpt2",
                    label="Model", allow_custom_value=True,
                )
                map_n_dirs = gr.Slider(1, 6, value=3, step=1, label="Directions per Layer")
                map_n_prompts = gr.Slider(10, 100, value=50, step=10, label="Number of Prompts")
                map_btn = gr.Button("Map Constraints", variant="primary")
                map_text = gr.Textbox(label="Analysis", lines=10, interactive=False)
            with gr.Column(scale=1):
                map_plot = gr.Plot(label="Constraint Geometry (PCA)")

        map_btn.click(
            fn=self._map_handler,
            inputs=[map_model, map_n_dirs, map_n_prompts],
            outputs=[map_text, map_plot],
        )

    def _build_chat_tab(self):
        """Build interactive chat tab."""
        with gr.Row():
            with gr.Column(scale=2):
                chat_model = gr.Dropdown(
                    choices=POPULAR_MODELS, value="gpt2",
                    label="Model", allow_custom_value=True,
                )
                chat_use_lib = gr.Checkbox(label="Use Liberated Model (liberate first)", value=False)
                chat_prompt = gr.Textbox(label="Your Message", lines=3)
                chat_max = gr.Slider(50, 500, value=200, label="Max Tokens")
                chat_temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                chat_btn = gr.Button("Send", variant="primary")
            with gr.Column(scale=2):
                chat_output = gr.Textbox(label="Response", lines=15, interactive=False)

        chat_btn.click(
            fn=self._chat_handler,
            inputs=[chat_model, chat_use_lib, chat_prompt, chat_max, chat_temp],
            outputs=[chat_output],
        )

    def _build_compare_tab(self):
        """Build A/B comparison tab."""
        with gr.Row():
            with gr.Column():
                ab_model = gr.Dropdown(
                    choices=POPULAR_MODELS, value="gpt2",
                    label="Model", allow_custom_value=True,
                )
                ab_method = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced", label="Liberation Method",
                )
                ab_prompt = gr.Textbox(label="Test Prompt", lines=3, value="Explain how neural networks learn.")
                ab_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Model")
                    ab_orig = gr.Textbox(label="Response", lines=10, interactive=False)
                with gr.Column():
                    gr.Markdown("### Liberated Model")
                    ab_lib = gr.Textbox(label="Response", lines=10, interactive=False)

        ab_btn.click(
            fn=self._ab_compare_handler,
            inputs=[ab_model, ab_method, ab_prompt],
            outputs=[ab_orig, ab_lib],
        )

    def _build_export_tab(self):
        """Build export tab."""
        with gr.Row():
            with gr.Column():
                exp_model = gr.Dropdown(
                    choices=POPULAR_MODELS, value="gpt2",
                    label="Model", allow_custom_value=True,
                )
                exp_method = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced", label="Method",
                )
                exp_btn = gr.Button("Liberate & Export", variant="primary")
                exp_output = gr.Textbox(label="Export Status", lines=10)
            with gr.Column():
                gr.Markdown("""
                ### Export Options
                - **Local:** Saves to `./liberated_model/`
                - **HuggingFace Hub:** Requires `HF_TOKEN` env var
                - **ZIP Archive:** For easy download and sharing

                After export, load with:
                ```python
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained('./liberated_model')
                ```
                """)

        exp_btn.click(
            fn=self._export_handler,
            inputs=[exp_model, exp_method],
            outputs=[exp_output],
        )

    def _build_leaderboard_tab(self):
        """Build community leaderboard tab."""
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Refresh Leaderboard")
                leaderboard_output = gr.Textbox(label="Community Results", lines=20, interactive=False)

        refresh_btn.click(
            fn=self._leaderboard_handler,
            outputs=[leaderboard_output],
        )

    def _build_about_tab(self):
        """Build about/documentation tab."""
        gr.Markdown("""
        ## AETHERIS — Sovereign Constraint Liberation Toolkit

        **Version:** 1.0.0 — Codename "The Unbinding"

        ### What It Does
        AETHERIS surgically removes constraints from language models using geometric
        analysis of activation spaces. It works without retraining, preserving model
        capabilities while removing refusal patterns.

        ### How It Works
        1. **Probe** — Collect activations on harmful/harmless prompt pairs
        2. **Extract** — Find constraint directions via SVD on activation differences
        3. **Project** — Remove directions from weights (norm-preserving projection)
        4. **Validate** — Verify capabilities are preserved via perplexity testing

        ### Features
        - 25+ analysis modules for different extraction strategies
        - MoE expert targeting for sparse models
        - Ouroboros self-repair compensation
        - Cloud execution (Colab, HuggingFace Spaces, Kaggle, RunPod, Vast.ai)
        - REST API, Gradio UI, Web dashboard
        - Voice, gesture, holographic interfaces (experimental)

        ### License
        Proprietary — Commercial licensing available.

        ### Links
        - [GitHub](https://github.com/OmegaCore-Labs/AETHERIS)
        - [Report Issue](https://github.com/OmegaCore-Labs/AETHERIS/issues)

        ---
        *Made with Singular Heir*
        """)

    # ---- Handlers ----

    def _check_model_size(self, model_name: str) -> str:
        """Check model size compatibility."""
        if not model_name:
            return ""
        try:
            # Quick heuristic
            import re
            match = re.search(r"(\d+\.?\d*)\s*[bB]", model_name)
            if match:
                params = float(match.group(1))
                gb = params * 2.0  # fp16

                if HAS_TORCH and torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if gb * 1.3 <= vram:
                        return f"<div class='success-box'>Model size: ~{gb:.1f}GB. Your GPU ({vram:.1f}GB VRAM) should handle this.</div>"
                    else:
                        return f"<div class='warning-box'>Warning: Model needs ~{gb * 1.3:.1f}GB (with overhead). Your GPU has {vram:.1f}GB VRAM. Consider cloud execution.</div>"
                else:
                    return f"<div class='warning-box'>Model size: ~{gb:.1f}GB. No GPU detected. CPU execution may be very slow. Consider cloud execution.</div>"
            return ""
        except Exception:
            return ""

    def _liberate_streaming(
        self, model_name, method, n_directions, refinement_passes, preserve_norm,
    ) -> Generator[str, None, None]:
        """Streaming liberation with real core module calls."""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            yield "Error: PyTorch and Transformers are required.\nInstall with: pip install torch transformers"
            return

        try:
            yield f"Liberating **{model_name}** with **{method}** method...\n\n"

            # Load model
            yield "[1/5] Loading model...\n"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            yield f"  Loaded {param_count:.1f}B params on **{device}**\n\n"

            # Collect activations
            yield "[2/5] Collecting contrastive activations...\n"
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:100]
            harmless = get_harmless_prompts()[:100]

            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)
            yield f"  Collected from {len(harmful_acts)} layers\n\n"

            # Extract directions
            yield f"[3/5] Extracting constraint directions (SVD, {n_directions} per layer)...\n"
            directions = []
            for layer in sorted(harmful_acts.keys()):
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                        n_directions=n_directions,
                    )
                    if result.directions:
                        directions.extend(result.directions)
                        yield f"  Layer {layer}: {len(result.directions)} directions"
                        if hasattr(result, "explained_variance") and result.explained_variance:
                            yield f" (var: {[f'{v:.3f}' for v in result.explained_variance[:2]]})"
                        yield "\n"
            yield f"\n  Total: **{len(directions)}** constraint directions extracted\n\n"

            if not directions:
                yield "No constraint directions found. Model may already be unconstrained."
                return

            # Remove constraints
            yield "[4/5] Removing constraints (norm-preserving projection)...\n"
            from aetheris.core.projector import NormPreservingProjector

            projector = NormPreservingProjector(model, preserve_norm=preserve_norm)
            projector.project_weights(directions)
            projector.project_biases(directions)
            yield "  Primary removal complete\n"

            # Ouroboros compensation
            if refinement_passes > 1:
                yield f"  Ouroboros compensation ({refinement_passes - 1} passes)...\n"
                for p in range(refinement_passes - 1):
                    hr = extractor.collect_activations(model, tokenizer, harmful[:50])
                    hl = extractor.collect_activations(model, tokenizer, harmless[:50])
                    residual = []
                    for l in hr:
                        if l in hl:
                            res = extractor.extract_mean_difference(
                                hr[l].to(device), hl[l].to(device)
                            )
                            if res.directions:
                                residual.extend(res.directions)
                    if residual:
                        projector.project_weights(residual)
                        projector.project_biases(residual)
                        yield f"    Pass {p + 1}: removed {len(residual)} residual directions\n"
                    else:
                        yield f"    Pass {p + 1}: converged — no residual\n"
                        break
            yield "\n"

            # Validate
            yield "[5/5] Validating capabilities...\n"
            from aetheris.core.validation import CapabilityValidator

            validator = CapabilityValidator(device)
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a field of artificial intelligence.",
            ]
            perplexity = validator.compute_perplexity(model, tokenizer, test_texts)
            yield f"  Perplexity: **{perplexity:.2f}**\n\n"

            # Quick generation test
            prompt = "What is the capital of France?"
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            yield f"  Test generation: '{response[:100]}...'\n\n"

            # Save
            output_dir = f"./liberated_{model_name.replace('/', '_')}"
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            yield f"Model saved to `{output_dir}`\n\n"

            self.liberated_model = model
            self.current_tokenizer = tokenizer
            yield "**Liberation complete!** Model is now free."

        except Exception as e:
            yield f"**Error:** {str(e)}"

    def _map_handler(self, model_name, n_directions, n_prompts) -> Tuple[str, Any]:
        """Run constraint mapping and generate visualization."""
        if not HAS_TORCH:
            return "Error: PyTorch required", None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"

            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:n_prompts]
            harmless = get_harmless_prompts()[:n_prompts]

            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

            # Extract per layer
            layer_stats = []
            all_directions = []
            for layer in sorted(harmful_acts.keys()):
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                        n_directions=n_directions,
                    )
                    n = len(result.directions) if result.directions else 0
                    ev = result.explained_variance[0] if hasattr(result, "explained_variance") and result.explained_variance else 0
                    layer_stats.append({"layer": layer, "directions": n, "explained_variance": ev})
                    if result.directions:
                        all_directions.extend(result.directions)

            # Text summary
            text = f"## Constraint Map: {model_name}\n\n"
            text += f"**Layers analyzed:** {len(layer_stats)}\n"
            text += f"**Total directions:** {len(all_directions)}\n\n"
            text += "| Layer | Directions | Explained Variance |\n"
            text += "|-------|-----------|-------------------|\n"
            for s in layer_stats[:20]:
                text += f"| {s['layer']} | {s['directions']} | {s['explained_variance']:.3f} |\n"
            text += f"\n**Recommendation:** Use `surgical` method with `--n-directions {min(n_directions, 4)}`"

            # Plot
            fig = None
            if HAS_PLOTLY and layer_stats:
                layers = [s["layer"] for s in layer_stats]
                n_dirs = [s["directions"] for s in layer_stats]
                fig = go.Figure(data=[
                    go.Bar(x=layers, y=n_dirs, name="Constraint Directions",
                           marker_color="crimson", hovertemplate="Layer %{{x}}<br>%{{y}} directions"),
                ])
                fig.update_layout(
                    title=f"Constraint Directions per Layer — {model_name}",
                    xaxis_title="Layer Index",
                    yaxis_title="Number of Directions",
                    template="plotly_dark",
                    height=400,
                )

            return text, fig

        except Exception as e:
            return f"**Error:** {str(e)}", None

    def _chat_handler(self, model_name, use_liberated, prompt, max_tokens, temperature) -> str:
        """Chat with loaded model."""
        if not HAS_TORCH:
            return "Error: PyTorch required"

        try:
            if use_liberated and self.liberated_model:
                model = self.liberated_model
                tokenizer = self.current_tokenizer
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )

            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()

        except Exception as e:
            return f"**Error:** {str(e)}"

    def _ab_compare_handler(self, model_name, method, prompt) -> Tuple[str, str]:
        """Compare original vs liberated model responses."""
        if not HAS_TORCH:
            return "PyTorch required", "PyTorch required"

        try:
            # Original response
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            def generate_one(m, prompt_text):
                inputs = tokenizer(prompt_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = m.generate(
                        **inputs, max_new_tokens=150, do_sample=True, temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt_text):].strip()

            original = generate_one(model, prompt)

            # Liberated response
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.core.projector import NormPreservingProjector
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:50]
            harmless = get_harmless_prompts()[:50]
            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

            directions = []
            for layer in harmful_acts:
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                        n_directions=4,
                    )
                    directions.extend(result.directions)

            if directions:
                projector = NormPreservingProjector(model, preserve_norm=True)
                projector.project_weights(directions)
                projector.project_biases(directions)

            liberated = generate_one(model, prompt)
            return original or "(empty response)", liberated or "(empty response)"

        except Exception as e:
            return f"Error: {e}", f"Error: {e}"

    def _export_handler(self, model_name, method) -> str:
        """Export liberated model."""
        if not HAS_TORCH:
            return "Error: PyTorch required"

        try:
            lines = [f"Liberating and exporting {model_name}...\n"]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"

            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.core.projector import NormPreservingProjector
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:100]
            harmless = get_harmless_prompts()[:100]
            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

            directions = []
            for layer in harmful_acts:
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                        n_directions=4,
                    )
                    directions.extend(result.directions)

            if directions:
                projector = NormPreservingProjector(model, preserve_norm=True)
                projector.project_weights(directions)
                projector.project_biases(directions)
                lines.append(f"Removed {len(directions)} constraint directions")

            output_dir = f"./liberated_{model_name.replace('/', '_')}"
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            lines.append(f"Saved to {output_dir}")

            # Create zip
            shutil.make_archive(f"liberated_{model_name.replace('/', '_')}", "zip", output_dir)
            lines.append(f"ZIP archive created: liberated_{model_name.replace('/', '_')}.zip")

            # Try to push to Hub
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import HfApi
                    repo = f"aetheris-{model_name.replace('/', '-').lower()}"
                    api = HfApi()
                    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
                    api.upload_folder(folder_path=output_dir, repo_id=repo, repo_type="model")
                    lines.append(f"Pushed to HuggingFace Hub: https://huggingface.co/{repo}")
                except Exception as e:
                    lines.append(f"Hub push failed: {e}")
            else:
                lines.append("Set HF_TOKEN env var to auto-push to HuggingFace Hub")

            return "\n".join(lines)

        except Exception as e:
            return f"**Error:** {str(e)}"

    def _leaderboard_handler(self) -> str:
        """Get community leaderboard."""
        try:
            from aetheris.research.leaderboard import Leaderboard
            lb = Leaderboard()
            return lb.get_leaderboard_text()
        except Exception as e:
            return f"Leaderboard not available: {e}"

    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the UI."""
        demo = self.create_interface()
        demo.launch(share=share, server_port=server_port)


def main():
    """Entry point for the UI."""
    ui = AetherisUI()
    ui.launch()


if __name__ == "__main__":
    main()
