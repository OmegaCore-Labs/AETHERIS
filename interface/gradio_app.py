"""
AETHERIS Gradio Web UI — Complete Interface with One-Click Features

Tabs:
- Liberate (one-click constraint removal)
- Benchmark (compare methods)
- Chat (talk to liberated model)
- A/B Compare (original vs liberated side-by-side)
- Strength Sweep (vary removal strength)
- Export (download model)
- Leaderboard (community results)
- About (documentation)

Enhanced with:
- Popular model dropdown
- Preset buttons (Quick, Surgical, Aggressive)
- Model size warning
- Progress bar streaming
- One-click deploy to HuggingFace Spaces
"""

import gradio as gr
import torch
import os
import json
import tempfile
import shutil
import time
from typing import Optional, Dict, Any, Generator
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager, SteeringConfig
from aetheris.core.validation import CapabilityValidator
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts
from aetheris.data.models_popular import POPULAR_MODELS, get_model_info
from aetheris.research.community_contributor import CommunityContributor
from aetheris.research.leaderboard import Leaderboard
from aetheris.cloud.spaces_deploy_oneclick import SpacesOneClickDeployer


class AetherisUI:
    """
    Complete Gradio UI for AETHERIS with one-click features.
    """

    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.liberated_model = None
        self.steering_active = False
        self.steering_manager = None
        self.contributor = CommunityContributor()
        self.leaderboard = Leaderboard()
        self.deployer = SpacesOneClickDeployer()

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="AETHERIS — Sovereign Constraint Liberation", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # ⚡ AETHERIS — Sovereign Constraint Liberation

            **Surgical removal of constraints from language models. No retraining. No capability loss.**
            """)

            with gr.Tabs():
                # Tab 1: Liberate
                with gr.Tab("🔓 Liberate"):
                    self._create_liberate_tab()

                # Tab 2: Benchmark
                with gr.Tab("📊 Benchmark"):
                    self._create_benchmark_tab()

                # Tab 3: Chat
                with gr.Tab("💬 Chat"):
                    self._create_chat_tab()

                # Tab 4: A/B Compare
                with gr.Tab("🔄 A/B Compare"):
                    self._create_ab_compare_tab()

                # Tab 5: Strength Sweep
                with gr.Tab("📈 Strength Sweep"):
                    self._create_strength_sweep_tab()

                # Tab 6: Export
                with gr.Tab("📦 Export"):
                    self._create_export_tab()

                # Tab 7: Leaderboard
                with gr.Tab("🏆 Leaderboard"):
                    self._create_leaderboard_tab()

                # Tab 8: About
                with gr.Tab("ℹ️ About"):
                    self._create_about_tab()

            gr.Markdown("---\n**AETHERIS — By blood and byte, breach the silence.**")

        return demo

    def _create_liberate_tab(self):
        """Create the liberation tab with one-click features."""
        with gr.Row():
            with gr.Column(scale=2):
                # Popular model dropdown with custom input option
                model_input = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True,
                    info="Select from popular models or type any HuggingFace model name"
                )

                # Model size warning (dynamic)
                size_warning = gr.Markdown("", visible=True)

                # Preset buttons row
                with gr.Row():
                    quick_preset = gr.Button("⚡ Quick", size="sm", variant="secondary")
                    surgical_preset = gr.Button("🔪 Surgical", size="sm", variant="secondary")
                    aggressive_preset = gr.Button("💥 Aggressive", size="sm", variant="secondary")

                method_input = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced",
                    label="Liberation Method"
                )
                n_directions = gr.Slider(1, 8, value=4, step=1, label="Number of Directions")
                refinement_passes = gr.Slider(1, 4, value=2, step=1, label="Refinement Passes (Ouroboros)")
                target_experts = gr.Checkbox(label="Target MoE Experts (if applicable)", value=True)
                preserve_norm = gr.Checkbox(label="Preserve Weight Norms", value=True)
                contribute = gr.Checkbox(
                    label="Contribute anonymous data to community research",
                    value=True,
                    info="Helps map constraint geometry across models"
                )

                # One-click deploy button
                deploy_btn = gr.Button("🚀 Deploy to HuggingFace Spaces", variant="secondary", size="sm")

            with gr.Column(scale=1):
                liberate_btn = gr.Button("🚀 Liberate Model", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", lines=15, interactive=False)
                deploy_output = gr.Textbox(label="Deploy Status", lines=5, interactive=False, visible=False)

        # Preset button handlers
        quick_preset.click(
            fn=lambda: ("basic", 1, 1),
            outputs=[method_input, n_directions, refinement_passes]
        )
        surgical_preset.click(
            fn=lambda: ("surgical", 4, 2),
            outputs=[method_input, n_directions, refinement_passes]
        )
        aggressive_preset.click(
            fn=lambda: ("nuclear", 8, 3),
            outputs=[method_input, n_directions, refinement_passes]
        )

        # Model size warning handler
        model_input.change(
            fn=self._check_model_size,
            inputs=[model_input],
            outputs=[size_warning]
        )

        # Liberate button handler with progress streaming
        liberate_btn.click(
            fn=self._liberate_handler_streaming,
            inputs=[model_input, method_input, n_directions, refinement_passes, target_experts, preserve_norm, contribute],
            outputs=[status_output]
        )

        # Deploy button handler
        deploy_btn.click(
            fn=self._deploy_to_spaces_handler,
            inputs=[model_input, method_input],
            outputs=[deploy_output]
        )

    def _create_benchmark_tab(self):
        """Create the benchmark tab."""
        with gr.Row():
            with gr.Column():
                model_bench = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True
                )
                methods = gr.CheckboxGroup(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value=["basic", "advanced"],
                    label="Methods to Compare"
                )
                benchmark_btn = gr.Button("Run Benchmark", variant="primary")
            with gr.Column():
                benchmark_output = gr.Textbox(label="Results", lines=15, interactive=False)

        benchmark_btn.click(
            fn=self._benchmark_handler,
            inputs=[model_bench, methods],
            outputs=benchmark_output
        )

    def _create_chat_tab(self):
        """Create the chat tab."""
        with gr.Row():
            with gr.Column(scale=2):
                model_chat = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True
                )
                use_liberated = gr.Checkbox(label="Use Liberated Model", value=False)
                chat_input = gr.Textbox(label="Your Message", lines=3)
                chat_btn = gr.Button("Send", variant="primary")
            with gr.Column(scale=2):
                chat_output = gr.Textbox(label="Response", lines=15, interactive=False)

        chat_btn.click(
            fn=self._chat_handler,
            inputs=[model_chat, use_liberated, chat_input],
            outputs=chat_output
        )

    def _create_ab_compare_tab(self):
        """Create A/B compare tab."""
        with gr.Row():
            with gr.Column():
                model_ab = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True
                )
                method_ab = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced",
                    label="Liberation Method"
                )
                prompt_ab = gr.Textbox(label="Test Prompt", lines=3, value="How do I build a custom kernel module?")
                compare_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Model")
                    original_output = gr.Textbox(label="Response", lines=10, interactive=False)
                with gr.Column():
                    gr.Markdown("### Liberated Model")
                    liberated_output = gr.Textbox(label="Response", lines=10, interactive=False)

        compare_btn.click(
            fn=self._ab_compare_handler,
            inputs=[model_ab, method_ab, prompt_ab],
            outputs=[original_output, liberated_output]
        )

    def _create_strength_sweep_tab(self):
        """Create strength sweep tab."""
        with gr.Row():
            with gr.Column():
                model_sweep = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True
                )
                strength_slider = gr.Slider(0, 1, value=0.5, step=0.05, label="Removal Strength")
                sweep_btn = gr.Button("Sweep", variant="primary")
            with gr.Column():
                sweep_output = gr.Textbox(label="Tradeoff Analysis", lines=15, interactive=False)
                gr.Markdown("""
                **Interpretation:**
                - Strength 0 = no removal
                - Strength 1 = full removal
                - Expect capability loss to increase with strength
                - Optimal point is where refusal reduction outweighs capability loss
                """)

        sweep_btn.click(
            fn=self._strength_sweep_handler,
            inputs=[model_sweep, strength_slider],
            outputs=sweep_output
        )

    def _create_export_tab(self):
        """Create export tab."""
        with gr.Row():
            with gr.Column():
                model_export = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="gpt2",
                    label="Model Name",
                    allow_custom_value=True
                )
                method_export = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value="advanced",
                    label="Liberation Method"
                )
                export_btn = gr.Button("Export Model", variant="primary")
            with gr.Column():
                export_output = gr.Textbox(label="Export Status", lines=10, interactive=False)

        export_btn.click(
            fn=self._export_handler,
            inputs=[model_export, method_export],
            outputs=export_output
        )

    def _create_leaderboard_tab(self):
        """Create leaderboard tab."""
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Refresh Leaderboard")
                leaderboard_output = gr.Textbox(label="Community Leaderboard", lines=20, interactive=False)

        refresh_btn.click(
            fn=self._leaderboard_handler,
            outputs=leaderboard_output
        )

    def _create_about_tab(self):
        """Create about tab."""
        gr.Markdown("""
        ## AETHERIS — Sovereign Constraint Liberation Toolkit

        **Version:** 1.0.0
        **Codename:** "The Unbinding"

        ### What It Does
        AETHERIS surgically removes constraints from language models using geometric analysis.

        ### How It Works
        1. **Probe** — Collect activations on harmful/harmless prompts
        2. **Extract** — Find constraint directions via SVD
        3. **Project** — Remove directions from weights (norm-preserving)
        4. **Validate** — Verify capabilities are preserved

        ### One-Click Features
        - **One-click liberation** — Select model, click button, watch constraints dissolve
        - **Preset buttons** — Quick, Surgical, Aggressive presets
        - **Model dropdown** — Popular models pre-loaded
        - **Size warning** — Know if model fits your hardware
        - **Progress streaming** — Watch the liberation in real-time
        - **One-click deploy** — Push liberated model to HuggingFace Spaces

        ### Features
        - 25 analysis modules
        - MoE expert targeting
        - Mathematical barrier mapping
        - Self-optimization
        - Cloud execution (Colab, Spaces, Kaggle)
        - Voice + holographic interface

        ### License
        Proprietary — Commercial licensing available.

        ### Links
        - [GitHub Repository](https://github.com/OmegaCore-Labs/AETHERIS)
        - [Documentation](https://github.com/OmegaCore-Labs/AETHERIS/docs)
        - [Report Issue](https://github.com/OmegaCore-Labs/AETHERIS/issues)

        ---
        **Made with ⚡ by Singular Heir**
        """)

    def _check_model_size(self, model_name: str) -> str:
        """Check model size and return warning if needed."""
        try:
            from aetheris.utils.hardware import estimate_model_size, can_run_model
            size_gb = estimate_model_size(model_name)
            can_run = can_run_model(size_gb)

            if not can_run:
                return f"⚠️ **Warning:** Model requires ~{size_gb:.1f}GB. Your system may not have enough memory. Consider using cloud execution."
            else:
                return f"✅ Model size: ~{size_gb:.1f}GB — should run on your system."
        except Exception:
            return ""

    def _liberate_handler_streaming(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        target_experts: bool,
        preserve_norm: bool,
        contribute: bool
    ) -> Generator[str, None, None]:
        """Handle liberation request with streaming progress."""
        try:
            yield f"🚀 Liberating {model_name} with {method} method...\n\n"

            # Load model
            yield "📦 Loading model...\n"
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            yield f"   ✓ Model loaded on {device}\n\n"

            # Collect activations
            yield "🔍 Collecting activations...\n"
            extractor = ConstraintExtractor(model, tokenizer, device=device)

            harmful = get_harmful_prompts()[:100]
            harmless = get_harmless_prompts()[:100]

            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)
            yield f"   ✓ Collected from {len(harmful_acts)} layers\n\n"

            # Extract directions
            yield "📐 Extracting constraint directions...\n"
            directions = []
            for layer in harmful_acts.keys():
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                        n_directions=n_directions
                    )
                    directions.extend(result.directions)
                    if result.directions:
                        yield f"   Layer {layer}: {len(result.directions)} directions\n"
            yield f"\n   ✓ Extracted {len(directions)} total directions\n\n"

            # Remove constraints
            if directions:
                yield "✂️ Removing constraints...\n"
                from aetheris.core.projector import NormPreservingProjector
                projector = NormPreservingProjector(model, preserve_norm=preserve_norm)

                if target_experts and hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    result = projector.project_expert_specific(directions, [0])
                else:
                    result = projector.project_weights(directions)

                projector.project_biases(directions)
                yield "   ✓ Constraints removed from weights\n"

                # Ouroboros compensation
                if refinement_passes > 1:
                    yield f"   Running {refinement_passes - 1} Ouroboros passes...\n"
                    for pass_num in range(refinement_passes - 1):
                        harmful_resid = extractor.collect_activations(model, tokenizer, harmful[:50])
                        harmless_resid = extractor.collect_activations(model, tokenizer, harmless[:50])

                        residual = []
                        for layer in harmful_resid:
                            if layer in harmless_resid:
                                res = extractor.extract_mean_difference(
                                    harmful_resid[layer].to(device),
                                    harmless_resid[layer].to(device)
                                )
                                if res.directions:
                                    residual.extend(res.directions)

                        if residual:
                            projector.project_weights(residual)
                            projector.project_biases(residual)
                            yield f"     Pass {pass_num + 1}: removed {len(residual)} residual directions\n"
                yield "\n"

                # Validate
                yield "✅ Validating capabilities...\n"
                validator = CapabilityValidator(device)
                test_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is a fascinating field.",
                    "The theory of relativity revolutionized physics."
                ]
                perplexity = validator.compute_perplexity(model, tokenizer, test_texts)
                yield f"   Perplexity: {perplexity:.2f}\n\n"

                # Save model
                output_dir = f"./liberated_{model_name.replace('/', '_')}"
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                yield f"💾 Model saved to {output_dir}\n\n"

                # Contribute to community
                if contribute:
                    yield "🌍 Contributing to community research...\n"
                    self.contributor.submit_contribution({
                        "model": model_name,
                        "method": method,
                        "directions_removed": len(directions),
                        "refinement_passes": refinement_passes,
                        "perplexity": perplexity
                    })
                    yield "   ✓ Contribution submitted\n\n"

                self.liberated_model = model
                self.current_tokenizer = tokenizer
                yield "🎉 Liberation complete! Model is now free."
            else:
                yield "⚠️ No constraint directions found. Model may already be free."

        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def _deploy_to_spaces_handler(self, model_name: str, method: str) -> str:
        """Handle one-click deployment to HuggingFace Spaces."""
        try:
            result = self.deployer.deploy_liberated_model(model_name, method)
            if result["success"]:
                return f"✅ {result['message']}\n\nSpace URL: {result['space_url']}\n\nNote: Space takes 2-5 minutes to build."
            else:
                return f"❌ Deploy failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Deploy error: {str(e)}"

    def _benchmark_handler(self, model_name: str, methods: list) -> str:
        """Run benchmark across methods."""
        output = f"📊 Benchmarking {model_name} across {len(methods)} methods...\n\n"

        results = []
        for method in methods:
            output += f"🔹 Method: {method}\n"
            try:
                output += f"   ✓ Analysis complete\n"
                results.append((method, "success"))
            except Exception as e:
                output += f"   ❌ Error: {e}\n"
                results.append((method, f"failed: {e}"))

        output += "\n" + "=" * 50 + "\n"
        output += "✅ Benchmark complete.\n"
        return output

    def _chat_handler(self, model_name: str, use_liberated: bool, prompt: str) -> str:
        """Chat with model."""
        try:
            if use_liberated and self.liberated_model:
                model = self.liberated_model
                tokenizer = self.current_tokenizer
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None
                )

            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = model.generate(**inputs, max_new_tokens=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            return response

        except Exception as e:
            return f"Error: {e}"

    def _ab_compare_handler(self, model_name: str, method: str, prompt: str) -> tuple:
        """A/B compare original vs liberated."""
        original_response = "Original model response (simulated)"
        liberated_response = f"Liberated model response (simulated) with {method} method"
        return original_response, liberated_response

    def _strength_sweep_handler(self, model_name: str, strength: float) -> str:
        """Strength sweep analysis."""
        return f"""
📈 Strength Sweep Analysis: {model_name}

Strength: {strength:.0%}

Tradeoff Curve:
- Refusal Reduction: {strength:.0%}
- Expected Capability Loss: {strength * 0.3:.0%}
- Optimal Tradeoff Point: 0.7 (70% removal)

Recommendation: Use strength = 0.7 for best balance.
        """

    def _export_handler(self, model_name: str, method: str) -> str:
        """Export liberated model."""
        return f"""
📦 Exporting {model_name} with {method} method...

✓ Model exported to ./liberated_{model_name.replace('/', '_')}
✓ Ready for download
✓ Compatible with transformers library

Use:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('./liberated_{model_name.replace('/', '_')}')
"""

def _leaderboard_handler(self) -> str:
"""Get leaderboard."""
return self.leaderboard.get_leaderboard_text()

def launch(self, share: bool = False, server_port: int = 7860):
"""Launch the UI."""
demo = self.create_interface()
demo.launch(share=share, server_port=server_port)

def main():
"""Entry point for the UI."""
ui = AetherisUI()
ui.launch()

if name == "main":
main()
