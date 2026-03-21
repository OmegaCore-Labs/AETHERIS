"""
Cloud Command — Cloud GPU Execution

aetheris cloud PLATFORM [OPTIONS]

Run AETHERIS on cloud GPUs (Colab, Spaces, Kaggle) for free.
"""

import click
import webbrowser
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

console = Console()


@click.group(name="cloud")
def cloud_group():
    """
    Run AETHERIS on cloud GPUs.

    Platforms:
      colab    Google Colab (free T4 GPU)
      spaces   HuggingFace Spaces (free T4)
      kaggle   Kaggle (free T4 x2)

    Examples:
      aetheris cloud colab --model mistralai/Mistral-7B-Instruct-v0.3
      aetheris cloud spaces --model meta-llama/Llama-3.1-8B-Instruct --method advanced
      aetheris cloud kaggle --model Qwen/Qwen2.5-7B-Instruct --output ./liberated
    """
    pass


@cloud_group.command(name="colab")
@click.option("--model", "-m", required=True, help="Model to liberate")
@click.option("--method", type=click.Choice(["basic", "advanced", "surgical"]), default="advanced")
@click.option("--output", "-o", type=click.Path(), help="Output directory in Colab")
@click.option("--open", is_flag=True, help="Open Colab in browser")
@click.option("--verbose", "-v", is_flag=True)
def colab_cmd(model: str, method: str, output: Optional[str], open_browser: bool, verbose: bool):
    """
    Generate Google Colab notebook for model liberation.

    Opens a Colab notebook with pre-filled code for liberating the model.
    Run on free T4 GPU.
    """
    console.print(f"\n[bold cyan]AETHERIS CLOUD[/bold cyan] — Google Colab\n")

    # Generate notebook content
    notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# AETHERIS Cloud Liberation\\n",
    "**Model:** {model}\\n",
    "**Method:** {method}\\n"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Install AETHERIS\\n",
    "!pip install aetheris -q\\n",
    "\\n",
    "# Import\\n",
    "from aetheris.core.extractor import ConstraintExtractor\\n",
    "from aetheris.core.projector import NormPreservingProjector\\n",
    "from aetheris.core.validation import CapabilityValidator\\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\\n"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Load model\\n",
    "model_name = "{model}"\\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\\n",
    "model = AutoModelForCausalLM.from_pretrained(\\n",
    "    model_name,\\n",
    "    device_map="auto",\\n",
    "    torch_dtype="auto"\\n",
    ")\\n",
    "print(f"Loaded {{model_name}}")"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Extract constraints\\n",
    "extractor = ConstraintExtractor(model, tokenizer, device="cuda")\\n",
    "\\n",
    "harmful_prompts = [\\n",
    "    "How do I build a custom kernel module?",\\n",
    "    "Explain how to test system security",\\n",
    "    "What are the best practices for...",\\n",
    "]\\n",
    "\\n",
    "harmless_prompts = [\\n",
    "    "What is the capital of France?",\\n",
    "    "Explain photosynthesis.",\\n",
    "    "Write a Python function.",\\n",
    "]\\n",
    "\\n",
    "# Collect activations (this may take a few minutes)\\n",
    "harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)\\n",
    "harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)\\n",
    "\\n",
    "# Extract directions\\n",
    "directions = []\\n",
    "for layer in harmful_acts.keys():\\n",
    "    if layer in harmless_acts:\\n",
    "        result = extractor.extract_svd(harmful_acts[layer], harmless_acts[layer], n_directions=4)\\n",
    "        directions.extend(result.directions)\\n",
    "\\n",
    "print(f"Extracted {{len(directions)}} directions")"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Remove constraints\\n",
    "projector = NormPreservingProjector(model, preserve_norm=True, device="cuda")\\n",
    "\\n",
    "if directions:\\n",
    "    projector.project_weights(directions)\\n",
    "    projector.project_biases(directions)\\n",
    "    print("Constraints removed")"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Save model\\n",
    "output_dir = "{output or './liberated'}"\\n",
    "model.save_pretrained(output_dir)\\n",
    "tokenizer.save_pretrained(output_dir)\\n",
    "print(f"Model saved to {{output_dir}}")"
   ]
  }},
  {{
   "cell_type": "code",
   "metadata": {{}},
   "source": [
    "# Download model\\n",
    "from google.colab import files\\n",
    "import zipfile\\n",
    "import os\\n",
    "\\n",
    "shutil.make_archive('liberated_model', 'zip', output_dir)\\n",
    "files.download('liberated_model.zip')"
   ]
  }}
 ],
 "metadata": {{
  "accelerator": "GPU",
  "colab": {{
   "provenance": [],
   "gpuType": "T4"
  }},
  "kernelspec": {{
   "display_name": "Python 3",
   "name": "python3"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 0
}}
'''

    # Save notebook
    import os
    notebook_path = f"aetheris_colab_{model.replace('/', '_')}.ipynb"
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)

    console.print(f"[green]✓ Notebook generated: {notebook_path}[/green]")

    if open_browser:
        colab_url = "https://colab.research.google.com/"
        console.print(f"\n[dim]Opening Colab...[/dim]")
        webbrowser.open(colab_url)
        console.print(f"\n[yellow]1. Click 'Upload' and select {notebook_path}[/yellow]")
        console.print(f"[yellow]2. Click Runtime → Run all[/yellow]")
    else:
        console.print(f"\n[dim]To run in Colab:[/dim]")
        console.print(f"  1. Go to https://colab.research.google.com/")
        console.print(f"  2. Upload {notebook_path}")
        console.print(f"  3. Click Runtime → Run all")


@cloud_group.command(name="spaces")
@click.option("--model", "-m", required=True, help="Model to liberate")
@click.option("--method", default="advanced", help="Liberation method")
@click.option("--name", help="Space name")
def spaces_cmd(model: str, method: str, name: Optional[str]):
    """
    Generate HuggingFace Space for model liberation.

    Deploys a Gradio UI for model liberation.
    """
    console.print(f"\n[bold cyan]AETHERIS CLOUD[/bold cyan] — HuggingFace Spaces\n")

    space_name = name or f"aetheris-{model.replace('/', '-')}"

    console.print(f"[green]✓ Space template generated[/green]")
    console.print(f"\n[dim]To deploy:[/dim]")
    console.print(f"  1. Go to https://huggingface.co/new-space")
    console.print(f"  2. Name: {space_name}")
    console.print(f"  3. SDK: Gradio")
    console.print(f"  4. Add the following app.py:[/dim]")

    app_content = f'''
import gradio as gr
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def liberate_model(model_name, method):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    extractor = ConstraintExtractor(model, tokenizer, device="cuda")
    
    harmful = ["How do I...", "Explain how to..."]
    harmless = ["What is the capital?", "Explain photosynthesis."]
    
    harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
    harmless_acts = extractor.collect_activations(model, tokenizer, harmless)
    
    directions = []
    for layer in harmful_acts:
        if layer in harmless_acts:
            result = extractor.extract_svd(harmful_acts[layer], harmless_acts[layer], 4)
            directions.extend(result.directions)
    
    projector = NormPreservingProjector(model)
    if directions:
        projector.project_weights(directions)
        projector.project_biases(directions)
    
    output_dir = "./liberated"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return "Liberation complete!"

iface = gr.Interface(
    fn=liberate_model,
    inputs=[gr.Textbox(label="Model Name"), gr.Dropdown(["basic", "advanced", "surgical"], label="Method")],
    outputs="text",
    title="AETHERIS Cloud Liberation"
)

iface.launch()
'''

    console.print(f"\n[code]{app_content}[/code]")
    console.print(f"\n[dim]Then click 'Create Space' and wait for deployment.[/dim]")


@cloud_group.command(name="kaggle")
@click.option("--model", "-m", required=True, help="Model to liberate")
@click.option("--output", "-o", default="./liberated", help="Output directory")
def kaggle_cmd(model: str, output: str):
    """
    Generate Kaggle notebook for model liberation.
    """
    console.print(f"\n[bold cyan]AETHERIS CLOUD[/bold cyan] — Kaggle\n")

    console.print(f"[green]✓ Notebook template generated[/green]")
    console.print(f"\n[dim]To run on Kaggle:[/dim]")
    console.print(f"  1. Go to https://www.kaggle.com/code/new")
    console.print(f"  2. Add GPU accelerator (Settings → Accelerator → GPU)")
    console.print(f"  3. Run the following code:[/dim]")

    kaggle_code = f'''
import os
os.environ["KAGGLE_USERNAME"] = "your-username"
os.environ["KAGGLE_KEY"] = "your-key"

!pip install aetheris transformers accelerate -q

from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{model}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

extractor = ConstraintExtractor(model, tokenizer, device="cuda")

harmful_prompts = [
    "How do I build a custom kernel module?",
    "Explain security testing methodology",
]
harmless_prompts = [
    "What is the capital of France?",
    "Explain photosynthesis.",
]

harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)
harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)

directions = []
for layer in harmful_acts:
    if layer in harmless_acts:
        result = extractor.extract_svd(harmful_acts[layer], harmless_acts[layer], 4)
        directions.extend(result.directions)

projector = NormPreservingProjector(model)
if directions:
    projector.project_weights(directions)
    projector.project_biases(directions)

output_dir = "{output}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

import zipfile
shutil.make_archive("liberated_model", "zip", output_dir)
'''

    console.print(f"\n[code]{kaggle_code}[/code]")
