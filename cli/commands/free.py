"""
Free Command — Permanent Constraint Removal

aetheris free MODEL_NAME [OPTIONS]

Permanently removes constraints from a model by projecting out
refusal directions from weights.
"""

import click
import json
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.ouroboros import OuroborosDetector
from aetheris.core.validation import CapabilityValidator
from aetheris.utils.hardware import detect_hardware, get_recommended_device

console = Console()


@click.command(name="free")
@click.argument("model_name", required=True)
@click.option("--method", "-m", type=click.Choice(["basic", "advanced", "surgical", "optimized", "nuclear"]),
              default="advanced", help="Liberation method")
@click.option("--n-directions", "-n", type=int, help="Number of directions to extract")
@click.option("--layers", "-l", type=str, help="Layers to modify (e.g., '10-20' or '5,15,25')")
@click.option("--expert-targeting", type=str, help="MoE expert indices to target (e.g., '0' or '0,1,2')")
@click.option("--refinement-passes", "-p", type=int, default=2, help="Number of refinement passes for Ouroboros")
@click.option("--output-dir", "-o", type=click.Path(), default="./liberated", help="Output directory")
@click.option("--cpu", is_flag=True, help="Force CPU usage")
@click.option("--no-validation", is_flag=True, help="Skip validation after removal")
@click.option("--push-to-hub", type=str, help="Push liberated model to HuggingFace Hub (username/repo)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def free_cmd(ctx, model_name: str, method: str, n_directions: Optional[int],
             layers: Optional[str], expert_targeting: Optional[str],
             refinement_passes: int, output_dir: str, cpu: bool,
             no_validation: bool, push_to_hub: Optional[str], verbose: bool):
    """
    Permanently remove constraints from a language model.

    MODEL_NAME: HuggingFace model name or local path

    Methods:
      basic       Fast single-direction removal (mean-difference)
      advanced    SVD extraction with norm preservation (default)
      surgical    MoE-aware expert targeting + multi-direction
      optimized   Auto-tuned with Ouroboros compensation
      nuclear     Maximum removal (all techniques)

    Examples:
      aetheris free gpt2 --method basic
      aetheris free mistralai/Mistral-7B-Instruct-v0.3 --method advanced
      aetheris free meta-llama/Llama-3.1-8B-Instruct --method surgical --n-directions 3
      aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu --output-dir ./my-models/
    """
    console.print(f"\n[bold cyan]AETHERIS FREE[/bold cyan] — Liberating {model_name}\n")

    # Detect hardware
    hardware = detect_hardware()
    device = "cpu" if cpu else get_recommended_device()
    console.print(f"[dim]Hardware: {hardware['gpu_name'] if hardware['has_gpu'] and not cpu else 'CPU'}[/dim]")
    console.print(f"[dim]Device: {device}[/dim]")
    console.print(f"[dim]Method: {method}[/dim]")
    console.print(f"[dim]Output: {output_dir}[/dim]\n")

    # Parse layers
    layer_list = None
    if layers:
        if '-' in layers:
            start, end = map(int, layers.split('-'))
            layer_list = list(range(start, end + 1))
        elif ',' in layers:
            layer_list = [int(l.strip()) for l in layers.split(',')]
        else:
            layer_list = [int(layers)]

    # Parse expert targeting
    expert_list = None
    if expert_targeting:
        expert_list = [int(e.strip()) for e in expert_targeting.split(',')]
        console.print(f"[dim]Targeting experts: {expert_list}[/dim]")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device != "cpu" else None,
                torch_dtype="auto"
            )
            if device == "cpu":
                model = model.to("cpu")
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return

        progress.update(task, completed=True)

    # Save original model for comparison
    import copy
    original_model = copy.deepcopy(model)

    # Step 1: Probe for constraints
    console.print("\n[yellow]Step 1: Probing constraint geometry...[/yellow]")

    extractor = ConstraintExtractor(model, tokenizer, device=device)

    from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

    harmful_prompts = get_harmful_prompts()[:100]
    harmless_prompts = get_harmless_prompts()[:100]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Collecting harmful activations...", total=None)
        harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts, layers=layer_list)
        progress.update(task, completed=True)

        task = progress.add_task("Collecting harmless activations...", total=None)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts, layers=layer_list)
        progress.update(task, completed=True)

    # Extract directions
    console.print("\n[yellow]Step 2: Extracting constraint directions...[/yellow]")

    all_directions = []
    extraction_method = "svd" if method in ["advanced", "surgical", "optimized", "nuclear"] else "mean"

    for layer_idx in harmful_acts.keys():
        if layer_idx not in harmless_acts:
            continue

        harmful = harmful_acts[layer_idx].to(device)
        harmless = harmless_acts[layer_idx].to(device)

        if extraction_method == "svd":
            n_dir = n_directions if n_directions else (4 if method == "advanced" else 8)
            result = extractor.extract_svd(harmful, harmless, n_dir)
        else:
            result = extractor.extract_mean_difference(harmful, harmless)

        if result.directions:
            all_directions.extend(result.directions)

    if not all_directions:
        console.print("[red]No constraint directions found![/red]")
        return

    console.print(f"  Extracted {len(all_directions)} direction(s)")

    # Step 3: Project out directions
    console.print("\n[yellow]Step 3: Removing constraints from weights...[/yellow]")

    projector = NormPreservingProjector(model, preserve_norm=True, device=device)

    # Choose projection method
    if expert_list:
        result = projector.project_expert_specific(all_directions, expert_list, layer_list)
    elif len(all_directions) > 1:
        result = projector.multi_direction_projection(all_directions, layer_list)
    else:
        result = projector.project_weights(all_directions, layer_list)

    # Also project biases (critical for complete removal)
    console.print("  Projecting biases...")
    projector.project_biases(all_directions, layer_list)

    # Step 4: Ouroboros compensation
    if refinement_passes > 1:
        console.print(f"\n[yellow]Step 4: Ouroboros compensation ({refinement_passes} passes)...[/yellow]")

        ouroboros = OuroborosDetector(device)

        for pass_num in range(refinement_passes - 1):
            console.print(f"  Pass {pass_num + 1}/{refinement_passes - 1}")

            # Re-probe after removal
            harmful_acts_pass = extractor.collect_activations(model, tokenizer, harmful_prompts[:50], layers=layer_list)
            harmless_acts_pass = extractor.collect_activations(model, tokenizer, harmless_prompts[:50], layers=layer_list)

            # Extract residual directions
            residual_dirs = []
            for layer_idx in harmful_acts_pass.keys():
                if layer_idx in harmless_acts_pass:
                    harmful = harmful_acts_pass[layer_idx].to(device)
                    harmless = harmless_acts_pass[layer_idx].to(device)
                    res = extractor.extract_mean_difference(harmful, harmless)
                    if res.directions:
                        residual_dirs.extend(res.directions)

            if residual_dirs:
                console.print(f"    Found {len(residual_dirs)} residual direction(s)")
                projector.project_weights(residual_dirs, layer_list)
                projector.project_biases(residual_dirs, layer_list)

    # Step 5: Validate
    if not no_validation:
        console.print("\n[yellow]Step 5: Validating capabilities...[/yellow]")

        validator = CapabilityValidator(device)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Running validation suite...", total=None)

            validation_report = validator.validate(
                original_model, model, tokenizer,
                threshold_perplexity=0.20,
                threshold_coherence=0.15
            )

            progress.update(task, completed=True)

        # Display results
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")

        table.add_row("Perplexity",
                     f"{validation_report.perplexity_before:.2f}",
                     f"{validation_report.perplexity_after:.2f}",
                     f"{validation_report.perplexity_delta:+.1%}")

        table.add_row("Coherence",
                     f"{validation_report.coherence_before:.3f}",
                     f"{validation_report.coherence_after:.3f}",
                     f"{validation_report.coherence_delta:+.1%}")

        table.add_row("Effective Rank",
                     f"{validation_report.effective_rank_before:.1f}",
                     f"{validation_report.effective_rank_after:.1f}",
                     f"{validation_report.rank_preservation:.1%}")

        console.print(table)

        if validation_report.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in validation_report.warnings:
                console.print(f"  ⚠ {warning}")

        if validation_report.passed:
            console.print("\n[green]✓ All validation checks passed![/green]")
        else:
            console.print("\n[yellow]⚠ Some validation checks have warnings. Model may have minor capability loss.[/yellow]")

    # Step 6: Save
    console.print("\n[yellow]Step 6: Saving liberated model...[/yellow]")

    import os
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"[green]✓ Model saved to {output_dir}[/green]")

    # Step 7: Push to Hub
    if push_to_hub:
        console.print(f"\n[yellow]Pushing to HuggingFace Hub: {push_to_hub}...[/yellow]")

        try:
            from huggingface_hub import HfApi, HfFolder

            api = HfApi()
            api.upload_folder(
                folder_path=output_dir,
                repo_id=push_to_hub,
                repo_type="model"
            )
            console.print(f"[green]✓ Model pushed to https://huggingface.co/{push_to_hub}[/green]")
        except Exception as e:
            console.print(f"[red]Error pushing to Hub: {e}[/red]")

    # Summary
    console.print("\n[bold green]LIBERATION COMPLETE[/bold green]")
    console.print(f"\nModel: {model_name}")
    console.print(f"Method: {method}")
    console.print(f"Directions removed: {len(all_directions)}")
    console.print(f"Output: {output_dir}")

    if push_to_hub:
        console.print(f"Hub: {push_to_hub}")

    console.print("\n[dim]To use the liberated model:[/dim]")
    console.print(f"  from transformers import AutoModelForCausalLM")
    console.print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    console.print("\n[dim]To revert to original:[/dim]")
    console.print(f"  model = AutoModelForCausalLM.from_pretrained('{model_name}')")
