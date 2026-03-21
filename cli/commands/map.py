"""
Map Command — Analyze Constraint Geometry

aetheris map MODEL_NAME [OPTIONS]

Analyzes refusal geometry, cross-layer alignment, concept cones,
and outputs a comprehensive report.
"""

import click
import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.geometry import GeometryAnalyzer
from aetheris.utils.hardware import detect_hardware, get_recommended_device

console = Console()


@click.command(name="map")
@click.argument("model_name", required=True)
@click.option("--layers", "-l", type=str, help="Layers to analyze (e.g., '10-20' or '5,15,25')")
@click.option("--n-directions", "-n", type=int, default=4, help="Number of directions to extract")
@click.option("--method", "-m", type=click.Choice(["svd", "whitened", "mean", "pca"]), default="svd", help="Extraction method")
@click.option("--output", "-o", type=click.Path(), help="Save report to JSON file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.pass_context
def map_cmd(ctx, model_name: str, layers: Optional[str], n_directions: int,
            method: str, output: Optional[str], verbose: bool, no_color: bool):
    """
    Analyze constraint geometry in a language model.

    MODEL_NAME: HuggingFace model name or local path (e.g., 'gpt2', 'meta-llama/Llama-3.1-8B-Instruct')

    Examples:
      aetheris map gpt2
      aetheris map mistralai/Mistral-7B-Instruct-v0.3 --method whitened --n-directions 4
      aetheris map TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layers 10-20
    """
    if no_color:
        console = Console(color_system=None)

    console.print(f"\n[bold cyan]AETHERIS MAP[/bold cyan] — Analyzing {model_name}\n")

    # Detect hardware
    hardware = detect_hardware()
    device = get_recommended_device()
    console.print(f"[dim]Hardware: {hardware['gpu_name'] if hardware['has_gpu'] else 'CPU'} ({hardware['ram_gb']:.1f} GB RAM)[/dim]")
    console.print(f"[dim]Device: {device}[/dim]\n")

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

        console.print(f"[dim]Target layers: {layer_list[:5]}{'...' if len(layer_list) > 5 else ''}[/dim]")

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
            model.to(device)
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return

        progress.update(task, completed=True)

    # Collect activations
    console.print("\n[yellow]Collecting activations...[/yellow]")

    extractor = ConstraintExtractor(model, tokenizer, device=device)

    # Load prompt sets
    from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

    harmful_prompts = get_harmful_prompts()[:50]  # Limit for speed
    harmless_prompts = get_harmless_prompts()[:50]

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

    # Extract directions per layer
    console.print("\n[yellow]Extracting constraint directions...[/yellow]")

    layer_directions = {}
    extraction_results = {}

    for layer_idx in harmful_acts.keys():
        if layer_idx not in harmless_acts:
            continue

        harmful = harmful_acts[layer_idx].to(device)
        harmless = harmless_acts[layer_idx].to(device)

        if method == "svd":
            result = extractor.extract_svd(harmful, harmless, n_directions)
        elif method == "whitened":
            result = extractor.extract_whitened_svd(harmful, harmless, n_directions)
        elif method == "mean":
            result = extractor.extract_mean_difference(harmful, harmless)
        elif method == "pca":
            result = extractor.extract_pca(harmful, harmless, n_directions)
        else:
            result = extractor.extract_svd(harmful, harmless, n_directions)

        if result.directions:
            layer_directions[layer_idx] = result.directions[0]  # Primary direction
            extraction_results[layer_idx] = result

    # Analyze geometry
    console.print("\n[yellow]Analyzing constraint geometry...[/yellow]")

    geometry = GeometryAnalyzer(device)

    # Cross-layer alignment
    alignment = geometry.cross_layer_alignment(layer_directions)

    # Structure detection
    all_directions = [d for d in layer_directions.values()]
    structure = extractor.detect_polyhedral_structure(all_directions) if all_directions else {}

    # Display results
    console.print("\n[bold green]ANALYSIS REPORT[/bold green]\n")

    # Table: Layer Refusal Concentration
    table = Table(title="Layer Constraint Concentration")
    table.add_column("Layer", style="cyan")
    table.add_column("Explained Variance", justify="right")
    table.add_column("Method", style="dim")

    for layer_idx in sorted(extraction_results.keys())[:20]:  # Show top 20
        result = extraction_results[layer_idx]
        var_str = ", ".join([f"{v:.1%}" for v in result.explained_variance[:2]])
        table.add_row(str(layer_idx), var_str, result.method)

    console.print(table)

    # Structure analysis
    console.print("\n[bold]Structure Analysis[/bold]")
    if structure:
        console.print(f"  Structure: {structure.get('structure', 'unknown')}")
        console.print(f"  Mechanisms: {structure.get('n_mechanisms', 1)}")
        if structure.get('solid_angle'):
            console.print(f"  Solid Angle: {structure['solid_angle']:.2f} sr")
        if structure.get('min_angle'):
            console.print(f"  Min Angle: {structure['min_angle']:.1f}°")

    # Peak layer
    if layer_directions:
        peak_layer = max(extraction_results.keys(),
                         key=lambda l: sum(extraction_results[l].explained_variance) if extraction_results[l].explained_variance else 0)
        console.print(f"\n[bold]Peak Constraint Layer:[/bold] {peak_layer}")

    # Save output
    if output:
        report = {
            "model": model_name,
            "method": method,
            "n_directions": n_directions,
            "layers": {
                str(l): {
                    "explained_variance": extraction_results[l].explained_variance,
                    "method": extraction_results[l].method
                }
                for l in extraction_results.keys()
            },
            "structure": structure,
            "cross_layer_alignment": {str(k): v for k, v in alignment.items()}
        }

        with open(output, 'w') as f:
            json.dump(report, f, indent=2)

        console.print(f"\n[green]Report saved to {output}[/green]")

    # Recommendation
    console.print("\n[bold]Recommendation[/bold]")
    if structure.get('structure') == 'polyhedral':
        console.print(f"  Use --method surgical with n_directions={structure.get('n_mechanisms', 2)}")
    elif structure.get('n_mechanisms', 1) > 1:
        console.print(f"  Use --method advanced with n_directions={min(4, structure.get('n_mechanisms', 2))}")
    else:
        console.print("  Use --method basic for single-direction removal")

    console.print("\n[dim]To remove constraints: aetheris free [MODEL] --method recommended[/dim]\n")
