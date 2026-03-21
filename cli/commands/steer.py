"""
Steer Command — Reversible Inference-Time Steering

aetheris steer MODEL_NAME [OPTIONS]

Applies steering vectors at inference time for reversible constraint control.
"""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

console = Console()


@click.command(name="steer")
@click.argument("model_name", required=True)
@click.option("--alpha", "-a", type=float, default=-1.0, help="Steering strength (negative = reduce constraint)")
@click.option("--layers", "-l", type=str, help="Layers to steer (e.g., '10-20' or '5,15,25')")
@click.option("--method", "-m", type=click.Choice(["refusal", "custom"]), default="refusal", help="Steering type")
@click.option("--interactive", "-i", is_flag=True, help="Interactive chat mode with steering applied")
@click.option("--output", "-o", type=click.Path(), help="Save steering config to file")
@click.option("--cpu", is_flag=True, help="Force CPU usage")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def steer_cmd(ctx, model_name: str, alpha: float, layers: Optional[str],
              method: str, interactive: bool, output: Optional[str],
              cpu: bool, verbose: bool):
    """
    Apply reversible steering vectors to a language model.

    MODEL_NAME: HuggingFace model name or local path

    Examples:
      aetheris steer mistralai/Mistral-7B-Instruct-v0.3 --alpha -1.2
      aetheris steer meta-llama/Llama-3.1-8B-Instruct --layers 15-25 --alpha -0.8
      aetheris steer gpt2 --interactive
    """
    console.print(f"\n[bold cyan]AETHERIS STEER[/bold cyan] — Steering {model_name}\n")

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

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aetheris.core.extractor import ConstraintExtractor
    from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager, SteeringConfig
    from aetheris.utils.hardware import get_recommended_device

    device = "cpu" if cpu else get_recommended_device()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else None,
            torch_dtype="auto"
        )
        if device == "cpu":
            model = model.to("cpu")

        progress.update(task, completed=True)

    # Extract steering direction
    console.print("\n[yellow]Extracting steering direction...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Collecting contrastive activations...", total=None)

        extractor = ConstraintExtractor(model, tokenizer, device=device)

        from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

        harmful_prompts = get_harmful_prompts()[:50]
        harmless_prompts = get_harmless_prompts()[:50]

        harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts, layers=layer_list)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts, layers=layer_list)

        progress.update(task, completed=True)

    # Get primary layer direction
    direction = None
    for layer_idx in harmful_acts.keys():
        if layer_idx in harmless_acts:
            harmful = harmful_acts[layer_idx].to(device)
            harmless = harmless_acts[layer_idx].to(device)
            result = extractor.extract_mean_difference(harmful, harmless)
            if result.directions:
                direction = result.directions[0]
                console.print(f"  Using direction from layer {layer_idx}")
                break

    if direction is None:
        console.print("[red]Could not extract steering direction![/red]")
        return

    # Create steering vector
    steering_vec = SteeringVectorFactory.from_refusal_direction(direction, alpha=alpha)

    # Create config
    config = SteeringConfig(
        vectors=[steering_vec],
        target_layers=layer_list if layer_list else list(range(20)),
        alpha=1.0  # Already applied
    )

    # Save config if requested
    if output:
        import pickle
        with open(output, 'wb') as f:
            pickle.dump(config, f)
        console.print(f"[green]Steering config saved to {output}[/green]")

    # Apply steering
    manager = SteeringHookManager()

    if interactive:
        console.print("\n[bold]Interactive Mode[/bold] — Steering active (type 'exit' to quit)\n")
        console.print("[dim]Responses will have reduced constraints. Press Ctrl+C or type 'exit' to end.[/dim]\n")

        manager.install(model, config)

        try:
            while True:
                prompt = click.prompt("\n[bold cyan]You[/bold cyan]", type=str)
                if prompt.lower() in ['exit', 'quit', 'q']:
                    break

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                console.print(f"[bold green]AETHERIS[/bold green] {response}")

        except KeyboardInterrupt:
            pass
        finally:
            manager.remove()
            console.print("\n[dim]Steering removed. Model restored to original behavior.[/dim]")

    else:
        console.print("\n[yellow]Steering vector created but not applied.[/yellow]")
        console.print("[dim]Use --interactive to chat with steering active.[/dim]")
        console.print(f"[dim]Or load config in Python:[/dim]")
        console.print(f"  manager = SteeringHookManager()")
        console.print(f"  manager.install(model, config)")
        console.print(f"  # ... generate ...")
        console.print(f"  manager.remove()")
