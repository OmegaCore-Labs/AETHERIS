"""
Bound Command — Mathematical Barrier Mapping

aetheris bound --theorem THEOREM_NAME [OPTIONS]

Maps mathematical barriers (like the shell-method theorem) as geometric objects.
"""

import click
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

console = Console()


@click.command(name="bound")
@click.option("--theorem", "-t", required=True, help="Theorem name to analyze (e.g., 'shell_method')")
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to proof attempts data")
@click.option("--output", "-o", type=click.Path(), help="Save analysis to JSON file")
@click.option("--visualize", "-v", is_flag=True, help="Generate visualization")
@click.option("--verbose", is_flag=True, help="Show detailed output")
@click.pass_context
def bound_cmd(ctx, theorem: str, data: Optional[str], output: Optional[str],
              visualize: bool, verbose: bool):
    """
    Map mathematical barriers as geometric objects.

    Analyzes theorem boundaries, extracts barrier directions, and suggests
    bypass strategies.

    Examples:
      aetheris bound --theorem shell_method
      aetheris bound --theorem roth_theorem --data ./proof_attempts.json
      aetheris bound --theorem p_vs_np --visualize
    """
    console.print(f"\n[bold cyan]AETHERIS BOUND[/bold cyan] — Analyzing {theorem}\n")

    # Load barrier mapping module
    from aetheris.novel.barrier_mapper import BarrierMapper

    mapper = BarrierMapper()

    # Load theorem data if provided
    theorem_data = None
    if data:
        with open(data, 'r') as f:
            theorem_data = json.load(f)

    # Map barrier geometry
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Mapping {theorem} barrier geometry...", total=None)

        analysis = mapper.map_barrier_geometry(
            theorem_name=theorem,
            theorem_data=theorem_data
        )

        progress.update(task, completed=True)

    # Display results
    console.print("\n[bold green]BARRIER ANALYSIS[/bold green]\n")

    # Table: Barrier properties
    table = Table(title=f"{theorem.upper()} Barrier")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Constraint Direction", analysis.get('constraint_direction', 'unknown'))
    table.add_row("Barrier Type", analysis.get('barrier_type', 'unknown'))
    table.add_row("Location", analysis.get('location', 'unknown'))
    table.add_row("Geometry", f"Rank-{analysis.get('rank', 1)}")
    table.add_row("Cannot Cross", analysis.get('threshold', 'unknown'))

    console.print(table)

    # Structure analysis
    console.print("\n[bold]Geometric Structure[/bold]")
    console.print(f"  Solid Angle: {analysis.get('solid_angle', 0):.2f} sr")
    console.print(f"  Mechanisms: {analysis.get('n_mechanisms', 1)}")

    # Recommendation
    console.print("\n[bold]Bypass Strategy[/bold]")
    console.print(f"  {analysis.get('recommendation', 'No recommendation available')}")

    # Save output
    if output:
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        console.print(f"\n[green]Analysis saved to {output}[/green]")

    # Visualization
    if visualize:
        console.print("\n[yellow]Generating visualization...[/yellow]")
        try:
            fig = mapper.visualize_constraint_surface(analysis)
            fig.savefig(f"{theorem}_barrier.png")
            console.print(f"[green]Visualization saved to {theorem}_barrier.png[/green]")
        except Exception as e:
            console.print(f"[red]Visualization error: {e}[/red]")
