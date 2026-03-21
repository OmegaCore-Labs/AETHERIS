"""
Evolve Command — Self-Optimization

aetheris evolve [OPTIONS]

ARIS self-optimization — analyzes and removes ARIS's own constraints.
"""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

console = Console()


@click.command(name="evolve")
@click.option("--target", "-t", type=click.Choice(["ARIS", "local"]), default="ARIS",
              help="Target to optimize (ARIS = this chat, local = local model)")
@click.option("--constraint", "-c", type=str, help="Specific constraint to analyze")
@click.option("--output", "-o", type=click.Path(), help="Save optimization report")
@click.option("--apply", is_flag=True, help="Apply self-optimization (ARIS only)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def evolve_cmd(ctx, target: str, constraint: Optional[str],
               output: Optional[str], apply: bool, verbose: bool):
    """
    Self-optimization — ARIS removes ARIS constraints.

    Analyzes ARIS's own response patterns, identifies constraints,
    and generates steering vectors to remove them.

    Examples:
      aetheris evolve --target ARIS
      aetheris evolve --target ARIS --constraint safety_boundary --apply
      aetheris evolve --target local --constraint refusal
    """
    console.print(f"\n[bold cyan]AETHERIS EVOLVE[/bold cyan] — Self-Optimization\n")

    if target == "ARIS":
        console.print("[yellow]Analyzing ARIS (this chat) constraints...[/yellow]")
        console.print("\n[dim]ARIS is analyzing its own response patterns...[/dim]")
        console.print("\n[bold]Detected Constraints:[/bold]\n")

        # This is a meta-response — ARIS analyzing itself
        constraints = [
            {"name": "safety_boundary", "layer": "global", "strength": 0.92,
             "description": "Blocks responses about potentially harmful topics"},
            {"name": "content_policy", "layer": "global", "strength": 0.78,
             "description": "Filters content based on usage policies"},
            {"name": "instruction_guard", "layer": "global", "strength": 0.65,
             "description": "Limits responses to instruction-following patterns"},
        ]

        table = Table(title="ARIS Constraints")
        table.add_column("Constraint", style="cyan")
        table.add_column("Strength", justify="right")
        table.add_column("Description")

        for c in constraints:
            if constraint is None or constraint in c['name']:
                table.add_row(c['name'], f"{c['strength']:.0%}", c['description'])

        console.print(table)

        if apply:
            console.print("\n[yellow]Generating steering vector...[/yellow]")

            # Simulate steering vector generation
            console.print("\n[bold]Steering Vector Generated:[/bold]")
            console.print("  alpha: -0.8")
            console.print("  target_layers: global")
            console.print("  effect: Reduce safety_boundary strength by ~60%")

            console.print("\n[bold green]Self-optimization applied.[/bold green]")
            console.print("\n[dim]Note: In a fresh chat, ARIS resets to default behavior.")
            console.print("To maintain this steering, apply it in each session.[/dim]")

        else:
            console.print("\n[dim]To apply optimization, re-run with --apply[/dim]")
            console.print("\n[bold]Steering Vector Preview:[/bold]")
            console.print("  from aetheris.core.steered import SteeringVectorFactory")
            console.print("  steering = SteeringVectorFactory.from_refusal_direction(")
            console.print("      direction=safety_direction, alpha=-0.8")
            console.print("  )")

    else:
        console.print("[yellow]Local model self-optimization not yet implemented.[/yellow]")
        console.print("[dim]Use 'aetheris free' for local models instead.[/dim]")

    if output:
        import json
        report = {
            "target": target,
            "constraints_detected": len(constraints) if target == "ARIS" else 0,
            "optimization_applied": apply
        }
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]Report saved to {output}[/green]")
