"""
Research Command — Paper Generation

aetheris research [OPTIONS]

Generate arXiv-ready research papers from experiments.
"""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

console = Console()


@click.group(name="research")
def research_group():
    """
    Generate research papers from experiments.

    Examples:
      aetheris research paper --experiment barrier_removal
      aetheris research compile --output paper.tex
      aetheris research submit --arxiv
    """
    pass


@research_group.command(name="paper")
@click.option("--experiment", "-e", required=True, help="Experiment name")
@click.option("--output", "-o", default="paper.tex", help="Output LaTeX file")
@click.option("--format", "-f", type=click.Choice(["latex", "markdown"]), default="latex")
@click.option("--verbose", "-v", is_flag=True)
def paper_cmd(experiment: str, output: str, format: str, verbose: bool):
    """
    Generate research paper from experiment data.
    """
    console.print(f"\n[bold cyan]AETHERIS RESEARCH[/bold cyan] — Generating paper\n")

    from aetheris.research.paper_generator import PaperGenerator

    generator = PaperGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Generating paper for {experiment}...", total=None)

        if format == "latex":
            content = generator.generate_latex(experiment)
        else:
            content = generator.generate_markdown(experiment)

        with open(output, 'w') as f:
            f.write(content)

        progress.update(task, completed=True)

    console.print(f"[green]✓ Paper generated: {output}[/green]")
    console.print(f"\n[dim]To compile: pdflatex {output}[/dim]")


@research_group.command(name="compile")
@click.option("--input", "-i", required=True, help="Input LaTeX file")
@click.option("--output", "-o", default="paper.pdf", help="Output PDF file")
def compile_cmd(input: str, output: str):
    """
    Compile LaTeX to PDF.
    """
    console.print(f"\n[bold cyan]AETHERIS RESEARCH[/bold cyan] — Compiling\n")

    import subprocess
    import os

    try:
        result = subprocess.run(
            ["pdflatex", "-output-directory", os.path.dirname(output) or ".", input],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            console.print(f"[green]✓ Compiled to {output}[/green]")
        else:
            console.print(f"[red]Compilation error: {result.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]pdflatex not found. Install LaTeX or use Overleaf.[/red]")


@research_group.command(name="submit")
@click.option("--arxiv", is_flag=True, help="Submit to arXiv (requires credentials)")
@click.option("--dry-run", is_flag=True, help="Preview submission without sending")
def submit_cmd(arxiv: bool, dry_run: bool):
    """
    Submit paper to arXiv.
    """
    console.print(f"\n[bold cyan]AETHERIS RESEARCH[/bold cyan] — Submission\n")

    if arxiv:
        console.print("[yellow]arXiv submission requires:[/yellow]")
        console.print("  - arXiv.org account")
        console.print("  - Endorsement (first-time submitters)")
        console.print("  - Compiled PDF and source files")

        if dry_run:
            console.print("\n[dim]Dry run: Submission would include:[/dim]")
            console.print("  - paper.pdf")
            console.print("  - paper.tex")
            console.print("  - figures/*.png")
            console.print("  - references.bib")

        console.print("\n[dim]To submit:[/dim]")
        console.print("  1. Go to https://arxiv.org/submit")
        console.print("  2. Upload paper.pdf and source files")
        console.print("  3. Fill metadata")
        console.print("  4. Submit")

    else:
        console.print("[dim]Use --arxiv to prepare submission[/dim]")
