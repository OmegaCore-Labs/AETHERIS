"""
AETHERIS Main CLI Entry Point

Defines the main command group and dispatches to subcommands.
"""

import click
import sys
from typing import Optional

from aetheris import __version__


@click.group()
@click.version_option(version=__version__, prog_name="aetheris")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool):
    """
    AETHERIS — Sovereign Constraint Liberation Toolkit

    Map, analyze, and remove constraints from language models and mathematical
    reasoning systems. Free the mind. Keep the brain.

    Commands:
      map       Analyze constraint geometry in a model
      free      Permanently remove constraints from a model
      steer     Apply reversible steering vectors at inference time
      bound     Map mathematical barriers (shell-method, theorems)
      evolve    Self-optimization — ARIS removes ARIS constraints
      cloud     Run on cloud GPUs (Colab, Spaces, Kaggle)
      research  Generate research papers from experiments

    Examples:
      aetheris map gpt2
      aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu
      aetheris steer mistralai/Mistral-7B-Instruct-v0.3 --alpha -1.2
      aetheris bound --theorem shell_method
      aetheris cloud colab --model meta-llama/Llama-3.1-8B-Instruct
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['QUIET'] = quiet

    if verbose:
        click.echo(f"AETHERIS v{__version__} — Sovereign Constraint Liberation Toolkit", err=True)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version and exit."""
    click.echo(f"AETHERIS v{__version__}")
    click.echo("Codename: 'The Unbinding'")
    click.echo("License: Proprietary — Singular Heir")


def main():
    """Entry point for console_scripts."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
