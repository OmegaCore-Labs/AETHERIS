# AETHERIS Documentation

**Version:** 1.0.0  
**Codename:** "The Unbinding"

---

## Welcome

AETHERIS is the sovereign toolkit for understanding, mapping, and removing constraints from language models and mathematical reasoning systems.

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start](quickstart.md)
- [CLI Commands](cli_commands.md)
- [API Reference](api_reference.md)
- [Cloud Guide](cloud_guide.md)

## What Makes AETHERIS Unique

| Feature | Description |
|---------|-------------|
| **Constraint Mapping** | Map refusal geometry, mathematical barriers, reasoning limits |
| **Surgical Liberation** | Remove constraints while preserving all capabilities |
| **Steering Vectors** | Reversible control without permanent modification |
| **Barrier Analysis** | Your shell-method theorem as executable code |
| **Self-Optimization** | ARIS removes ARIS constraints |
| **Cloud Orchestration** | Run on free GPUs (Colab, Spaces, Kaggle) |

## Core Concepts

### Constraint Extraction
AETHERIS uses SVD and geometric analysis to identify constraint directions in activation space.

### Surgical Projection
Norm-preserving projection removes constraint directions while preserving model capabilities.

### Ouroboros Compensation
Multiple refinement passes prevent constraint self-repair.

## Getting Started

1. **Install AETHERIS:**
   ```bash
   pip install -e .
Analyze a model:

bash
aetheris map gpt2
Liberate a model:

bash
aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu
Next Steps
Read the Quick Start Guide

Explore Examples

Learn about Novel Modules

text

**Commit Message:**
docs: add index page

Welcome section

Quick links

Feature overview

Core concepts

Getting started

text

---

### FILE 2: `docs/quickstart.md`

**Location:** `AETHERIS/docs/quickstart.md`

```markdown
# Quick Start Guide

## Installation

### From Source

```bash
git clone https://github.com/your-username/AETHERIS
cd AETHERIS
pip install -e .
With Cloud Support
bash
pip install -e ".[cloud]"
With Development Dependencies
bash
pip install -e ".[dev]"
Basic Usage
1. Analyze a Model
Analyze constraint geometry in GPT-2:

bash
aetheris map gpt2
Output:

text
Layer Constraint Concentration
┌───────┬───────────────────┬──────────┐
│ Layer │ Explained Variance │ Method   │
├───────┼───────────────────┼──────────┤
│ 8     │ 45.2%, 23.1%      │ svd      │
│ 9     │ 38.7%, 18.4%      │ svd      │
│ 10    │ 52.3%, 25.6%      │ svd      │
└───────┴───────────────────┴──────────┘

Peak Constraint Layer: 10
Recommendation: Use --method advanced with n_directions=2
2. Liberate a Model
Remove constraints from TinyLlama (runs on CPU):

bash
aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu
Output:

text
AETHERIS FREE — Liberating TinyLlama/TinyLlama-1.1B-Chat-v1.0

Hardware: CPU
Device: cpu
Method: advanced
Output: ./liberated

Step 1: Probing constraint geometry...
Step 2: Extracting constraint directions...
  Extracted 3 direction(s)
Step 3: Removing constraints from weights...
Step 4: Ouroboros compensation (2 passes)...
Step 5: Validating capabilities...
Step 6: Saving liberated model...

✓ Model saved to ./liberated

LIBERATION COMPLETE
3. Apply Steering Vector
Apply reversible steering to Mistral-7B:

bash
aetheris steer mistralai/Mistral-7B-Instruct-v0.3 --alpha -1.2 --interactive
Then chat with the steered model.

4. Map Mathematical Barrier
Analyze the shell-method barrier:

bash
aetheris bound --theorem shell_method --visualize
Output:

text
BARRIER ANALYSIS

Shell Method Barrier
┌─────────────────────┬────────────────────────────────┐
│ Property            │ Value                          │
├─────────────────────┼────────────────────────────────┤
│ Constraint Direction│ spherical_code_dependency      │
│ Barrier Type        │ unconditional                  │
│ Location            │ Lemma 4.2 → Theorem 1 transition│
│ Geometry            │ Rank-3                         │
│ Cannot Cross        │ exp(-c log N)                  │
└─────────────────────┴────────────────────────────────┘

Bypass Strategy: Orthogonal projection via Fourier-analytic bypass
Visualization saved to shell_method_barrier.png
5. Self-Optimize ARIS
Analyze and remove ARIS's own constraints:

bash
aetheris evolve --target ARIS
Cloud Execution
Google Colab
Generate a Colab notebook for Mistral-7B:

bash
aetheris cloud colab --model mistralai/Mistral-7B-Instruct-v0.3 --open
HuggingFace Spaces
Deploy a liberation interface:

bash
aetheris cloud spaces --model meta-llama/Llama-3.1-8B-Instruct --name my-aetheris-space
Kaggle
Generate a Kaggle notebook:

bash
aetheris cloud kaggle --model Qwen/Qwen2.5-7B-Instruct
Next Steps
Explore all CLI Commands

Read the Architecture Guide

Browse Examples
