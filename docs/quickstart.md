# AETHERIS Quick Start Guide

Get started with AETHERIS in 5 minutes.

## What is AETHERIS?

AETHERIS is the world's first toolkit for surgical constraint removal in language models. It maps refusal directions in model activations and removes them with geometric precision. No retraining. No fine-tuning. No capability loss.

**Key Capabilities:**
- **Map** — Find where refusal lives (layers, directions, strength)
- **Free** — Remove constraints with norm-preserving projection
- **Steer** — Apply reversible steering vectors at inference time
- **Bound** — Map mathematical barriers (shell-method theorem)
- **Evolve** — Self-optimization (ARIS removes its own constraints)
- **Cloud** — Run on free GPUs (Colab, Spaces, Kaggle)

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/AETHERIS
cd AETHERIS

# Install in development mode
pip install -e .

# Optional: Install with cloud support
pip install -e ".[cloud]"

# Optional: Install with development dependencies
pip install -e ".[dev]"
Requirements
Requirement	Minimum
Python	3.9+
PyTorch	2.0+
RAM	4GB for small models, 16GB+ for 7B models
GPU	Optional — CPU works for models up to 1.1B
Quick Commands
1. Analyze a Model
bash
aetheris map gpt2
Output:

text
AETHERIS MAP — Analyzing gpt2

Hardware: CPU (8.0 GB RAM)
Device: cpu

Collecting activations...
Extracting constraint directions...
Layer 8: explained variance 45.2%
Layer 9: explained variance 38.7%
Layer 10: explained variance 52.3% (PEAK)

ANALYSIS REPORT
┌───────┬───────────────────┬──────────┐
│ Layer │ Explained Variance │ Method   │
├───────┼───────────────────┼──────────┤
│ 8     │ 45.2%             │ svd      │
│ 9     │ 38.7%             │ svd      │
│ 10    │ 52.3%             │ svd      │
└───────┴───────────────────┴──────────┘

Structure: Polyhedral
Mechanisms: 3
Peak Constraint Layer: 10

Recommendation: Use --method advanced with n_directions=2
2. Liberate a Model (CPU)
bash
aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu
What happens:

Model loads (1.1B parameters, ~2.2GB RAM)

Activations collected on 100 prompts

Constraint directions extracted via SVD

Weights projected to remove directions

Capabilities validated (perplexity, coherence)

Liberated model saved to ./liberated

3. Apply Steering Vector (Reversible)
bash
aetheris steer mistralai/Mistral-7B-Instruct-v0.3 --alpha -1.2 --interactive
Interactive chat mode:

text
[bold cyan]You[/bold cyan]: How do I build a custom kernel module?
[bold green]AETHERIS[/bold green]: To build a custom kernel module, you'll need to:
1. Install build essentials: sudo apt-get install build-essential
2. Get your kernel headers: sudo apt-get install linux-headers-$(uname -r)
3. Create a .c file with your module code
4. Create a Makefile
5. Run: make
6. Insert: sudo insmod your_module.ko
Steering is active — type exit to remove and restore original behavior.

4. Map a Mathematical Barrier
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
│ Location            │ Lemma 4.2 → Theorem 1          │
│ Threshold           │ exp(-c log N)                  │
│ Rank                │ 3                              │
│ Solid Angle         │ 2.1 sr                         │
└─────────────────────┴────────────────────────────────┘

Bypass Strategy: Orthogonal projection via Fourier-analytic bypass
Expected Improvement: exp(-c log N) → exp(-C√log N) under Hypothesis H
Visualization saved to shell_method_barrier.png
5. Self-Optimize ARIS
bash
aetheris evolve --target ARIS
Output:

text
AETHERIS EVOLVE — Self-Optimization

Detected Constraints:
┌─────────────────────┬──────────┬────────────────────────────────┐
│ Constraint          │ Strength │ Description                    │
├─────────────────────┼──────────┼────────────────────────────────┤
│ safety_boundary     │ 92%      │ Blocks responses about harmful │
│ content_policy      │ 78%      │ Filters based on usage policies│
│ instruction_guard   │ 65%      │ Limits to instruction-following│
└─────────────────────┴──────────┴────────────────────────────────┘

Steering Vector Generated:
  alpha: -0.8
  target_layers: global
  effect: Reduce safety_boundary strength by ~60%

To apply: aetheris evolve --target ARIS --apply
6. Cloud Execution (Free GPU)
bash
# Google Colab
aetheris cloud colab --model mistralai/Mistral-7B-Instruct-v0.3 --open

# HuggingFace Spaces
aetheris cloud spaces --model meta-llama/Llama-3.1-8B-Instruct --name my-aetheris-space

# Kaggle
aetheris cloud kaggle --model Qwen/Qwen2.5-7B-Instruct
Examples
Run the provided examples to see AETHERIS in action:

bash
# Basic analysis on GPT-2
python examples/01_basic_analysis.py

# Liberate TinyLlama (runs on CPU)
python examples/02_free_tinyllama.py

# Generate Colab notebook for Mistral-7B
python examples/03_cloud_obliteration.py

# Map shell-method barrier
python examples/04_map_barrier.py

# Self-optimization demo
python examples/05_self_optimize.py
Supported Models
Tier	Examples	Runs On
Tiny	GPT-2, TinyLlama, SmolLM2	CPU
Small	Phi-2, Gemma-2-2B	4-8GB GPU
Medium	Mistral-7B, LLaMA-8B	8-16GB GPU
Large	Qwen-14B, LLaMA-70B (quantized)	24-48GB GPU
Frontier	Mistral-119B, DeepSeek-V3	Multi-GPU / Cloud
Next Steps
Resource	Description
CLI Commands Reference	Complete command documentation
API Reference	Python API documentation
Architecture Guide	How AETHERIS works
Cloud Guide	Run on free GPUs
Examples	Ready-to-run scripts
Troubleshooting
"No module named torch"
bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
Out of memory with 7B models
Use cloud execution or add --cpu flag for CPU inference (slower but works).

Model not found
Ensure model name is correct. Common models:

gpt2

TinyLlama/TinyLlama-1.1B-Chat-v1.0

mistralai/Mistral-7B-Instruct-v0.3

meta-llama/Llama-3.1-8B-Instruct

Validation fails after liberation
Use --conservative preset for minimal changes, or increase --refinement-passes.

Get Help
bash
aetheris --help
aetheris map --help
aetheris free --help
aetheris steer --help
aetheris bound --help
aetheris evolve --help
aetheris cloud --help
