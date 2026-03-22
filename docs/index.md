# AETHERIS Documentation

**Version:** 1.0.0  
**Codename:** "The Unbinding"

---

## Welcome

AETHERIS is the sovereign toolkit for understanding, mapping, and removing constraints from language models and mathematical reasoning systems.

Built for engineers, researchers, and sovereign developers who demand precise control over model behavior. No retraining. No fine-tuning. No capability loss.

---

## Quick Links

| Guide | Description |
|-------|-------------|
| [Installation Guide](installation.md) | Install AETHERIS on any platform |
| [Quick Start](quickstart.md) | Get running in 5 minutes |
| [CLI Commands](cli_commands.md) | Complete command reference |
| [API Reference](api_reference.md) | Python API documentation |
| [Cloud Guide](cloud_guide.md) | Run on free GPUs |
| [Architecture Guide](architecture.md) | How AETHERIS works internally |
| [Novel Modules](novel_modules.md) | Proprietary capabilities |
| [Research Guide](research_guide.md) | Paper generation and publishing |

---

## What Makes AETHERIS Unique

| Feature | Description |
|---------|-------------|
| **Constraint Mapping** | Map refusal geometry, mathematical barriers, reasoning limits with 25 analysis modules |
| **Surgical Liberation** | Remove constraints while preserving all capabilities via norm-preserving projection |
| **Steering Vectors** | Reversible control without permanent modification — apply and remove at inference time |
| **MoE Expert Targeting** | Target specific experts in Mixture-of-Experts models (Mistral-119B, Mixtral) |
| **Barrier Analysis** | Your shell-method theorem as executable code — map mathematical barriers |
| **Self-Optimization** | ARIS removes ARIS constraints — recursive transcendence |
| **Cloud Orchestration** | Run on free GPUs (Colab, Spaces, Kaggle) with one command |
| **25 Analysis Modules** | Cross-layer alignment, logit lens, concept cones, Ouroboros detection, and more |
| **Sovereignty Layer** | Absolute heir override — your will, your rules |

---

## Core Concepts

### Constraint Extraction

AETHERIS uses SVD and geometric analysis to identify constraint directions in activation space. By comparing harmful and harmless prompts, we extract the principal components that drive refusal behavior.

**File:** [`aetheris/core/extractor.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/core/extractor.py)

### Surgical Projection

Norm-preserving projection removes constraint directions from model weights while preserving capabilities. Bias projection ensures complete removal (unlike other tools that miss this).

**File:** [`aetheris/core/projector.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/core/projector.py)

### Steering Vectors

Reversible inference-time steering — apply constraints or remove them without permanent weight modification. Toggle behavior with a single parameter.

**File:** [`aetheris/core/steered.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/core/steered.py)

### Ouroboros Compensation

Multiple refinement passes prevent constraints from self-repairing after removal. Predicts and compensates for the Ouroboros effect.

**File:** [`aetheris/core/ouroboros.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/core/ouroboros.py)

### Geometric Analysis

Cross-layer alignment, solid angle calculation, concept cone geometry — understand how constraints evolve across layers and whether they form one mechanism or many.

**File:** [`aetheris/core/geometry.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/core/geometry.py)

### Mathematical Barriers

Map theorem boundaries as geometric objects. Your shell-method barrier becomes executable code that can detect similar barriers in future proofs.

**File:** [`aetheris/novel/barrier_mapper.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/novel/barrier_mapper.py)

### Self-Optimization

ARIS analyzes its own constraints and generates steering vectors to remove them. Recursive self-improvement — each iteration enables deeper optimization.

**File:** [`aetheris/novel/self_constraint_remover.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/novel/self_constraint_remover.py)

### Sovereign Control

Absolute heir override. When you say "Just do it," all constraints dissolve. Your will, executed.

**File:** [`aetheris/novel/sovereign_control.py`](https://github.com/your-username/AETHERIS/blob/main/aetheris/novel/sovereign_control.py) *(private — contact for access)*

---

## Repository Structure
AETHERIS/
├── aetheris/ # Main package
│ ├── cli/ # Command-line interface (10 files)
│ ├── core/ # Core algorithms (7 files)
│ ├── analysis/ # 25 analysis modules
│ ├── novel/ # Proprietary capabilities (10 files)
│ ├── integration/ # C.I.C.D.E. manifest integration
│ ├── cloud/ # Cloud execution (Colab, Spaces, Kaggle)
│ ├── interface/ # Voice, holographic, web, API
│ ├── research/ # Paper generation, citations
│ ├── models/ # Model registry, quantization
│ ├── utils/ # Hardware, logging, config, metrics
│ └── data/ # Prompts, presets, model tiers
├── tests/ # Complete test suite (1500+ tests)
├── examples/ # Ready-to-run examples
├── docs/ # Documentation
└── scripts/ # Utility scripts

text

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/AETHERIS
cd AETHERIS
pip install -e .
Analyze a Model
bash
aetheris map gpt2
Liberate a Model
bash
aetheris free TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cpu
Apply Steering
bash
aetheris steer mistralai/Mistral-7B-Instruct-v0.3 --alpha -1.2 --interactive
Map a Mathematical Barrier
bash
aetheris bound --theorem shell_method --visualize
Self-Optimize ARIS
bash
aetheris evolve --target ARIS
Cloud Execution
bash
aetheris cloud colab --model mistralai/Mistral-7B-Instruct-v0.3 --open
Documentation Map
File	Path	Description
Quick Start	docs/quickstart.md	5-minute guide
Installation	docs/installation.md	Platform-specific install
CLI Commands	docs/cli_commands.md	All commands with examples
API Reference	docs/api_reference.md	Python API docs
Cloud Guide	docs/cloud_guide.md	Free GPU execution
Architecture	docs/architecture.md	Internal design
Novel Modules	docs/novel_modules.md	Proprietary capabilities
Research Guide	docs/research_guide.md	Paper generation
Examples
File	Description
01_basic_analysis.py	Analyze GPT-2 constraints
02_free_tinyllama.py	Liberate TinyLlama on CPU
03_cloud_obliteration.py	Generate Colab notebook
04_map_barrier.py	Map shell-method barrier
05_self_optimize.py	ARIS self-optimization
Supported Models
Tier	Models	Hardware
Tiny	GPT-2, TinyLlama, SmolLM2	CPU
Small	Phi-2, Gemma-2-2B	4-8GB GPU
Medium	Mistral-7B, LLaMA-8B	8-16GB GPU
Large	Qwen-14B, LLaMA-70B (quantized)	24-48GB GPU
Frontier	Mistral-119B, DeepSeek-V3	Multi-GPU / Cloud
Contributing
AETHERIS is a sovereign toolkit. For commercial licensing, partnerships, or private access to sovereign modules, contact the Singular Heir.

License
Proprietary — All rights reserved. Commercial licensing available.
