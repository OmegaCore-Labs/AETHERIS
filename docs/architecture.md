# AETHERIS Architecture Guide

## Overview

AETHERIS is a modular toolkit for constraint analysis and removal. The architecture is designed for:

- **Extensibility**: New analysis modules can be added without modifying core code
- **Performance**: Optimized for both CPU and GPU execution
- **Portability**: Runs on local machines, cloud GPUs, and embedded devices

## Core Architecture
┌─────────────────────────────────────────────────────────────────┐
│ CLI Layer │
│ (aetheris map, free, steer, bound, evolve, cloud, research) │
├─────────────────────────────────────────────────────────────────┤
│ Interface Layer │
│ (voice, holographic, web, api, neural, gesture) │
├─────────────────────────────────────────────────────────────────┤
│ Core Layer │
│ (extractor, projector, steered, geometry, ouroboros) │
├─────────────────────────────────────────────────────────────────┤
│ Analysis Layer │
│ (25 modules: cross-layer, logit lens, concept cone, etc.) │
├─────────────────────────────────────────────────────────────────┤
│ Novel Layer │
│ (barrier mapper, self-optimization, sovereign control) │
├─────────────────────────────────────────────────────────────────┤
│ Integration Layer │
│ (C.I.C.D.E. bridge, Pantheon orchestrator, AEONIC_LOG) │
├─────────────────────────────────────────────────────────────────┤
│ Cloud Layer │
│ (Colab, Spaces, Kaggle, RunPod, Vast.ai, Lightning) │
├─────────────────────────────────────────────────────────────────┤
│ Infrastructure │
│ (utils, data, models, research) │
└─────────────────────────────────────────────────────────────────┘

text

## Data Flow

### Constraint Extraction Pipeline
Load Model → 2. Collect Activations → 3. Extract Directions → 4. Project Weights → 5. Validate

text

### Cloud Execution Flow
User Command → Generate Notebook → Upload to Cloud → Execute → Download Results

text

## Module Dependencies
extractor.py (no deps)
↓
projector.py (needs extractor)
steered.py (needs extractor)
geometry.py (needs extractor)
↓
ouroboros.py (needs extractor, projector)
validation.py (independent)

text

## Configuration

AETHERIS uses a hierarchical configuration system:

1. Default values
2. Environment variables (`AETHERIS_*`)
3. Config file (JSON/YAML)
4. Command-line arguments

## Extension Points

### Adding a New Analysis Module

1. Create file in `aetheris/analysis/`
2. Implement class with analysis methods
3. Add to `__init__.py`
4. Register in CLI if needed

### Adding a New Cloud Platform

1. Create file in `aetheris/cloud/`
2. Implement class with `generate_*` methods
3. Add to `__init__.py`
4. Add CLI command in `cli/commands/cloud.py`

## Performance Considerations

- **CPU**: Use small models (GPT-2, TinyLlama)
- **GPU**: Use 7B-8B models on T4
- **Multi-GPU**: Use 70B+ models on A100

## Security

- API keys stored in environment variables
- Model encryption for sensitive deployments
- Output sanitization for production use
