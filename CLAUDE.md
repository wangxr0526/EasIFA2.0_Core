# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EasIFA Core is an **inference-only** deep learning package for predicting enzyme active sites from protein structures/sequences, optionally with reaction information. Extracted from the larger EasIFA2.0 training project — **no training code exists here**.

## Setup & Development

```bash
# One-command environment setup (downloads checkpoints + creates conda env)
bash quick_setup_intl.sh        # International (Hugging Face)
bash quick_setup_ch.sh          # Mainland China (accelerated mirrors)

# Development install (makes code changes immediately effective)
pip install -e .
```

**System requirements:** Python >=3.8, PyTorch >=1.10, 16GB RAM, 20GB disk for model checkpoints. GPU optional.

## Common Commands

```bash
# CLI inference (requires pip install -e .)
easifa-predict --enzyme-structure test/test_inferece_input/AF-F6KCZ5-F1-model_v4.pdb --output result.json
easifa-predict --enzyme-sequence "MSPRL..." --rxn-smiles "CC(=O)O>>CCO" --output result.json
easifa-predict --batch-input batch_input.json --output batch_results.json --verbose

# Standalone script (no installation needed)
python easifa_predict.py --enzyme-sequence "MSPRL..." --output result.json

# Validate installation
python -c "from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig; print('OK')"
```

**No unit test suite exists.** Validation is inference-based only, using test PDB files in `test/test_inferece_input/`.

## Architecture

### Four-Model Ensemble

The system dynamically selects one of four trained models based on available inputs (`_judge_data()` in `easifa_core/interface/inference.py`):

| Model | Inputs | Encoders |
|-------|--------|----------|
| `all_features` | structure + reaction | ESM + GearNet + Reaction attention |
| `wo_structures` | sequence + reaction | ESM + Reaction attention |
| `wo_reactions` | structure only | ESM + GearNet |
| `wo_rxn_structures` | sequence only | ESM only |

Selection happens **at inference time**, not configuration time. When both structure and sequence are provided, structure is preferred.

### Key Code Paths

- **Public API:** `easifa_core/__init__.py` exports `EasIFAInferenceAPI` and `EasIFAInferenceConfig`
- **Core inference logic:** `easifa_core/interface/inference.py` — `EasIFAInferenceAPI` class (model loading, `_judge_data()` selection, `inference()` method)
- **CLI:** `easifa_core/cli.py` — entry point registered as `easifa-predict`
- **Configuration:** `easifa_core/config.py` — `EasIFAInferenceConfig` with checkpoint paths, device allocation, model selection
- **Data preprocessing:** `easifa_core/data_loaders/` — protein parsing (`enzyme_dataloader.py`), reaction graph construction (`enzyme_rxn_dataloader.py`, `rxn_dataloader.py`)
- **Model architectures:** `easifa_core/model_structure/` — neural network definitions (must match trained checkpoint formats exactly)
- **Utilities:** `easifa_core/common/utils.py` — device management (`cuda()`), checkpoint loading (`read_model_state()` supports `.safetensors` and `.pth`)
- **Standalone script:** `easifa_predict.py` — mirrors CLI without requiring installation

### Dual Data Representation

Proteins are represented in two parallel formats simultaneously:
- **Graph-based** (TorchDrug): For GearNet structure encoding — built via `MyProtein.from_pdb()`
- **Sequence tokens** (ESM/SaProt): For transformer encoding — built via `MyProtein.from_sequence()`

Reactions use **DGL graphs** with separate reactant/product subgraphs + atom distance matrices, parsed from SMILES with `>>` separator via RDKit.

### Output Format

```python
pred, prob = easifa.inference(...)
# pred: List[int] — per-residue labels (0=non-site, 1=BINDING, 2=ACT_SITE, 3=SITE)
# prob: List[List[float]] — 4-class probability distributions per residue
# Returns None if input is invalid or sequence exceeds max_enzyme_aa_length (default 1000)
```

## Critical Conventions

- **Inference-only:** Model architectures in `model_structure/` must match trained checkpoint formats exactly. Changing layer dimensions or activations breaks checkpoint loading. Retraining happens in the main EasIFA2.0 project.
- **Device allocation is per-model**, not global. Configure via `EasIFAInferenceConfig.gpu_allocations` dict.
- **All 4 models loaded eagerly** by default (~8GB VRAM / ~4GB RAM). Reduce with `model_to_use=["wo_rxn_structures"]`.
- **No batch support at API level.** `inference()` processes one protein at a time; loop externally. CLI handles batching internally.
- **Always check return values:** `inference()` returns `None` silently for sequences exceeding length limit or invalid SMILES.
- **Checkpoint format is custom** (directory with `model.safetensors`/`model.pth` + `args.yml`), not standard `torch.save()`.
