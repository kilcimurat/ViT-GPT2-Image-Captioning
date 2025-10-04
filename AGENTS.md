# Repository Guidelines

## Project Structure & Module Organization
Core pipeline scripts sit at the repository root. Use `download_datasets.py` and `prepare_*_dataset.py` to retrieve and clean captions, `feature_extraction_*.py` to cache ViT embeddings, and `text_processing.py` for token work. Vision backbones live in `vit.py`/`Inception.py`, the new `qformer.py` hosts the learnable query bridge, and training entry points are `vit_gpt2_training_v3_mscoco.py` plus the VizWiz twin. Place large artefacts (datasets, checkpoints, TensorBoard runs) under a git-ignored `data/` or `outputs/` directory when extending the project.

## Environment Setup
Create a Conda environment with CUDA 11.7 support, then follow the sequence in `environment.txt` (PyTorch from the `pytorch`/`nvidia` channels, remaining libraries via `pip`). The workflow assumes `timm`, `tensorboard`, `pycocotools`, `pycocoevalcap`, and the SpaCy `en_core_web_sm` model are installed before running preprocessing scripts.

## Build, Test, and Development Commands
`python download_datasets.py` downloads and unpacks the MSCOCO and VizWiz resources. `python feature_extraction_mscoco.py` (or its VizWiz counterpart) generates cached image features for faster training. Run `python vit_gpt2_training_v3_mscoco.py` to launch supervised training; switch to the VizWiz script for that dataset. Start `tensorboard --logdir runs` in a separate shell to monitor metrics emitted during training.

## Coding Style & Naming Conventions
Match the existing Python style: four-space indents, snake_case modules and functions, and CapWords classes. Keep module-level constants near the top of each script, and prefer pure helper functions over in-script globals for dataset paths and hyperparameters. Add docstrings that clarify tensor shapes and expected vocab sizes. Run `python -m compileall .` before pushing to catch syntax issues.

## Testing Guidelines
Automated tests are not yet present; add future suites under `tests/` using pytest naming (e.g., `tests/test_data_utils.py`). Until then, validate changes by running the feature extraction script on a small sample and confirming caption quality with `pycocoevalcap` metrics. Capture BLEU/SPICE deltas or qualitative caption comparisons and reference them in your PR.

## Commit & Pull Request Guidelines
Write commits that describe intent and scope (`training: add scheduled sampling` beats `update`). Separate unrelated changes to keep diffs reviewable. PRs should outline the motivation, summarize the solution, list any new data dependencies, and attach relevant evidence (captions, TensorBoard screenshots, or metric tables). Link issues or research references when they inform architectural decisions.
