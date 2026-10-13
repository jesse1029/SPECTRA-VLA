# SPECTRA-VLA

**Hyperspectral Base-Fine Token Transfer for Frozen VLA Models**  
Chih-Chung Hsu, National Yang Ming Chiao Tung University

SPECTRA-VLA adds object-specific hyperspectral information to a frozen vision-language-action model through a base-fine token bridge. The bridge preserves the model's token width and combines the existing visual state with a learned, bounded spectral residual. An explicit zero-injection endpoint returns the operational base exactly.

The study examines material recognition, recovery of within-material spectral variation, and information accessible through frozen decoder outputs. Evaluation includes held-out acquisition sessions, content interventions, and a second frozen VLA backbone.

## Release Status

This repository contains the implementation, experiment recipes, and regression tests. Start with [Reproduction](docs/REPRODUCTION.md) for the required environments, data layout, and executable commands. [Experiment Coverage](docs/EXPERIMENT_COVERAGE.md) distinguishes raw-input checks, fresh training runs, retained-prediction replay, and known numerical portability differences.

The paper is in submission preparation. This page will be updated with the public manuscript and reproducibility instructions when available.

## Contents

- Base-fine adapter training and token injection.
- Fixed evaluation partitions and validation-only model selection.
- Material and decoder-output readouts.
- Acquisition-session uncertainty and content-intervention analyses.
- Tests, environment specifications, and instructions for obtaining the original datasets and frozen model checkpoints.

The [historical recipes](docs/HISTORICAL_RECIPES.md) provide the original fixed-split, nested-session, grounding, and external spectral-model experiments. Their protocols are distinct from the five-group mechanism evaluation; use the documented inputs and numerical environment for each.

Raw datasets and third-party pretrained weights retain their original licenses and will not be redistributed as part of this source repository.

## Contact

Chih-Chung Hsu: chihchung@nycu.edu.tw
