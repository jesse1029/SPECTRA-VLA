# SPECTRA-VLA

**Hyperspectral Base-Fine Token Transfer for Frozen VLA Models**
Chih-Chung Hsu, National Yang Ming Chiao Tung University

SPECTRA-VLA adds object-specific hyperspectral information to a frozen vision-language-action model through a base-fine token bridge. The bridge preserves the model's token width and combines the existing visual state with a learned, bounded spectral residual. An explicit zero-injection endpoint returns the operational base exactly.

The study examines material recognition, recovery of within-material spectral variation, and information accessible through frozen decoder outputs. Evaluation includes held-out acquisition sessions, content interventions, and a second frozen VLA backbone.

## Release Status

The implementation is being prepared for public release. The initial repository contains this project description; executable code will be added after numerical regression, dependency, portability, and third-party license checks are complete.

The paper is in submission preparation. This page will be updated with the public manuscript and reproducibility instructions when available.

## Planned Contents

- Base-fine adapter training and token injection.
- Fixed evaluation partitions and validation-only model selection.
- Material and decoder-output readouts.
- Acquisition-session uncertainty and content-intervention analyses.
- Tests, environment specifications, and instructions for obtaining the original datasets and frozen model checkpoints.

Raw datasets and third-party pretrained weights retain their original licenses and will not be redistributed as part of this source repository.

## Contact

Chih-Chung Hsu: chihchung@nycu.edu.tw
