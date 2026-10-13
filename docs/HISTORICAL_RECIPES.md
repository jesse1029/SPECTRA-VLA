# Historical Experiment Recipes

The five-group evaluation, common-compute session comparison and nested search
are separate protocols. Do not substitute one for another. The original model
families remain available for controlled comparisons; SPECTRA-VLA is the single
proposed method.

## Nested Session Search

Use the pinned OpenVLA numerical environment and the cache created by
`scripts/cache_visual_features.py`:

```bash
python scripts/run_nested_sessions.py --inputs cache/inputs.json --protocol configs/nested_sessions.json --outdir runs/nested_sessions
```

The recipe selects rank, fine-loss weight, injection scale and ridge strength
inside each outer-training partition. It uses the original 12-epoch candidate
search, 60-epoch refit and seed schedule. The complete protocol is in
`configs/nested_sessions.json`. All 86 sessions must finish before an aggregate
result is produced. For an explicit partial run, use `--start-fold` and
`--stop-fold`; it is not an all-session result.

To independently reaggregate retained predictions without training:

```bash
python scripts/run_nested_sessions.py --inputs cache/inputs.json --protocol configs/nested_sessions.json --replay-folds retained/folds --outdir results/nested_replay
```

Several retained-fold directories may be supplied. Conflicting duplicate folds
are rejected. This mode reports `retained_prediction_replay`, not model retraining.

## Original Fixed-Split And Grounding Pipeline

Install the `historical` and `vision` extras in the pinned OpenVLA environment.
Run the following commands from the `historical` directory. Set the dataset and
checkpoint locations in `configs/spectra_spectralwaste.yaml`; use fresh output
directories and preserve the original train/validation/test partitions.

```bash
python -m eval.tpami.features --config configs/spectra_spectralwaste.yaml --outdir runs/fixed/features --what static
python eval/tpami26/cache_full_pipeline_inputs.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --outdir cache/full
python eval/tpami26/cache_falsecolor_variants.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --full-cache cache/full --outdir cache/variants
python eval/tpami26/regenerate_token_masks.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --outdir cache/area
python eval/tpami26/build_area_pooled_rgb_cache.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --paired-grid-cache cache/full --area-mask-cache cache/area --outdir cache/rgb
python eval/tpami26/audit_rgb_cache_alignment.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --manifest-root cache/full --rgb-cache cache/rgb
```

The static features establish original object ordering. The corrected fixed-sensor
false-color grids, nonempty area masks and area-pooled RGB features are then used
by the action-aligned experiments:

```bash
python eval/tpami26/run_action_aligned_composition.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --variant-cache cache/variants --shared-cache cache/area --rgb-cache cache/rgb --outdir runs/composition --rank 64 --epochs 60 --fine-budget 0.1 --hard-pair-weight 4 --seeds 0,1,2
python eval/tpami26/audit_final_payload_provenance.py --composition-dir runs/composition --out runs/composition/payload_provenance_audit.json
python eval/tpami26/run_action_aligned_logits.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --variant-cache cache/variants --shared-cache cache/area --rgb-cache cache/rgb --composition-dir runs/composition --model checkpoints/openvla-7b --outdir runs/action --cache-dir cache/action
```

`run_action_aligned_interventions.py`, `run_action_aligned_rank_capacity.py`,
`run_base_shift_augmentation.py` and `run_ambiguity_robustness_final.py` retain
their original command-line interfaces. The independent nested-action recipe
is `run_nested_loso_action_confirmatory.py`. Use `--help` for their named input
directories; do not feed five-group outputs into these fixed-split recipes.

For grounding, use the original trained FCN checkpoint or train the same backend
with `eval/tpami/strong_segmentation_backend.py`. Reconstruct the perturbations:

```bash
python eval/tpami26/build_grounding_perturbation_cache.py --config configs/spectra_spectralwaste.yaml --manifest cache/full/test/object_manifest.csv --reference-spectra cache/variants/raw_object_spectra__test.npy --fcn-checkpoint checkpoints/strong_fcn_resnet50_segmentation.pt --outdir cache/grounding --device cuda
python eval/tpami26/run_action_aligned_grounding.py --config configs/spectra_spectralwaste.yaml --campaign runs/fixed --variant-cache cache/variants --shared-cache cache/area --rgb-cache cache/rgb --composition-dir runs/composition --main-action-dir runs/action --main-action-cache cache/action --grounding-cache cache/grounding --model checkpoints/openvla-7b --outdir runs/grounding --cache-dir cache/grounding_logits
```

CUDA is part of the retained predicted-mask protocol. A CPU execution can change
boundary pixels through interpolation and convolution rounding. The raw-data
CUDA replay matches all five retained perturbation conditions exactly. That
check covers mask construction and masked inputs, not a fresh FCN training run.
The FCN is a grounding backend, not part of the proposed spectral bridge.

## Spectral Foundation Features

The original extraction recipes are `external_peer_comparison.py`,
`spectralgpt_peer_comparison.py` and `foundation_peer_expansion.py`, under
`historical/eval/tpami`. Obtain model implementations and weights from their
publishers. Do not mix checkpoint families or reinterpret a partial checkpoint
load as a ranked comparison.

The final SpectralWaste input uses 120 samples over 900--1700 nm. HyperSL receives
that complete wavelength vector; DOFA receives the same vector at its twelve
selected band indices, expressed in micrometers. The older visible-range
wavelength configuration is not used for the final peer comparison.

Final peer readouts need a separate environment described by
`requirements-peer-readouts.txt`. In that environment:

```bash
python scripts/evaluate_spectral_peers.py --inputs cache/inputs.json --peers configs/peers.example.json --output results/peers
```

Each feature archive must preserve object order and labels. Ridge strength is
selected on validation balanced accuracy, with smaller strength breaking ties.
The OpenVLA adapter environment and the final peer-readout environment are
different numerical profiles; exchanging them can change boundary decisions.
All six final peer prediction vectors match under the specified profile.

## External Material Data

`external_wood_material_transfer.py` implements the official Mendeley wood split;
`external_spectex_material_transfer.py` implements the documented within-source
SpecTex split. Their adapters are trained separately for each dataset. These are
material-transfer experiments, not zero-shot VLA policy transfer.

The current paper does not report a complete official WoodVIT benchmark run.
Invalid exploratory WoodVIT subsets are excluded from this release and must not
be used as published comparison results.
