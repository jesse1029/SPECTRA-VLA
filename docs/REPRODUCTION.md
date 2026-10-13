# Reproduction

## Environment

The checked environment uses Python 3.11.9, PyTorch 2.6.0 with CUDA 12.4,
NumPy 1.26.4 and scikit-learn 1.6.1. Exact Python dependencies are listed in
`requirements-reproduction.txt`. Obtain the CUDA-enabled PyTorch wheel from
the official PyTorch package index, then install this project:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

The adapter and statistical tests support CPU execution. Frozen OpenVLA decoder
inference requires CUDA. The checked 16 GB configuration uses batch size 4,
bfloat16 and SDPA; measured peak allocation was 13,861,235,200 bytes. It does
not silently fall back to CPU or change numerical precision.

## Statistical Reproduction Without Model Training

The numerical attachment contains out-of-fold predictions, spectral losses,
recipient/donor session identifiers, and reference results. It contains no
raw HSI cubes, model weights or executable Python objects.

The numerical attachment is separate from this source release. Use an authorized
copy; no private attachment link is distributed with the code.

Archive SHA-256:
`c7a672acfd1a58aab6b45fde253454e8341b418c65e963ce72820b8a74e8b057`

Extract the archive and pass the directory containing `DATA_INVENTORY.json`:

```bash
python scripts/reproduce_statistics.py --root DATA --output results/session_statistics.json
python scripts/reproduce_donor_node.py --root DATA --output results/donor_statistics.json
```

These commands reproduce 15 decoder contrasts, the continuous-versus-posterior
spectral comparison, and seven recipient/donor-node-aware intervals. They
average optimization seeds before interpreting sampling uncertainty. They do
not retrain adapters and do not constitute a raw-data end-to-end rerun.

## Raw Images To Frozen Visual Features

Obtain the SpectralWaste segmentation data and the official OpenVLA-7B
checkpoint under their publishers' terms. The data directory must contain
`hyper`, `rgb` and `labels_hyper_lt`, each with the original `train`, `val`
and `test` partitions. The model directory must include the official processor,
configuration, custom model source and safetensors shards.

Verify the local publisher files before extraction. This catches changed model
code, processor settings or weight shards without uploading any weights:

```bash
python scripts/verify_checkpoint.py --backbone openvla --model OPENVLA
```

```bash
python scripts/cache_visual_features.py --data SPECTRALWASTE --model OPENVLA --output cache
```

The fixed false-color view uses resampled bands 119, 60 and 0, divided by the
uint16 sensor range. No dataset-wide image statistics are fitted. The command
extracts connected components of at least 100 pixels after resizing to 224 by
224, then forms nonempty 16-by-16 area masks. It preserves the original object
order and writes `cache/inputs.json` for subsequent commands. Allow about 5 GB
for the resulting features. An existing output directory is never overwritten.

The original 852 images and 2,295 objects were checked against the archived
spectra, masks and both sets of float16 patch tokens. All matched exactly in the
checked environment. CPU-pooled RGB features differ from the original platform
by at most 3.82e-6; their ambiguity-pair inventories are unchanged. These small
differences can affect a separately refitted readout near a decision boundary,
so bitwise reproduction of arbitrary downstream RGB classifiers is not claimed.

## Bridge Training From Frozen Visual Features

`configs/inputs.example.json` defines explicit paths to numeric arrays. Paths
may be absolute or relative to the configuration file. The input contract is:

- Visual tokens: float16 arrays with shape `(objects, 256, 4096)`.
- Target masks: float16 occupancy arrays with shape `(objects, 256)`.
- Operational base: float32 target-pooled, normalized features of width 4096.
- Object spectra: float32 arrays with 120 bands.
- Paired-RGB tokens: aligned width-4096 features used to construct ambiguity controls.
- Metadata: a non-pickled NPZ with aligned `y_material`, `y_action`, `image_id`,
  `session` and `split` vectors.

Shard order must match metadata order. `configs/partitions/fold0.json` through
`fold4.json` contain the fixed acquisition-group partitions. They are valid only
for the original 2,295-object inventory and are bound to its metadata order;
2,291 eligible objects are evaluated
out of fold across 86 acquisition sessions.

```bash
python scripts/train_bridge.py --inputs inputs.json --partition configs/partitions/fold0.json --seed 0 --output runs/fold0_seed0 --device cuda
```

Run the five folds with seeds 0, 1 and 2. The script uses rank 64, norm budget
0.1, the original 60-epoch optimizer schedule, and validation-only selection.
It saves adapter weights, fitted NMF and scaler state, selection details and
representations. Both the unrestricted and fixed-displacement operating points
are retained. No test metric is used to choose either point.

To reuse a saved checkpoint without fitting preprocessing or selecting a new
operating point:

```bash
python scripts/encode_bridge.py --inputs inputs.json --checkpoint runs/fold0_seed0/adapter.pt --preprocessing runs/fold0_seed0/spectral_preprocessing.npz --selection runs/fold0_seed0/selection.json --output runs/fold0_seed0/reencoded.npz --device cuda
```

## Frozen Decoder And Content Controls

The matched category-probability control trains five session-disjoint predictors
inside the training partition. Training objects receive out-of-fold predictions;
validation and test objects receive predictions from the outer-training model.
The resulting six probabilities are zero-padded to the same 136-input width.

```bash
python scripts/prepare_category_control.py --inputs inputs.json --partition configs/partitions/fold0.json --seed 0 --output runs/category_fold0_seed0 --device cuda
python scripts/train_bridge.py --inputs inputs.json --partition configs/partitions/fold0.json --seed 0 --category-codes runs/category_fold0_seed0/codes.npz --output runs/category_bridge_fold0_seed0 --device cuda
```

Code files are bound to metadata order, partition and seed. The general saved
checkpoint encoder above accepts continuous-spectrum bridges only; this control
uses the representations saved by its training command.

Saved residuals also carry an object-order identity. Decoder and donor commands
reject residuals from another inventory, even when the array dimensions match.
The encoder verifies that preprocessing and validation selection belong to the
supplied checkpoint. Historical residual files without this identity are not
accepted implicitly.

Obtain OpenVLA-7B from its [official repository](https://huggingface.co/openvla/openvla-7b).
Do not redistribute its checkpoint through this repository.

```bash
python scripts/decode_bridge.py --inputs inputs.json --representations runs/fold0_seed0/representations.npz --model MODEL --composition addition --output runs/fold0_seed0/addition.npy
python scripts/decode_bridge.py --inputs inputs.json --representations runs/fold0_seed0/representations.npz --model MODEL --composition replacement --output runs/fold0_seed0/replacement.npy
python scripts/prepare_content_control.py --inputs inputs.json --partition configs/partitions/fold0.json --representations runs/fold0_seed0/representations.npz --seed 0 --output runs/fold0_seed0/donor.npz
```

Pass `donor.npz` to the same decoder commands for the wrong-object controls.
The donor diagnostic uses material labels to replace an object's residual
with another same-material object's residual, then matches recipient geometry
and norm. It is an evaluation intervention, not an inference requirement.

```bash
python scripts/probe_spectra.py --inputs inputs.json --partition configs/partitions/fold0.json --features runs/fold0_seed0/addition.npy --conditional --output runs/fold0_seed0/spectral_probe
```

This command measures spectral-shape information in decoder logits. To reproduce
the primary native-width, within-material representation probe instead, use the
validation-selected common-budget representation:

```bash
python scripts/probe_spectra.py --inputs inputs.json --partition configs/partitions/fold0.json --features runs/fold0_seed0/representations.npz --feature-key pooled_budget --conditional --output runs/fold0_seed0/representation_shape_probe
```

Apply the same command to the category-control bridge for its matched comparison.
`--pool signed` selects the 256-dimensional sensitivity analysis; `--target level`
keeps absolute spectral intensity rather than unit-normalized spectral shape.
`--shuffled-training-targets` runs the within-material shuffled-target control.
These are distinct predeclared analyses, not alternatives selected on test scores.

The readout measures spectral information in fixed features. Its ridge
regularization is chosen on validation data; training and validation examples
are not refitted together. Conditional centering uses material labels only as
a common diagnostic. The readout is not an OpenVLA policy head.

## Fixed And Refitted Decoder Readouts

Create the zero-injection logits with `decode_bridge.py --zero-injection`, using
the same inputs and saved representations. Then fit a readout on those logits:

```bash
python scripts/evaluate_features.py --metadata cache/metadata.npz --partition configs/partitions/fold0.json --features runs/fold0_seed0/zero.npy --label action --pool none --output runs/fold0_seed0/base_readout
python scripts/evaluate_features.py --metadata cache/metadata.npz --partition configs/partitions/fold0.json --features runs/fold0_seed0/addition.npy --label action --pool none --fixed-readout runs/fold0_seed0/base_readout --output runs/fold0_seed0/addition_fixed
python scripts/evaluate_features.py --metadata cache/metadata.npz --partition configs/partitions/fold0.json --features runs/fold0_seed0/addition.npy --label action --pool none --output runs/fold0_seed0/addition_refit
```

The first comparison retains the base-fitted scaler and classifier. The second
fits a new classifier on training objects and selects regularization only on
validation objects. Both save predictions and a numeric, non-pickled readout.
For material representations use `--label material --pool signed`; an NPZ input
also requires `--feature-key`, such as `pooled_unrestricted`.

The default preserves the original numerical dtype. For a separately identified
high-precision refit, pass `--fit-dtype float64`. The saved readout records this
choice and applies the same dtype at inference. This option is not a replacement
for a published float32 readout. The RGB sensitivity check finds identical
decisions with the original fixed readout and with either float64 refit; the
additional float32 refit can change a boundary decision after tiny pooling
roundoff changes.

## SpatialVLA Interface

SpatialVLA needs a separate environment because its Transformers version differs
from OpenVLA. Install the project with `pip install -e ".[spatial]"` in that
environment, not on top of the pinned OpenVLA environment. Obtain
`IPEC-COMMUNITY/spatialvla-4b-224-pt` from its official publisher.
The SpatialVLA extra pins NumPy 2.2.6, matching the retained extraction environment;
NumPy scalar-promotion changes otherwise alter the float16 false-color conversion.
The retained checkpoint revision is `886c4557dfde53b44c0f74fde7d89bf83a26c3cd`.
Run `python scripts/verify_checkpoint.py --backbone spatialvla --model SPATIALVLA`
to check all configuration, tokenizer, custom source and weight files.

Visual extraction keeps the original grouping: up to four consecutive target
objects from the same image per batch, restarting the grouping for each image.
This is part of the inference protocol, not a throughput setting. SpatialVLA's
ZoeDepth component combines domain votes across a batch before choosing its
depth-estimation branch. Changing batch size or mixing objects from different
images can therefore change visual features. The extraction command fixes this
grouping and records it in `inputs.json`; do not reduce the batch size after an
out-of-memory error or replace it with cross-image batching.

```bash
python scripts/cache_spatial_features.py --data SPECTRALWASTE --openvla-inputs cache/inputs.json --model SPATIALVLA --output spatial_cache
python scripts/train_spatial_bridge.py --inputs spatial_cache/inputs.json --seed 0 --output runs/spatial_seed0
python scripts/decode_bridge.py --backbone spatialvla --inputs spatial_cache/inputs.json --representations runs/spatial_seed0/representations.npz --model SPATIALVLA --output runs/spatial_seed0/logits.npy
```

The interface preserves SpatialVLA's native 2,304-wide visual tokens and
8,194-entry first-position action vocabulary. Paired-RGB ambiguity features
retain their OpenVLA width and are not substituted for SpatialVLA visual tokens.
The historical SpatialVLA experiment used a different per-image false-color
view, implemented explicitly in `historical_false_color`; the main OpenVLA view
remains the fixed-sensor construction.

Repeat the training command for seeds 1 and 2. This recipe retains the historical
image-disjoint partition, final-epoch checkpoint, and pooled-state selection;
it is distinct from the session-group OpenVLA recipe. Scale selection uses
alpha=1 and validation balanced accuracy plus 0.25 times pair resolution.
After fixing the representation, the final readout regularization is selected
on validation data. The generated `partition.json` explicitly records an image
split, not a session holdout. Use it with `evaluate_features.py --pool signed`.
For the frozen base-fitted action readout, first decode with `--zero-injection`,
fit the base readout, and pass it through `--fixed-readout` for the injected state.
`encode_bridge.py` also restores this recipe without fitting new preprocessing.

All 2,295 objects have been re-extracted from raw images with the original
per-image batch grouping. All 1,392 training, 442 validation and 461 test token
grids match the archived float16 arrays exactly, including their payload hashes.
The portable training
recipe, all three selected scales, residuals, representations and restored
checkpoint outputs match the unchanged reference implementation exactly in the
same environment. Separate cold-start checks also match the original and
portable wrapper tokens and logits exactly for six objects.

The initial portability check incorrectly used extraction batches of two;
the original run used four. Because ZoeDepth routing is batch-dependent, this
changed features even with identical images and weights. Restoring the original
grouping resolves that discrepancy without changing the archived experiment.
Adapter retraining on a different GPU can still select different scales;
the same-runtime implementation comparison and historical raw-token replay
are separate checks. Exact cross-hardware retraining is not asserted.
Neither the historical predictions nor manuscript numbers have been replaced.

## The 86-Session Representation Comparison

```bash
python scripts/run_session_confirmation.py --inputs cache/inputs.json --output runs/session_confirmation
```

This is the common-compute comparison: each eligible acquisition session is
held out in turn, spectral preprocessing is refitted on the other sessions,
and every method is trained for 60 epochs. Injection scales, alignment strength
and readout regularization are inherited from the original validation-only
experiment in `configs/session_confirmation.json`. They are not selected on
the held-out session. This recipe is not a nested hyperparameter search.

The command runs the base, proposed bridge, constraint ablation, capacity
control and high-anchoring direct replacement. It writes per-session predictions,
aggregate results and paired session-bootstrap intervals. `--start-fold` and
`--stop-fold` allow an explicitly partial run; such a run is marked `PARTIAL`
and does not produce an all-session summary. Existing outputs are not overwritten.

The first and last session have been fully refitted with the portable recipe.
All ten material/state prediction arrays and token-cosine results match the
retained results exactly for each session. All 86 retained prediction files were
also reaggregated: point estimates and all eight paired confidence intervals
match exactly. These checks do not assert that all 86 models were retrained.

## Verification Scope

The release checks cover adapter training, preprocessing serialization,
injection, frozen-decoder inference, readouts, content controls and numerical
reanalysis. The decoder outputs are first-position action-token logits, not a
robot trajectory.

The current portable entry points include raw-image feature extraction, training,
saved-checkpoint inference and decoding for both OpenVLA and SpatialVLA. The
historical second-backbone image-token cache is reproduced exactly under the
documented extraction grouping. The 86-session
common-compute recipe is included separately. Nested search, external
foundation-model recipes and the full grounding grid have their own
[historical recipes](HISTORICAL_RECIPES.md). The five acquisition-group partitions
must not be relabeled as those experiments. Numerical replay must not be
presented as reproducing their upstream computations. Original datasets and checkpoints must be obtained under their
publishers' terms. The official checkpoint's local custom Python modules are
executed when its vision encoder is loaded; use the original publisher's
files, not an untrusted model directory.

See [experiment coverage](EXPERIMENT_COVERAGE.md) for the distinction between
implemented recipes, regression checks and historical computations.
