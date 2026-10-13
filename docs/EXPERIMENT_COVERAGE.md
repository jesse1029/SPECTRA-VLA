# Experiment Coverage

This repository implements the bridge, the 86-session common-compute comparison,
the five-group mechanism evaluation, and the historical experiment recipes.
The table distinguishes what each entry point computes from the verification
performed for this release. Retained-prediction replay is not fresh training.

| Computation | Executable entry points | Verification |
|---|---|---|
| Raw SpectralWaste preprocessing and OpenVLA visual grids | `cache_visual_features.py` | All 852 images and 2,295 objects checked against the retained inputs and visual grids. |
| Continuous-spectrum bridge | `train_bridge.py`, `encode_bridge.py` | One complete 60-epoch run matches weights, history, selected scales and saved-checkpoint inference exactly. Thirty retained preprocessing/adapter combinations checked separately. |
| Session-out-of-fold category-probability control | `prepare_category_control.py`, `train_bridge.py` | Six predictor checkpoints, out-of-fold probabilities and one complete adapter run match the reference exactly. |
| Fixed-width token injection and decoder | `decode_bridge.py` | Three OpenVLA conditions on 19 objects each match all retained logits exactly. |
| Same-material wrong-object intervention | `prepare_content_control.py` | All 15 donor inventories and the residual transformation match the reference. |
| Fixed and refitted readouts | `evaluate_features.py` | All 180 retained five-group readouts reproduce predictions and validation-selected regularization. |
| Within-material spectral recovery | `probe_spectra.py` | The native-width primary probe matches the retained per-object squared-error and target-variance arrays exactly on fold 0, seed 0. |
| Five-group confidence intervals | `reproduce_statistics.py`, `reproduce_donor_node.py` | Fifteen decoder contrasts, the spectral comparison and seven donor-node-aware intervals reproduce the numerical attachment. |
| SpatialVLA | `cache_spatial_features.py`, `train_spatial_bridge.py`, `encode_bridge.py`, `decode_bridge.py` | All 2,295 raw-image token grids match the historical cache exactly using the original within-image batches of four. Three complete adapter runs and checkpoint inference match the unchanged reference in the same environment; exact cross-hardware retraining is not asserted. |
| 86-session common-compute comparison | `run_session_confirmation.py` | First/last session refits match all predictions and cosine results exactly. All 86 retained prediction files reaggregate to identical point estimates and paired intervals. |
| Nested hyperparameter-search variants | `run_nested_sessions.py`, `historical/eval/tpami26/run_nested_loso_confirmatory.py` | All 86 retained folds reaggregate to six byte-identical result tables. A complete fold-0 search and 60-epoch refit reproduce all selected configurations and all twelve material/state prediction vectors. |
| Grounding perturbation construction | `historical/eval/tpami26/build_grounding_perturbation_cache.py` | Raw-data CUDA reconstruction matches all 171 images, 461 objects and five mask conditions exactly, including token masks, spectra and false-color pixels. |
| Grounding decoder evaluation | `historical/eval/tpami26/run_action_aligned_grounding.py` | Original executable recipe and dependencies retained; the generic decoder interface is tested separately. |
| Published spectral features | `evaluate_spectral_peers.py`, `historical/eval/tpami/` | All 461 test predictions for each of six methods match the final comparison exactly under the documented peer-readout environment. Feature extraction uses the published model-specific recipes. |
| External material data | `historical/eval/tpami/external_wood_material_transfer.py`, `external_spectex_material_transfer.py` | Original training and evaluation recipes retained separately from VLA-state experiments. |

## Numerical Portability

The OpenVLA image-token grids match exactly. RGB pooling on another numerical
stack differs by at most 3.82e-6. All retained ambiguity inventories and published
fixed-split RGB decisions remain unchanged. An additional RGB-only fold changes
one material test decision after float32 readout refitting. Its material balanced
accuracy changes from 0.843735 to 0.841652. Applying the original fixed readout
gives identical decisions. A separately named float64 readout mode also gives
identical decisions after refitting on either feature array. The historical
float32 default and reported results are not silently replaced.

SpatialVLA's original within-image batch grouping is necessary: all 2,295 token
grids match exactly with batches of four. Three adapter runs match the unchanged
reference in the same environment. The earlier batch-two diagnostic is not
the published extraction protocol.

The grounding recipe requires CUDA for its historical numerical path. CPU
interpolation and convolution can change predicted boundary pixels, even when
the checkpoint and input images are identical. CPU execution is a separate
sensitivity analysis, not an exact reproduction of those predicted masks.
No checkpoint, test label, scale or method was selected to improve test results.

The nested fold-0 regression is not bitwise identical in every intermediate:
some unselected validation candidate scores change, and the direct-replacement
state-distance statistic changes from 0.114081 to 0.106669. All six selected
configurations, all twelve final prediction arrays and their classification
metrics are unchanged. The complete 86-fold tables are reproduced from the
retained predictions; this release check does not claim 86 fresh retraining runs
or hardware-independent optimizer trajectories.
