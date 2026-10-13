# Third-Party Components

The implementation uses PyTorch, torchvision, NumPy, SciPy, scikit-learn, Transformers, Accelerate, safetensors, timm, Pillow, imageio, tifffile, sentencepiece, and threadpoolctl as external dependencies. Their licenses apply to those packages. This repository does not bundle their implementations.

Frozen decoder evaluation uses the original [OpenVLA checkpoint](https://huggingface.co/openvla/openvla-7b). The model card identifies its release license as MIT. The [OpenVLA source license](https://github.com/openvla/openvla/blob/main/LICENSE) names Moo Jin Kim, Karl Pertsch, and Siddharth Karamcheti as copyright holders. Download checkpoints from the original publisher and retain their accompanying notices.

SpectralWaste and other datasets must be obtained from their original publishers under their own terms. Raw hyperspectral cubes, synchronized images, dataset archives, and pretrained third-party weights are not included in this source distribution.

The optional second-backbone interface loads the official [SpatialVLA checkpoint](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt) and its publisher-supplied custom Python modules. Its model card identifies the checkpoint license as MIT. The [SpatialVLA source repository](https://github.com/SpatialVLA/SpatialVLA) and checkpoint retain their original notices; neither is redistributed here.

No license grant for an external dataset or model is implied by the SPECTRA-VLA source release.

The historical spectral-feature interfaces load separately obtained SpectralGPT,
HyperSL, MaskedSST, DOFA and HyperSIGMA source/checkpoints. These model
implementations, their weights and the external wood/SpecTex datasets are not
bundled. Follow each original publisher's license and access conditions before
downloading or redistributing them. `historical/` contains this study's extraction
and evaluation code, not a relicensed copy of those foundation models.

Use only trusted local model and experiment checkpoints. Original PyTorch
checkpoint loading and legacy object-array archives may deserialize Python
objects. The cleaned numerical interfaces use non-pickled arrays and validate
object identities; do not supply untrusted historical archives.
