"""Frozen SpatialVLA interface with its native visual and action vocabulary widths."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch


SPATIAL_EXTRACTION_BATCH_SIZE = 4


def spatial_object_batches(images):
    """Keep the historical per-image groups: ZoeDepth routing is batch-dependent."""
    for start in range(0, len(images), SPATIAL_EXTRACTION_BATCH_SIZE):
        stop = min(start + SPATIAL_EXTRACTION_BATCH_SIZE, len(images))
        yield start, stop, images[start:stop]


class FrozenSpatialVLA:
    def __init__(self, model_dir, device='cuda'):
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('the SpatialVLA reproduction requires CUDA')
        model_dir = Path(model_dir).resolve()
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype).eval().to(self.device)
        self.model.requires_grad_(False)
        self.image_token_index = int(self.model.config.image_token_index)
        self.action_token_begin = int(self.model.config.action_token_begin_idx)
        self.action_token_count = int(self.model.config.spatial_token_num)
        self.visual_token_count = int(self.processor.image_seq_length)
        self.token_dim = int(self.model.config.text_config.hidden_size)

    @torch.inference_mode()
    def extract_visual_tokens(self, images):
        if not len(images):
            raise ValueError('image batch is empty')
        inputs = self.processor(images=list(images), text=['observe the object'] * len(images),
                                return_tensors='pt', padding=True)
        pixels = inputs['pixel_values'].to(self.device, dtype=self.dtype)
        intrinsic = inputs['intrinsic'].to(self.device)
        tokens = self.model.get_image_features(pixels, intrinsic)
        if tokens.shape != (len(images), self.visual_token_count, self.token_dim):
            raise ValueError('SpatialVLA returned an unexpected visual token shape')
        if not torch.isfinite(tokens).all():
            raise FloatingPointError('nonfinite SpatialVLA visual tokens')
        return tokens

    @torch.inference_mode()
    def forward_from_visual_tokens(self, visual_tokens, instructions):
        if (visual_tokens.ndim != 3 or len(visual_tokens) != len(instructions) or
                visual_tokens.shape[1:] != (self.visual_token_count, self.token_dim) or
                not len(visual_tokens) or not torch.isfinite(visual_tokens).all()):
            raise ValueError('invalid visual tokens or instruction count')
        dummy = Image.new('RGB', (224, 224), color=0)
        inputs = self.processor(images=[dummy] * len(instructions), text=list(instructions),
                                return_tensors='pt', padding=True)
        ids = inputs['input_ids'].to(self.device)
        attention = inputs['attention_mask'].to(self.device)
        embedded = self.model.get_input_embeddings()(ids).clone()
        image_mask = ids == self.image_token_index
        if not torch.all(image_mask.sum(1) == self.visual_token_count):
            raise ValueError('each prompt must contain exactly one complete visual-token grid')
        expanded = image_mask.unsqueeze(-1).expand_as(embedded)
        embedded = embedded.masked_scatter(expanded, visual_tokens.to(self.device, dtype=embedded.dtype))
        output = self.model(input_ids=ids, inputs_embeds=embedded, attention_mask=attention,
                            pixel_values=None, intrinsic=None, use_cache=False,
                            return_dict=True, num_logits_to_keep=1)
        logits = output.logits[:, -1, self.action_token_begin:self.action_token_begin + self.action_token_count]
        if logits.shape != (len(visual_tokens), self.action_token_count) or not torch.isfinite(logits).all():
            raise FloatingPointError('invalid SpatialVLA action logits')
        return logits.float()

    def forward(self, visual):
        instruction = 'What action should the robot take to sort the object into the correct bin?'
        return self.forward_from_visual_tokens(visual, [instruction] * len(visual)).cpu().numpy()


def historical_false_color(cube):
    """Per-image view used for the published SpatialVLA replication."""
    from .raw_data import band_resample, resize_image

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or not np.isfinite(cube).all():
        raise ValueError('HSI must be a finite band-first cube')
    low = np.percentile(cube, 2, axis=(1, 2), keepdims=True)
    high = np.percentile(cube, 98, axis=(1, 2), keepdims=True)
    normalized = np.clip((cube - low) / (high - low + 1e-6), 0, 1).astype(np.float32)
    # The original dataset quantizes resized cubes to float16 before selecting bands.
    normalized = resize_image(band_resample(normalized)).astype(np.float16)
    bands = normalized[[24, 60, 96]]
    channels = []
    for band in bands:
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        channels.append(np.clip((band - lo) / (hi - lo + 1e-6), 0, 1))
    return Image.fromarray((np.stack(channels, -1) * 255).astype(np.uint8))
