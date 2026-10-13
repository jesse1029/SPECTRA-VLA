"""Load only the official OpenVLA vision backbone and projector."""
import json
from functools import partial
from pathlib import Path

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
import torch
from transformers import AutoProcessor, LlamaConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def _second_last_patches(featurizer, pixels):
    # Recent timm releases return a list where the original OpenVLA code expects a tuple.
    result = featurizer.get_intermediate_layers(pixels, n={len(featurizer.blocks) - 2})
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError('expected exactly one second-to-last vision feature map')
    return result[0]


class FrozenOpenVLAVision:
    def __init__(self, model_dir, device='cuda'):
        model_dir = Path(model_dir).resolve()
        if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
            raise RuntimeError('the checked vision encoder requires CUDA')
        config = json.loads((model_dir / 'config.json').read_text())
        backbone_class = get_class_from_dynamic_module('modeling_prismatic.PrismaticVisionBackbone',
                                                       str(model_dir), local_files_only=True)
        projector_class = get_class_from_dynamic_module('modeling_prismatic.PrismaticProjector',
                                                        str(model_dir), local_files_only=True)
        with init_empty_weights():
            self.backbone = backbone_class(config['use_fused_vision_backbone'], config['image_sizes'],
                                            config['timm_model_ids'], config['timm_override_act_layers'])
            self.projector = projector_class(config['use_fused_vision_backbone'], self.backbone.embed_dim,
                                              LlamaConfig(**config['text_config']).hidden_size)
        for name in ['featurizer', 'fused_featurizer']:
            featurizer = getattr(self.backbone, name, None)
            if featurizer is not None:
                featurizer.forward = partial(_second_last_patches, featurizer)
        index = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
        needed = {k: v for k, v in index.items() if k.startswith(('vision_backbone.', 'projector.'))}
        loaded = {'vision_backbone': set(), 'projector': set()}
        modules = {'vision_backbone': self.backbone, 'projector': self.projector}
        for shard in sorted(set(needed.values())):
            filename = (model_dir / shard).resolve()
            if not filename.is_relative_to(model_dir):
                raise ValueError('checkpoint shard path escapes model directory')
            with safe_open(filename, framework='pt', device='cpu') as source:
                for key in needed:
                    if needed[key] != shard:
                        continue
                    prefix, name = key.split('.', 1)
                    set_module_tensor_to_device(modules[prefix], name, device,
                                                value=source.get_tensor(key), dtype=torch.bfloat16)
                    loaded[prefix].add(name)
        for prefix, module in modules.items():
            missing = set(module.state_dict()) - loaded[prefix]
            if missing or any(p.is_meta for p in module.parameters()):
                raise RuntimeError(f'incomplete {prefix} checkpoint: {sorted(missing)[:5]}')
            if any(p.dtype != torch.bfloat16 for p in module.parameters()):
                raise RuntimeError(f'{prefix} contains parameters outside the checked bfloat16 precision')
            module.eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True,
                                                       local_files_only=True)
        self.device = device

    @torch.inference_mode()
    def encode(self, images):
        pixels = self.processor.image_processor(images, return_tensors='pt')['pixel_values']
        pixels = pixels.to(self.device, dtype=torch.bfloat16)
        return self.projector(self.backbone(pixels)).float().cpu().numpy()
