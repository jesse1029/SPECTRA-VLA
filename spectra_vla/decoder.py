"""Frozen OpenVLA first-position action logits from supplied visual tokens."""
from __future__ import annotations

import json
from pathlib import Path

import torch


class FrozenOpenVLADecoder:
    def __init__(self, model_dir: str | Path, cpu_layers: int = 0):
        from accelerate import cpu_offload, init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("the reference BF16 decoder requires CUDA")
        model_dir = Path(model_dir).resolve()
        config = json.loads((model_dir / "config.json").read_text())
        llama_config = LlamaConfig(**config["text_config"])
        llama_config._attn_implementation = "sdpa"
        count = llama_config.num_hidden_layers
        if not 0 <= cpu_layers <= count:
            raise ValueError("cpu_layers is outside the decoder layer count")
        self.offloaded_layers = list(range(count - cpu_layers, count))
        with init_empty_weights():
            self.model = LlamaForCausalLM(llama_config)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        assigned = set()
        for file in sorted(set(index.values())):
            shard = (model_dir / file).resolve()
            if not shard.is_relative_to(model_dir):
                raise ValueError("checkpoint shard path escapes model directory")
            with safe_open(str(shard), framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key.startswith("language_model."):
                        name = key[len("language_model."):]
                        layer = int(name.split(".")[2]) if name.startswith("model.layers.") else -1
                        destination = "cpu" if layer in self.offloaded_layers else "cuda"
                        set_module_tensor_to_device(self.model, name, destination,
                                                    value=tensors.get_tensor(key), dtype=torch.bfloat16)
                        assigned.add(name)
        if set(dict(self.model.named_parameters())) != assigned:
            raise ValueError("decoder checkpoint parameter inventory does not match")
        if any(parameter.is_meta for parameter in self.model.parameters()):
            raise ValueError("decoder contains unloaded parameters")
        for name, buffer in self.model.named_buffers():
            if not buffer.is_meta:
                set_module_tensor_to_device(self.model, name, "cuda", value=buffer)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for layer in self.offloaded_layers:
            cpu_offload(self.model.model.layers[layer], execution_device=torch.device("cuda"), offload_buffers=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.vocab_size = llama_config.vocab_size - config.get("pad_to_multiple_of", 64)
        self.action_ids = torch.arange(self.vocab_size - 1, self.vocab_size - 257, -1, device="cuda")
        prompt = "In: What action should the robot take to sort the object into the correct bin?\nOut:"
        self.prompt_ids = torch.tensor(self.tokenizer(prompt, add_special_tokens=True).input_ids + [29871], device="cuda")[None]
        self.embedding = self.model.get_input_embeddings()(self.prompt_ids)

    @torch.inference_mode()
    def forward(self, visual):
        if visual.ndim != 3 or visual.shape[-1] != self.embedding.shape[-1]:
            raise ValueError("visual tokens must have shape (batch, patches, decoder width)")
        emb = self.embedding.expand(len(visual), -1, -1)
        mm = torch.cat([emb[:, :1], visual.to(device=emb.device, dtype=emb.dtype), emb[:, 1:]], dim=1)
        attention = torch.ones(mm.shape[:2], dtype=torch.long, device="cuda")
        out = self.model(inputs_embeds=mm, attention_mask=attention, use_cache=False)
        return out.logits[:, -1].index_select(-1, self.action_ids).float().cpu().numpy()
