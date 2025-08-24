# peft_injector.py
from typing import List, Optional
import torch.nn as nn

from peft import LoraConfig, get_peft_model

def _default_targets(model: nn.Module, preset: str) -> List[str]:
    """
    Returns a list of module-name substrings or exact names to target.
    If preset == 'all_linear', we enumerate every nn.Linear by full name.
    """
    preset = preset.lower()

    if preset == "all_linear":
        names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names.append(name)      # exact names are fine
        return names

    if preset == "attn":
        return ["qkv_proj", "o_proj"]

    if preset == "attn_mlp":
        return [
            "qkv_proj", "o_proj",
            "input_ff_layer.hidden_layer.0",
            "input_ff_layer.output_layer",
            "horizon_ff_layer.hidden_layer.0",
            "horizon_ff_layer.output_layer",
        ]

    if preset == "attn_mlp_gating":
        return [
            "qkv_proj", "o_proj",
            "input_ff_layer.hidden_layer.0",
            "input_ff_layer.output_layer",
            "horizon_ff_layer.hidden_layer.0",
            "horizon_ff_layer.output_layer",
            "moe.gate.to_gates",
        ]

    if preset == "experts_heavy":
        return [
            "qkv_proj", "o_proj",
            "input_ff_layer.hidden_layer.0", "input_ff_layer.output_layer",
            "horizon_ff_layer.hidden_layer.0", "horizon_ff_layer.output_layer",
            "moe.gate.to_gates", "experts.experts", "gate_proj", "down_proj",
        ]

    raise ValueError(f"Unknown lora_targets_preset: {preset}")


def resolve_linear_targets(model: nn.Module, patterns: List[str]) -> List[str]:
    hits = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name for p in patterns) or name in patterns:
                hits.append(name)
    return hits


def _unfreeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def wrap_with_peft(model: nn.Module,
                   *,
                   peft_type: Optional[str],
                   lora_r: int,
                   lora_alpha: int,
                   lora_dropout: float,
                   lora_bias: str,
                   lora_init: Optional[str],
                   lora_targets_preset: str,
                   lora_extra_target_patterns: Optional[List[str]] = None,
                   train_base: bool = False,
                   task_type: str = "FEATURE_EXTRACTION"):
    """
    If peft_type in {"lora","dora"}: attach adapters, optionally unfreeze base.
    If train_base=True, all base params are set requires_grad=True after wrapping.
    """
    if peft_type is None:
        return model

    use_dora = (peft_type.lower() == "dora")

    patterns = _default_targets(model, lora_targets_preset)
    if lora_extra_target_patterns:
        patterns.extend(lora_extra_target_patterns)

    lora_kwargs = dict(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=patterns,
        use_dora=use_dora
    )
    if lora_init is not None:
        lora_kwargs["init_lora_weights"] = lora_init

    peft_cfg = LoraConfig(**lora_kwargs)
    peft_model = get_peft_model(model, peft_cfg)

    # Optional: print trainable summary
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    if train_base:
        # Unfreeze *all* parameters (base + adapters)
        _unfreeze_all_params(peft_model)

    return peft_model
