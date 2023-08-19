from typing import Dict, List
import loralib as lora
import torch.nn as nn
import torch
import copy


def make_layer_lorable(layer: nn.Module, **lora_config):
    if isinstance(layer, nn.Linear):
        lora_layer = lora.Linear(
            layer.in_features, layer.out_features, **lora_config)
    elif isinstance(layer, nn.Conv2d):
        if len(layer.kernel_size) != 1:
            raise TypeError('The convolution kernel of Conv2d must be square.')
        lora_layer = lora.Conv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size[0], **lora_config)
    elif isinstance(layer, nn.Embedding):
        lora_layer = lora.Embedding(
            layer.num_embeddings, layer.embedding_dim, **lora_config)
    else:
        raise ValueError(
            'LoRA only supports nn.Linear, nn.Conv2d, nn.Embedding.')
    return lora_layer


def make_lora_model_by_type(model: nn.Module, replace: List[str], lora_bias: str, **lora_replace_config) -> nn.Module:
    model = copy.deepcopy(model)

    need_replace_module = []
    if 'linear' in replace:
        need_replace_module.append(nn.Linear)
    if 'conv2d' in replace:
        need_replace_module.append(nn.Conv2d)
    if 'embedding' in replace:
        need_replace_module.append(nn.Embedding)
    if len(need_replace_module) == 0:
        return model
    need_replace_module = tuple(need_replace_module)

    # replace
    for name, submodel in model.named_modules():
        if isinstance(submodel, need_replace_module):
            setattr(model, name, make_layer_lorable(
                submodel, **lora_replace_config))

    # only make lora layer trainable
    lora.mark_only_lora_as_trainable(model, lora_bias)

    return model


def make_lora_model_by_name(model: nn.Module, name_contains: List[str],  lora_bias: str, **lora_replace_config) -> nn.Module:
    model = copy.deepcopy(model)

    for name, submodel in model.named_modules():
        for need_replace_name in name_contains:
            if name.find(need_replace_name) != - 1:
                setattr(model, name, make_layer_lorable(
                    submodel, **lora_replace_config))
                break

    return model


def save_lora_checkpoint(model: nn.Module, checkpoint_filepath: str, lora_bias: str):
    lora_dict = lora.lora_state_dict(model, lora_bias)
    saved_dict = {
        'weights': lora_dict
    }
    torch.save(saved_dict, checkpoint_filepath)


def load_lora_checkpoint(model: nn.Module, pretrain_dict: Dict, lora_dict: Dict):
    model.load_state_dict(pretrain_dict, strict=False)
    model.load_state_dict(lora_dict, strict=False)
