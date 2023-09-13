from typing import Dict, List
import loralib as lora
import torch.nn as nn
import torch
import copy


def make_layer_lorable(layer: nn.Module, **lora_config):
    if isinstance(layer, nn.Linear):
        lora_layer = lora.Linear(layer.in_features, layer.out_features, **lora_config)
    elif isinstance(layer, nn.Conv2d):
        if len(layer.kernel_size) != 1:
            raise TypeError('The convolution kernel of Conv2d must be square.')
        lora_layer = lora.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size[0], **lora_config)
    elif isinstance(layer, nn.Embedding):
        lora_layer = lora.Embedding(layer.num_embeddings, layer.embedding_dim, **lora_config)
    else:
        raise ValueError('LoRA only supports nn.Linear, nn.Conv2d, nn.Embedding.')
    return lora_layer


def _set_module(model: nn.Module, submodule_name: str, new_module: nn.Module):
    module_name_path = submodule_name.split('.')
    submodule_names = module_name_path[:-1]
    current_model = model
    for name in submodule_names:
        current_model = getattr(current_model, name)
    setattr(current_model, module_name_path[-1], new_module)


def make_lora_model_by_type(model: nn.Module, replace: List[str], lora_bias: str, **lora_replace_config) -> nn.Module:
    model = copy.deepcopy(model)

    replace_module_type = []
    if 'linear' in replace:
        replace_module_type.append(nn.Linear)
    if 'conv2d' in replace:
        replace_module_type.append(nn.Conv2d)
    if 'embedding' in replace:
        replace_module_type.append(nn.Embedding)
    if len(replace_module_type) == 0:
        return model
    replace_module_type = tuple(replace_module_type)

    # replace
    name2submodel = dict(model.named_modules())
    submodels_name = list(name2submodel.keys())
    for name in submodels_name:
        if isinstance(name2submodel[name], replace_module_type):
            _set_module(model, name, make_layer_lorable(name2submodel[name], **lora_replace_config))

    # only make lora layer trainable
    lora.mark_only_lora_as_trainable(model, lora_bias)

    return model


def make_lora_model_by_name(model: nn.Module, name_contains: List[str], **lora_replace_config) -> nn.Module:
    model = copy.deepcopy(model)

    name2submodel = dict(model.named_modules())
    submodels_name = list(name2submodel.keys())
    for name in submodels_name:
        for need_replace_name in name_contains:
            if name.find(need_replace_name) != - 1:
                _set_module(model, name, make_layer_lorable(name2submodel[name], **lora_replace_config))
                break

    return model


def save_lora_weights(model: nn.Module, checkpoint_filepath: str, lora_bias: str):
    lora_dict = lora.lora_state_dict(model, lora_bias)
    torch.save(lora_dict, checkpoint_filepath)


def load_checkpoint_with_lora(model: nn.Module, pretrain_dict: Dict, lora_dict: Dict):
    model.load_state_dict(pretrain_dict, strict=False)
    model.load_state_dict(lora_dict, strict=False)
