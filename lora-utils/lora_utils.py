from typing import Any, Dict, List
import loralib as lora
import torch.nn as nn
import torch
import copy


class LoRALayerType(lora.LoRALayer, nn.Module):
    pass


def _get_other_conv_params(layer: nn.Module) -> Dict[str, Any]:
    """Get the parameters of the convolutional layer except in_channels, out_channels, kernel_size.

    Args:
        layer (nn.Module): convolutional layer.

    Returns:
        Dict[str, Any]: other parameters of the convolutional layer.
    """
    assert isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    res = {
        'stride': layer.stride,
        'padding': layer.padding,
        'dilation': layer.dilation,
        'groups': layer.groups,
        'bias': layer.bias is not None,
        'padding_mode': layer.padding_mode
    }
    return res


def _get_other_embedding_params(layer: nn.Embedding) -> Dict[str, Any]:
    assert isinstance(layer, nn.Embedding)
    res = {
        'padding_idx': layer.padding_idx,
        'max_norm': layer.max_norm,
        'norm_type': layer.norm_type,
        'scale_grad_by_freq': layer.scale_grad_by_freq,
        'sparse': layer.sparse,
        '_weight': layer.weight,
        '_freeze': not layer.weight.requires_grad,
    }
    return res


def _set_module(model: nn.Module, submodule_name: str, new_module: nn.Module):
    """Replaces the sub-model with the specified name in the model.

    Args:
        model (nn.Module): model to be replaced.
        submodule_name (str): submodel name.
        new_module (nn.Module): new sub-model.
    """    
    module_name_path = submodule_name.split('.')
    submodule_names = module_name_path[:-1]
    current_model = model
    for name in submodule_names:
        current_model = getattr(current_model, name)
    setattr(current_model, module_name_path[-1], new_module)


def make_layer_lorable(layer: nn.Module, **lora_config) -> LoRALayerType:
    """Returns the layer to which the LoRA is applied according to the type of layer passed in.

    Args:
        layer (nn.Module): layer that require LoRA to be applied.

    Returns:
        LoRALayer
    """    
    if isinstance(layer, nn.Linear):
        lora_layer = lora.Linear(layer.in_features, layer.out_features, **lora_config)
    elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if len(set(layer.kernel_size)) != 1:
            raise TypeError('All dimensions of the convolution kernel of Conv must be the same.')
        other_conv_params = _get_other_conv_params(layer)
        if isinstance(layer, nn.Conv1d):            
            lora_layer = lora.Conv1d(layer.in_channels, layer.out_channels, layer.kernel_size[0], **lora_config, **other_conv_params)
        elif isinstance(layer, nn.Conv2d):
            lora_layer = lora.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size[0], **lora_config, **other_conv_params)
        elif isinstance(layer, nn.Conv3d):
            lora_layer = lora.Conv3d(layer.in_channels, layer.out_channels, layer.kernel_size[0], **lora_config, **other_conv_params)
    elif isinstance(layer, nn.Embedding):
        other_embedding_params = _get_other_embedding_params(layer)
        lora_layer = lora.Embedding(layer.num_embeddings, layer.embedding_dim, **lora_config, **other_embedding_params)
    else:
        raise ValueError('LoRA only supports nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d and nn.Embedding.')
    return lora_layer


def make_lora_model_by_type(model: nn.Module, types: List[str], **lora_config) -> nn.Module:
    """Replaces the corresponding layer in the model with the corresponding layer in loralib based on the type passed in.

    Args:
        model (nn.Module): model to be replaced.
        types (List[str]): types of layer to be replaced.
        lora_bias (str): lora bias

    Returns:
        nn.Module: model after sub-model replacement.
    """
    model = copy.deepcopy(model)

    replace_module_type = []
    if 'linear' in types:
        replace_module_type.append(nn.Linear)
    if 'conv1d' in types:
        replace_module_type.append(nn.Conv2d)
    if 'conv2d' in types:
        replace_module_type.append(nn.Conv2d)
    if 'conv3d' in types:
        replace_module_type.append(nn.Conv2d)
    if 'embedding' in types:
        replace_module_type.append(nn.Embedding)
    if len(replace_module_type) == 0:
        return model
    replace_module_type = tuple(replace_module_type)

    # replace
    name2submodel = dict(model.named_modules())
    submodels_name = list(name2submodel.keys())
    for name in submodels_name:
        if isinstance(name2submodel[name], replace_module_type):
            _set_module(model, name, make_layer_lorable(name2submodel[name], **lora_config))

    return model


def make_lora_model_by_name(model: nn.Module, name_contains: List[str], **lora_config) -> nn.Module:
    """Replaces the corresponding layer in the model with the corresponding layer in loralib based on the type passed in.

    Args:
        model (nn.Module): model to be replaced.
        name_contains (List[str]): names of layer to be replaced.
        lora_bias (str): lora bias.

    Returns:
        nn.Module: model after sub-model replacement.
    """
    model = copy.deepcopy(model)

    name2submodel = dict(model.named_modules())
    submodels_name = list(name2submodel.keys())
    for name in submodels_name:
        for need_replace_name in name_contains:
            if name.find(need_replace_name) != - 1:
                _set_module(model, name, make_layer_lorable(name2submodel[name], **lora_config))
                break

    return model


def save_lora_weights(model: nn.Module, checkpoint_filepath: str, lora_bias: str):
    lora_dict = lora.lora_state_dict(model, lora_bias)
    torch.save(lora_dict, checkpoint_filepath)


def load_checkpoint_with_lora(model: nn.Module, pretrain_dict: Dict, lora_dict: Dict):
    model.load_state_dict(pretrain_dict, strict=False)
    model.load_state_dict(lora_dict, strict=False)
