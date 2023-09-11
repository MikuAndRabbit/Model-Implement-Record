from typing import Callable, Optional
from torch import Tensor, nn
import copy
import torch
from torch.nn import functional as F

def _get_clones(module, N: int):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def generate_square_subsequent_mask(sz: int, device = 'cpu', dtype = torch.float) -> Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        res = torch.triu(torch.full((sz, sz), float('-inf'), device = device), diagonal = 1)
        if dtype == torch.bool:
            res = res == 0
        return res


def _paddingmask_add_attnmask_return(key_padding_mask: Tensor, attn_mask: Tensor):
    if key_padding_mask.dtype == torch.bool:
        return key_padding_mask & attn_mask
    else:
        return key_padding_mask + attn_mask


def key_padding_mask_add_attn_mask(key_padding_mask: Tensor, attn_mask: Optional[Tensor], explict_nheads: int = 1, query_seq_len: Optional[int] = None):
    """Used to merge key_padding_mask and attn_mask into a usable mask.

    Args:
        key_padding_mask (Tensor): Represents the padded mask, False means the position is not padded, True means the position is padded.
        attn_mask (Tensor): Attention mask. True means that the position participates in the attention calculation, and False means that the position does not participate in the attention calculation.
        explict_nheads (int, optional): When key_padding_mask_dim is 2 and attn_mask_dim is 2, explict_nheads will be used as the value of n_heads of the expanded mask.. Defaults to 1.
        query_seq_len (int, optional): When attn_mask is None, query_seq_len must be set to generate final fusion mask.
    
    Notice:
        The method only supports key_padding_mask and attn_mask in the following dimensions:
        * attn_mask
            * attn_mask_dim = 4: [batch_size, n_heads, query_seq_len, kv_seq_len]
            * attn_mask_dim = 2: [query_seq_len, kv_seq_len]
            * None
        * key_padding_mask
            * key_padding_mask_dim = 4: [batch_size, n_heads, 1 / query_seq_len, kv_seq_len]
            * key_padding_mask_dim = 2: [batch_size, kv_seq_len]

    Returns:
        Tensor: [batch_size, n_heads, query_seq_len, kv_seq_len]
    """
    
    # when attn_mask is none, convert key_padding_mask to expected shape
    if attn_mask is None:
        if query_seq_len is None:
            raise ValueError('When only key_padding_mask is provided, you must specify query_seq_len.')
        
        key_padding_mask_dim = len(key_padding_mask.shape)
        if key_padding_mask_dim == 2:
            batch_size, kv_seq_len = key_padding_mask.size()
            key_padding_mask_expand = key_padding_mask.view(batch_size, 1, 1, kv_seq_len).expand(-1, explict_nheads, query_seq_len, -1)
            return key_padding_mask_expand
        if key_padding_mask_dim == 4:
            return key_padding_mask
        raise ValueError('Only cases with key_padding_mask_dim of 2 and 4 are supported.')
    
    if attn_mask.size(-1) != key_padding_mask.size(-1):
        raise ValueError('The last one of attn_mask and key_padding_mask should be key_valeu_seq_len, that is, they should be the same.')
    if attn_mask.dtype != key_padding_mask.dtype:
        raise ValueError('The type of attn_mask and key_padding_mask must be matched.')
    
    if key_padding_mask.dtype == torch.bool:
        key_padding_mask = ~key_padding_mask

    attn_mask_dim = len(attn_mask.shape)
    key_padding_mask_dim = len(key_padding_mask.shape)
    
    if attn_mask_dim == 4:
        batch_size, n_heads, query_seq_len, kv_seq_len = attn_mask.size()
        if key_padding_mask_dim == 4:
            return _paddingmask_add_attnmask_return(key_padding_mask, attn_mask)
        elif key_padding_mask_dim == 2:
            if key_padding_mask.size() != (batch_size, kv_seq_len):
                key_padding_mask_size = key_padding_mask.size()
                raise ValueError(f'The shape of key_padding_mask must math with attn_mask, it shoule be [{batch_size}, {kv_seq_len}], now get [{key_padding_mask_size[0]}, {key_padding_mask_size[1]}]')
            key_padding_mask_expand = key_padding_mask.view(batch_size, 1, 1, kv_seq_len).expand(-1, n_heads, -1, -1)
            return _paddingmask_add_attnmask_return(key_padding_mask_expand, attn_mask)
    elif attn_mask_dim == 2:
        query_seq_len, kv_seq_len = attn_mask.size()
        if key_padding_mask_dim == 4:
            batch_size, n_heads, x, kv_seq_len = key_padding_mask.size()
            if x != query_seq_len and x != 1:
                raise ValueError(f'The shape of key_padding_mask must math with attn_mask, it shoule be [{batch_size}, {n_heads}, 1, {kv_seq_len}] or [{batch_size}, {n_heads}, {query_seq_len}, {kv_seq_len}], now get [{batch_size}, {n_heads}, {x}, {kv_seq_len}].')
            attn_mask_expand = attn_mask.view(1, 1, query_seq_len, kv_seq_len).expand(batch_size, n_heads, -1, -1)
            return _paddingmask_add_attnmask_return(key_padding_mask, attn_mask_expand)
        elif key_padding_mask_dim == 2:
            batch_size, kv_seq_len = key_padding_mask.size()
            attn_mask_expand = attn_mask.view(1, 1, query_seq_len, kv_seq_len).expand(batch_size, explict_nheads, -1, -1)
            key_padding_mask_expand = key_padding_mask.view(batch_size, 1, 1, kv_seq_len).expand(-1, explict_nheads, -1, -1)
            return _paddingmask_add_attnmask_return(key_padding_mask_expand, attn_mask_expand)
    
    raise ValueError('The shape of key_padding_mask must math with attn_mask')
