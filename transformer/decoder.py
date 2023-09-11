from typing import Callable, Optional, Union
from torch import Tensor, nn
from torch.nn import functional as F
from attention import MultiHeadAttention
from utils import _get_clones, _get_activation_fn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, drop_prob: float = 0.1, layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.norm_first = norm_first
        self.activation = activation
        
        # self & cross attention
        self.self_attn = MultiHeadAttention(embed_dim = d_model, heads_num = nhead, dropout = drop_prob)
        self.multihead_attn = MultiHeadAttention(embed_dim = d_model, heads_num = nhead, dropout = drop_prob)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt (Tensor): The sequence to the decoder layer.
            memory (Tensor): The sequence from the last layer of the encoder.
            tgt_mask (Optional[Tensor], optional): The mask for the tgt sequence. Defaults to None.
            memory_mask (Optional[Tensor], optional): The mask for the memory sequence. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): The key padding mask for the memory sequence. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): The key padding mask for the tgt sequence. Defaults to None.
            tgt_is_causal (bool, optional): If specified to True, applies a causal mask as tgt mask. Defaults to False.
            memory_is_causal (bool, optional): If specified to True, applies a causal mask as tgt mask. Defaults to False.
        
        Notice:
            The meaning of *_attn_mask or *_key_padding_mask can be found in attention.py.
        
        Returns:
            Tensor
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask = attn_mask, key_padding_mask = key_padding_mask, is_causal = is_causal)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem, attn_mask = attn_mask, key_padding_mask = key_padding_mask, is_causal = is_causal)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt (Tensor): The sequence to the decoder.
            memory (Tensor): The sequence from the last layer of the encoder.
            tgt_mask (Optional[Tensor], optional): The mask for the tgt sequence. Defaults to None.
            memory_mask (Optional[Tensor], optional): The mask for the memory sequence. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): The key padding mask for the memory sequence. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): The key padding mask for the tgt sequence. Defaults to None.
            tgt_is_causal (bool, optional): If specified to True, applies a causal mask as tgt mask. Defaults to False.
            memory_is_causal (bool, optional): If specified to True, applies a causal mask as tgt mask. Defaults to False.
        
        Notice:
            The meaning of *_attn_mask or *_key_padding_mask can be found in attention.py.

        Returns:
            Tensor
        """
        output = tgt

        for decoder_layer in self.layers:
            output = decoder_layer(output, memory, tgt_mask = tgt_mask, memory_mask = memory_mask, 
                                   tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask, 
                                   tgt_is_causal = tgt_is_causal, memory_is_causal = memory_is_causal)
        if self.norm is not None:
            output = self.norm(output)

        return output
