from typing import Callable, Optional, Union
from torch import Tensor, nn
from attention import MultiHeadAttention
from torch.nn import functional as F
from utils import _get_clones, _get_activation_fn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, drop_prob: float = 0.1, layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.activation = activation
        self.self_attn = MultiHeadAttention(embed_dim = d_model, heads_num = nhead, dropout = drop_prob)
        
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(p = drop_prob)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.dropout2 = nn.Dropout(p = drop_prob)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


    # self-attention block
    def _sa_block(self, x: Tensor, src_mask: Optional[Tensor], src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = None) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask = src_mask, key_padding_mask = src_key_padding_mask, is_causal = is_causal)
        return self.dropout1(x)


    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        """Transformer Encoder Layer

        Args:
            src (Tensor): the sequence to the encoder.
            src_mask (Optional[Tensor], optional): the mask for the src sequence. Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): the mask for the src keys per batch. Defaults to None.
            is_causal (bool, optional): If specified, applies a causal mask as mask (optional) and ignores attn_mask for computing scaled dot product attention. Defaults to False.
        
        Notice:
            The meaning of *_attn_mask or *_key_padding_mask can be found in attention.py.

        Returns:
            Tensor
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(x = self.norm1(x), src_mask = src_mask, src_key_padding_mask = src_key_padding_mask, is_causal = is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x = x, src_mask = src_mask, src_key_padding_mask = src_key_padding_mask, is_causal = is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int, norm_layer = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm_layer = norm_layer


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        """Transformer Encoder

        Args:
            src (Tensor): the sequence to the encoder.
            src_mask (Optional[Tensor], optional): the mask for the src sequence. Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): the mask for the src keys per batch. Defaults to None.
            is_causal (bool, optional): If specified, applies a causal mask as mask (optional) and ignores attn_mask for computing scaled dot product attention. Defaults to False.
        
        Notice:
            The meaning of *_attn_mask or *_key_padding_mask can be found in attention.py. 

        Returns:
            Tensor
        """
        output = src
        for model in self.layers:
            output = model(output, src_mask = src_mask, src_key_padding_mask = src_key_padding_mask, is_causal = is_causal)
        if self.norm_layer is not None:
            output = self.norm_layer(output)
        return output
