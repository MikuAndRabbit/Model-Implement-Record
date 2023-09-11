from torch import nn
from typing import Any, Callable, Optional, Union
from torch import Tensor, nn
from torch.nn import functional as F
from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None, layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False,) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # encoder
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, norm_first, activation)
            encoder_norm = nn.LayerNorm(d_model, eps = layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, norm_first, activation)
            decoder_norm = nn.LayerNorm(d_model, eps = layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)


    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
                src_is_causal: bool = False, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        """Transformer

        Args:
            src (Tensor): The sequence to the encoder.
            tgt (Tensor): The sequence to the decoder.
            src_mask (Optional[Tensor], optional): The additive mask for the src sequence. Defaults to None.
            tgt_mask (Optional[Tensor], optional): The additive mask for the tgt sequence. Defaults to None.
            memory_mask (Optional[Tensor], optional): The additive mask for the encoder output. Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): The key padding mask for the src sequence. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): The key padding mask for the tgt sequence. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): The key padding mask for the memory sequence. Defaults to None.
            src_is_causal (bool, optional): If specified to True, applies a causal mask as src mask. Defaults to False.
            tgt_is_causal (bool, optional): If specified to True, applies a causal mask as tgt mask. Defaults to False.
            memory_is_causal (bool, optional): If specified to True, applies a causal mask as memory mask. Defaults to False.

        Returns:
            Tensor
        """

        # batch check
        is_batched = src.dim() == 3
        if src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # encode & decode
        memory = self.encoder(src, src_mask = src_mask, src_key_padding_mask = src_key_padding_mask, is_causal = src_is_causal)
        output = self.decoder(tgt, memory, tgt_mask = tgt_mask, memory_mask = memory_mask, 
                              tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask, 
                              tgt_is_causal = tgt_is_causal, memory_is_causal = memory_is_causal)

        return output
