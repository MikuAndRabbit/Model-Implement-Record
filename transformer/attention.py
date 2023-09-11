from typing import Optional
from torch import nn
from utils import key_padding_mask_add_attn_mask
from torch import Tensor
import warnings


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.attention = nn.functional.scaled_dot_product_attention

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False):
        """Attention

        Args:
            query (Tensor): The Query. The shape `(N, ..., L, E)`.
            key (Tensor): The Key. The shape `(N, ..., S, E)`.
            value (Tensor): The Value. The shape `(N, ..., S, Ev)`.
            attn_mask (Optional[Tensor], optional): The attention mask. The shape `(N, ..., L, S)`.
                Two types of masks are supported. 
                A boolean mask where a value of True indicates that the element should take part in attention. 
                A float mask of the same type as query, key, value that is added to the attention score. 
                Defaults to None.
            dropout_p (float, optional): The probability of dropout. Defaults to 0.0.
            is_causal (bool, optional): Whether to use causal mask. Defaults to False.

        Returns:
            Tensor
        
        Shape legend:
            * N: Batch size (optional)
            * S: Source sequence length
            * L: Target sequence length
            * E: Embedding dimension of the query and key
            * Ev: Embedding dimension of the value
        """
        if not is_causal:
            return self.attention(query, key, value, attn_mask, dropout_p, is_causal)
        else:
            if attn_mask is not None:
                warnings.warn(f'attn_mask is not None. it will be ignored!', Warning)
            return self.attention(query, key, value, dropout_p = dropout_p, is_causal = is_causal)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, heads_num: int, kdim: Optional[int] = None, vdim: Optional[int] = None, bias: bool = False, dropout: float = .0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % heads_num == 0, 'embed_dim must be divisible by heads_num.'
        self.embed_dim = embed_dim
        self.n_head = heads_num
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.bias = bias
        self.dropout = dropout
        
        # linear
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias = bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias = bias)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias = bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        
        # attention
        self.attention = ScaleDotProductAttention()


    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, is_causal: bool = False, dropout: float = .0):
        """Multi-head Attention

        Args:
            query (Tensor): The Query. The shape `(N, H, L, E)`.
            key (Tensor): The Key. The shape `(N, H, S, E)`.
            value (Tensor): The Value. The shape `(N, H, S, Ev)`.
            attn_mask (Optional[Tensor], optional): The attention mask. The shape `(N, H, L, S)` or `(L, S)`. 
                Two types of masks are supported. 
                A boolean mask where a value of True indicates that the element should take part in attention. 
                A float mask of the same type as query, key, value that is added to the attention score. 
                Defaults to None.
            key_padding_mask (Optional[Tensor], optional): The key padding mask. The shape `(N, H, 1, S)` or `(N, H, L, S)` or `(N, S)`.
                Two types of masks are supported. If both `attn_mask` and `key_padding_mask` are provided, both should be of the same shape.
                For boolean mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention. 
                For a float mask, it will be directly added to the corresponding key value.
                Defaults to None.
            dropout (float, optional): The probability of dropout. Defaults to 0.0.
            is_causal (bool, optional): Whether to use causal mask. If set to True, it will ignore attn_mask. Defaults to False.

        Returns:
            Tensor
        
        Shape legend:
            * N: Batch size (optional)
            * H: The number of head
            * S: Source sequence length
            * L: Target sequence length
            * E: Embedding dimension of the query and key
            * Ev: Embedding dimension of the value
        """
        dropout = self.dropout if dropout is None else dropout
        
        # 1. project
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # 2. split tensor by number of heads
        query, key, value = self.split4heads(query), self.split4heads(key), self.split4heads(value)

        # 3. attention
        if is_causal and attn_mask is not None:
            raise ValueError('Explicit attn_mask should not be set when is_causal=True')
        if key_padding_mask is not None:
            query_dim = len(query.shape)
            if query_dim == 4:
                _, _, query_seq_len, _ = query.size()
            elif query_dim == 3:
                _, query_seq_len, _ = query.size()
            else:
                raise ValueError('The shape of query must be [n_heads, query_seq_len, query_dim] or [batch_size, n_heads, query_seq_len, query_dim].')
            fusion_mask = key_padding_mask_add_attn_mask(key_padding_mask, attn_mask, self.n_head, query_seq_len)
        else:
            fusion_mask = attn_mask
        out = self.attention(query, key, value, attn_mask = fusion_mask, dropout_p = dropout, is_causal = is_causal)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.out_proj(out)

        return out


    def split4heads(self, tensor: Tensor):
        """split tensor by number of head

        Args:
            tensor (Tensor): [batch_size, length, d_model]
        
        Returns:
            Tensor: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor


    def concat(self, tensor: Tensor):
        """inverse function of self.split(tensor : torch.Tensor)

        Args:
            tensor (Tensor): [batch_size, head, length, d_tensor]
        
        Returns:
            Tensor: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
