import torch
from torch import nn


class Cross_TransFormer(nn.Module):
    def __init__(self,
                config):
        super(Cross_TransFormer, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                        nhead=self.num_attention_heads,
                                                        dim_feedforward=self.intermediate_size,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_hidden_layers)

    def forward(self, input_embeddings, input_mask = None):
        if input_mask is not None:
            input_mask = ~input_mask
            out = self.encoder(input_embeddings, input_mask)
        else:
            out = self.encoder(input_embeddings)

        return out


class ModelTypeEmbedding(nn.Module):
    r"""
    model_type: num of model types
    hidden_size: d_model

    """
    def __init__(self,
                model_type,
                hidden_size):
        super(ModelTypeEmbedding, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size

        self.model_type_embedding = nn.Embedding(num_embeddings=self.model_type,
                                                 embedding_dim=self.hidden_size)

    def forward(self, model_type_ids, input_embeddings):
        output = input_embeddings.clone()
        model_type_embeddings = self.model_type_embedding(model_type_ids)
        output = output + model_type_embeddings
        return output


class TV_Mask(nn.Module):
    def __init__(self, num_attention_heads):
        super(TV_Mask, self).__init__()
        self.num_attention_heads = num_attention_heads


    def forward(self, text_mask, image_mask):
        text_seq = text_mask.shape[1]
        image_seq = image_mask.shape[1]
        total_seq = text_seq + image_seq

        cross_mask = torch.cat((text_mask, image_mask), dim=1)
        # [batch_size, 1, sequence_length] ----> [batch_size*num_heads, sequence_length, sequence_length]
        cross_mask = torch.unsqueeze(cross_mask, dim=1).repeat(self.num_attention_heads, total_seq, 1)

        if cross_mask.dtype != torch.bool:
            cross_mask =cross_mask.bool()

        return cross_mask
