import os
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import ViTModel, ViTConfig, ViTForMaskedImageModeling
from model.TVA_Transformer import Cross_TransFormer
from model.Discrete_vae import DiscreteVAE

def vit_backbone(vit_config):
    vit_net = None
    visual_width = 0
    if vit_config.vit_name == "base":
        base = vit_config.vit_base
        vit_cfg = ViTConfig(hidden_size=base.hidden_size,
                            num_hidden_layers=base.num_hidden_layers,
                            num_attention_heads=base.num_attention_heads,
                            intermediate_size=base.intermediate_size,
                            image_size=vit_config.image_size,
                            patch_size=vit_config.patch_size)
        visual_width = base.hidden_size
        vit_net = ViTModel(vit_cfg)
        if vit_config.vit_base_ckpt:
            if vit_cfg.image_size == 224 and vit_cfg.patch_size == 16:
                vit_net = vit_net.from_pretrained("google/vit-base-patch16-224-in21k")
    #后续有其他配置的需求的时候，再把其他的预训练权重加进来

    if vit_config.vit_name == "large":
        large = vit_config.vit_large
        vit_cfg = ViTConfig(hidden_size=large.hidden_size,
                            num_hidden_layers=large.num_hidden_layers,
                            num_attention_heads=large.num_attention_heads,
                            intermediate_size=large.intermediate_size,
                            image_size=vit_config.image_size,
                            patch_size=vit_config.patch_size)
        visual_width = large.hidden_size
        vit_net = ViTModel(vit_cfg)
        if vit_config.vit_large_ckpt:
            if vit_cfg.image_size == 224 and vit_cfg.patch_size == 16:
                vit_net = vit_net.from_pretrained("google/vit-large-patch16-224-in21k")
        #后续有其他配置的需求的时候，再把其他的预训练权重加进来

    return vit_net, visual_width


def vit_mask_backbone(vit_config):
    vit_net = None
    visual_width = 0
    if vit_config.vit_name == "base":
        base = vit_config.vit_base
        vit_cfg = ViTConfig(hidden_size=base.hidden_size,
                            num_hidden_layers=base.num_hidden_layers,
                            num_attention_heads=base.num_attention_heads,
                            intermediate_size=base.intermediate_size,
                            image_size=vit_config.image_size,
                            patch_size=vit_config.patch_size)
        visual_width = base.hidden_size
        vit_net = ViTForMaskedImageModeling(vit_cfg)
        if vit_config.vit_base_ckpt:
            if vit_cfg.image_size == 224 and vit_cfg.patch_size == 16:
                vit_net = vit_net.from_pretrained("google/vit-base-patch16-224-in21k")
    #后续有其他配置的需求的时候，再把其他的预训练权重加进来

    if vit_config.vit_name == "large":
        large = vit_config.vit_large
        vit_cfg = ViTConfig(hidden_size=large.hidden_size,
                            num_hidden_layers=large.num_hidden_layers,
                            num_attention_heads=large.num_attention_heads,
                            intermediate_size=large.intermediate_size,
                            image_size=vit_config.image_size,
                            patch_size=vit_config.patch_size)
        visual_width = large.hidden_size
        vit_net = ViTForMaskedImageModeling(vit_cfg)
        if vit_config.vit_large_ckpt:
            if vit_cfg.image_size == 224 and vit_cfg.patch_size == 16:
                vit_net = vit_net.from_pretrained("google/vit-large-patch16-224-in21k")
        #后续有其他配置的需求的时候，再把其他的预训练权重加进来

    return vit_net, visual_width


def bert_backbone(bert_config):
    bert_net = None
    text_width = 0

    if bert_config.bert_name == "base":
        base = bert_config.bert_base
        bert_cfg = BertConfig(hidden_size=base.hidden_size,
                              num_hidden_layers=base.num_hidden_layers,
                              num_attention_heads=base.num_attention_heads,
                              intermediate_size=base.intermediate_size,
                              vocab_size=bert_config.vocab_size)
        text_width = base.hidden_size
        bert_net = BertModel(bert_cfg)
        if bert_config.bert_base_ckpt:
            bert_net = bert_net.from_pretrained("bert-base-uncased")

    if bert_config.bert_name == "large":
        large = bert_config.bert_large
        bert_cfg = BertConfig(hidden_size=large.hidden_size,
                              num_hidden_layers=large.num_hidden_layers,
                              num_attention_heads=large.num_attention_heads,
                              intermediate_size=large.intermediate_size,
                              vocab_size=bert_config.vocab_size)
        text_width = large.hidden_size
        bert_net = BertModel(bert_cfg)
        if bert_config.bert_large_ckpt:
            bert_net = bert_net.from_pretrained("bert-large-uncased")

    return bert_net, text_width


def cross_backbone(cross_cfg):
    #Cross Encoder
    cross_net = None
    cross_width = None

    if cross_cfg.cross_name == "base":
        cross_net = Cross_TransFormer(cross_cfg.cross_base)
        cross_width = cross_cfg.cross_base.hidden_size
    if cross_cfg.cross_name == "large":
        cross_net = Cross_TransFormer(cross_cfg.cross_large)
        cross_width = cross_cfg.cross_large.hidden_size

    return cross_net,cross_width


def init_tokenizer():
    vocab_file = "/home/theone/TMEP_v2/vocab/en_vocab.txt"
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    return tokenizer


def dvae_backbone(dvae_config):
    dvae = DiscreteVAE(
        image_size=dvae_config.image_size,
        num_tokens=dvae_config.num_tokens,
        codebook_dim=dvae_config.codebook_dim,
        num_layers=dvae_config.num_layers,
        hidden_dim=dvae_config.hidden_dim
    )
    state_dict = torch.load(os.path.join(dvae_config.model_path, "dvae_base_weights.pth"), map_location="cpu")["weights"]

    dvae.load_state_dict(state_dict)

    dvae.requires_grad_(False)

    return dvae



