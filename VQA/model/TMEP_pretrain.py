import os
import torch
from torch import nn
import torch.nn.functional as F
from model.TMEP import bert_backbone, \
    vit_backbone, \
    cross_backbone, \
    init_tokenizer, \
    dvae_backbone, \
    vit_mask_backbone
from model.TVA_Transformer import ModelTypeEmbedding, TV_Mask
from torch.cuda.amp import autocast
from config.lora import VIT_LORA_CONFIG, BERT_LORA_CONFIG
from peft import get_peft_model


class MultiEncoder(nn.Module):
    def __init__(self,
                cross_config):
        super().__init__()
        self.cross_config = cross_config
        self.cross_encoder, cross_width = cross_backbone(self.cross_config)
        self.cross_width = cross_width
        self.post_processer = ModelTypeEmbedding(self.cross_config.model_type,self.cross_width)
        self.num_heads = 12
        if self.cross_config.cross_name == "large":
            self.num_heads = self.cross_config.cross_large.num_attention_heads
        self.tv_mask = TV_Mask(self.num_heads)

    def forward(self, text_embeddings, image_embeddings, text_mask, image_mask):
        # [batch_size, image_sl + text_sl, hidden_size]
        input_embeddings = torch.cat((text_embeddings,image_embeddings),dim=1)

        text_type_ids = torch.zeros(text_embeddings.size()[:-1], dtype=torch.int).to(text_embeddings.device)
        image_type_ids = torch.zeros(image_embeddings.size()[:-1], dtype=torch.int).to(image_embeddings.device)

        model_type_ids = torch.cat((text_type_ids,image_type_ids),dim=1)

        input_embeddings = self.post_processer(model_type_ids, input_embeddings)
        input_mask = self.tv_mask(text_mask, image_mask)

        cross_encoder_output = self.cross_encoder(input_embeddings, input_mask)

        text_seq = text_embeddings.shape[1]
        img_seq = image_embeddings.shape[1]


        # text_crossout, image_crossout = torch.split(cross_encoder_output,text_seq,dim=1)
        out = torch.split(cross_encoder_output,[text_seq, img_seq],dim=1)
        text_crossout = out[0]
        image_crossout = out[1]

        return text_crossout, image_crossout


class MLMHead(nn.Module):
    def __init__(self,
                hidden_size,
                vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)

        pooler_in = self.gelu(hidden_states)
        pooler_in = self.layernorm(pooler_in)

        prediction_scores = self.decoder(pooler_in)

        return prediction_scores


class MLMLoss(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, prediction_scores, labels):
        prediction_logits = torch.reshape(prediction_scores, (-1, self.vocab_size))
        labels = torch.reshape(labels, (-1, ))

        masked_lm_loss = self.loss_fn(prediction_logits, labels)

        return masked_lm_loss


class MIMHead(nn.Module):
    def __init__(self,
                 hidden_size,
                 image_vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_vocab_size = image_vocab_size
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.image_vocab_size)

    def forward(self, image_hidden_states):
        image_without_cls = image_hidden_states[:, 1:]
        inputs = self.layernorm(image_without_cls)
        inputs = self.decoder(inputs)
        return inputs


class MIMLoss(nn.Module):
    def __init__(self,image_vocab_size):
        super().__init__()
        self.image_vocab_size = image_vocab_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        prediction_logits = torch.reshape(inputs, (-1, self.image_vocab_size))
        labels = torch.reshape(targets, (-1,))

        masked_img_loss = self.loss_fn(prediction_logits, labels)

        return masked_img_loss


class SimilarForTextImage(nn.Module):
    r"""
        依据文字表示和图片表示直接计算两个模态的相似度，方法就是矩阵相乘
        我们根据 不同模态的cls 来在整体语义上计算相似

           输入：文本的单模态编码 图片的单模态编码
               text_embeddings, image_embeddings

           输出：文字对应batch内所有图片的相似度, 图片对应batch内所有文字的相似度
               sim_t2i,sim_i2t
    """
    def __init__(self,
                text_width,
                image_width):
        super().__init__()
        self.text_width = text_width
        self.image_width = image_width

        self.text_dense = nn.Linear(self.text_width, self.text_width)
        self.image_dense = nn.Linear(self.image_width, self.image_width)

    def forward(self, text_embeddings, image_embeddings):
        text_cls = text_embeddings[:, :1, :]
        image_cls = image_embeddings[:, :1, :]

        # 特征计算，先过全连接层再归一化
        text_feats = self.text_dense(text_cls.squeeze())
        image_feats = self.image_dense(image_cls.squeeze())

        with torch.no_grad():
            # [batch_size, hidden_size]
            # 防止过拟合
            text_feats = F.normalize(text_feats)
            image_feats = F.normalize(image_feats)

            # 每一个文字对应图片的相似度
            # [batch_size, batch_size]
            sim_t2i = text_feats @ image_feats.T
            sim_t2i = F.softmax(sim_t2i, dim=1)
            # 只取负样本，所以主对角线的元素要设为0
            sim_t2i.fill_diagonal_(0)

            # 每一张图片对应Batch中的负样本的文字的相似度
            sim_i2t = image_feats @ text_feats.T
            sim_i2t = F.softmax(sim_i2t, dim=1)
            sim_i2t.fill_diagonal_(0)

        return sim_t2i, sim_i2t


class NegDataCollector(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, text_embeddings, text_attns, image_embeddings, image_attns, sim_t2i, sim_i2t):
        text_embed_neg = []
        text_attns_neg = []
        image_embed_neg = []
        image_attns_neg = []

        batch_size = text_embeddings.shape[0]
        for b in range(batch_size):
            neg_image_idx = torch.argmax(sim_t2i[b])
            image_embed_neg.append(image_embeddings[neg_image_idx])
            image_attns_neg.append(image_attns[neg_image_idx])
            neg_text_idx = torch.argmax(sim_i2t[b])
            text_embed_neg.append(text_embeddings[neg_text_idx])
            text_attns_neg.append(text_attns[neg_text_idx])

        text_embed_neg = torch.stack(text_embed_neg, dim=0)
        text_attns_neg = torch.stack(text_attns_neg, dim=0)
        image_embed_neg = torch.stack(image_embed_neg, dim=0)
        image_attns_neg = torch.stack(image_attns_neg, dim=0)

        text_embed_all = torch.cat((text_embeddings, text_embed_neg), dim=0)
        text_attns_all = torch.cat((text_attns, text_attns_neg), dim=0)
        image_embed_all = torch.cat((image_embeddings, image_embed_neg), dim=0)
        image_attns_all = torch.cat((image_attns, image_attns_neg), dim=0)

        return text_embed_all, text_attns_all, image_embed_all, image_attns_all


class ITMLoss(nn.Module):
    def __init__(self,
                cross_width):
        super().__init__()
        self.cross_width = cross_width
        self.dense = nn.Linear(self.cross_width, self.cross_width)
        self.tanh = nn.Tanh()
        self.itm_head = nn.Linear(self.cross_width, 2)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, positive_output, negative_output):
        positive_cls = positive_output[:, :1, :]
        negative_cls = negative_output[:, :1, :]

        vl_embedding = torch.cat((positive_cls, negative_cls),dim=0).squeeze()
        vl_embedding = self.dense(vl_embedding)
        vl_embedding = self.tanh(vl_embedding)
        itm_inputs = self.itm_head(vl_embedding)

        bs = positive_cls.shape[0]
        itm_lables = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],dim=0).to(positive_output.device)

        positive_prob = itm_inputs[:, :1]

        ret = {
            "positive_prob": positive_prob
        }

        ITM_loss = self.loss_fn(itm_inputs, itm_lables)

        return ITM_loss, ret


class TMEP_v1(nn.Module):
    def __init__(self,
                 bert_config,
                 vit_config,
                 cross_config,
                 frozen_single=True):
        super().__init__()
        self.bert_config = bert_config
        self.vit_config = vit_config
        self.cross_config = cross_config
        self.frozen_single = frozen_single

        self.tokenizer = init_tokenizer()
        self.mlm_probability = 0.15

        self.bert_net, self.text_width = bert_backbone(self.bert_config)
        self.visual_net, self.visual_width = vit_backbone(self.vit_config)
        self.cross_encoder = MultiEncoder(cross_config=self.cross_config)
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.text_vocab_size = self.bert_config.vocab_size
        if cross_config.cross_name == "large":
            self.num_attention_heads = self.cross_config.cross_large.num_attention_heads
            self.hidden_size = self.cross_config.cross_large.hidden_size

        self.tv_maks = TV_Mask(self.num_attention_heads)

        # MLM
        self.mlm_head = MLMHead(self.hidden_size, self.text_vocab_size)
        self.mlm_fn = MLMLoss(self.text_vocab_size)

        # ITM
        self.sim_ti = SimilarForTextImage(self.text_width, self.visual_width)
        self.neg_data = NegDataCollector()
        self.itm_fn = ITMLoss(self.hidden_size)

        #Uncertain Weigth
        self.sigma1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.sigma2 = nn.Parameter(torch.ones(1, requires_grad=True) * 10)

        if self.frozen_single:
            self.frozen_single_param()

    def save_pretrained(self, save_path):
        state_file = os.path.join(save_path, "tmep_model.pth")
        state = {
            "bert_net": self.bert_net.state_dict(),
            "visual_net": self.visual_net.state_dict(),
            "cross_encoder": self.cross_encoder.state_dict(),
            "mlm_head": self.mlm_head.state_dict(),
            "sim_ti": self.sim_ti.state_dict(),
            "itm_fn": self.itm_fn.state_dict()
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location=None):
        state_file = os.path.join(load_path, "tmep_model.pth")
        if not os.path.exists(state_file):
            return
        state = torch.load(state_file, map_location=map_location)
        self.bert_net.load_state_dict(state["bert_net"], strict=False)
        self.visual_net.load_state_dict(state["visual_net"], strict=False)
        self.cross_encoder.load_state_dict(state["cross_encoder"], strict=False)
        self.mlm_head.load_state_dict(state["mlm_head"], strict=False)
        self.sim_ti.load_state_dict(state["sim_ti"], strict=False)
        self.itm_fn.load_state_dict(state["itm_fn"], strict=False)

    def uncertain_weight(self, loss, sigma):
        return 0.5 / (sigma ** 2) * loss + torch.log(1 + sigma ** 2)

    @autocast()
    def forward(self, caption, image):
        image_embeddings = self.visual_net(image, return_dict=True).last_hidden_state
        image_attns = torch.ones(image_embeddings.size()[:-1],dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=128,
                              return_tensors="pt").to(image.device)

        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, text_labels = self.text_mask(input_ids, self.bert_net.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix)

        text_embeddings = self.bert_net(input_ids, attention_mask=text.attention_mask, return_dict=True).last_hidden_state
        text_attns = text.attention_mask

        text_crossout, image_crossout = self.cross_encoder(text_embeddings, image_embeddings, text_attns, image_attns)

        # ===============MLM===============================================
        prediction_scores = self.mlm_head(text_crossout)
        mlm_loss = self.mlm_fn(prediction_scores, text_labels)

        # ================ITM==============================================
        sim_t2i, sim_i2t = self.sim_ti(text_embeddings, image_embeddings)
        positive_output = text_crossout.clone()
        text_embed_all, text_attns_all, image_embed_all, image_attns_all = self.neg_data(text_embeddings, text_attns, image_embeddings, image_attns, sim_t2i, sim_i2t)
        text_crossout_neg, image_crossout_neg = self.cross_encoder(text_embed_all, image_embed_all, text_attns_all, image_attns_all)
        negative_output = text_crossout_neg.clone()
        itm_loss, prob_ret = self.itm_fn(positive_output, negative_output)

        loss = self.uncertain_weight(itm_loss, self.sigma1) + self.uncertain_weight(mlm_loss, self.sigma2)

        infer_cls = text_crossout[:, :1, :]
        infer_cls.squeeze()

        ret = {
            "infer_cls": infer_cls,
            "positive_prob": prob_ret["positive_prob"]
        }

        return loss, itm_loss, mlm_loss, ret


    @torch.no_grad()
    def text_mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def frozen_single_param(self):
        for name, parameter in self.bert_net.named_parameters():
            parameter.requires_grad = False
        for name, parameter in self.visual_net.named_parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    @autocast()
    def predict(self, text_ids, text_attention_masks, patch_images):
        text_ids = text_ids.squeeze()
        text_attention_masks = text_attention_masks.squeeze()
        image_embeddings = self.visual_net(patch_images, return_dict=True).last_hidden_state
        # print(image_embeddings)
        image_attns = torch.ones(image_embeddings.size()[:-1], dtype=torch.long).to(patch_images.device)
        text_embeddings = self.bert_net(text_ids, attention_mask=text_attention_masks, return_dict=True).last_hidden_state
        # print(text_embeddings)
        # print("=="*20)
        text_crossout, image_crossout = self.cross_encoder(text_embeddings, image_embeddings, text_attention_masks, image_attns)
        # print(text_crossout, image_crossout)
        infer_cls = text_crossout[:, :1, :]
        # print("==="*10)
        # print(infer_cls)
        infer_cls.squeeze()
        cross_prediction_scores = self.mlm_head(text_crossout)
        single_prediction_scores = self.mlm_head(text_embeddings)

        ret = {
            "cls": infer_cls,
            "cross_text": text_crossout,
            "cross_vision": image_crossout,
            "text_prob": cross_prediction_scores,
            "bert_text_prob": single_prediction_scores,
        }
        return ret


class TMEP_V2(nn.Module):
    def __init__(self,
                 bert_config,
                 vit_config,
                 cross_config,
                 dvae_config,
                 frozen_single=True):
        super().__init__()
        self.bert_config = bert_config
        self.vit_config = vit_config
        self.cross_config = cross_config
        self.dvae_config = dvae_config
        self.frozen_single = frozen_single

        self.bert_net, self.text_width = bert_backbone(self.bert_config)
        self.visual_net, self.visual_width = vit_mask_backbone(self.vit_config)
        self.cross_encoder = MultiEncoder(cross_config=self.cross_config)
        self.dvae_net = dvae_backbone(self.dvae_config)
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.text_vocab_size = self.bert_config.vocab_size
        if cross_config.cross_name == "large":
            self.num_attention_heads = self.cross_config.cross_large.num_attention_heads
            self.hidden_size = self.cross_config.cross_large.hidden_size

        self.tv_maks = TV_Mask(self.num_attention_heads)

        # MLM
        self.mlm_head = MLMHead(self.hidden_size, self.text_vocab_size)
        self.mlm_fn = MLMLoss(self.text_vocab_size)

        # ITM
        self.sim_ti = SimilarForTextImage(self.text_width, self.visual_width)
        self.neg_data = NegDataCollector()
        self.itm_fn = ITMLoss(self.hidden_size)

        # MIM
        self.layernorm = nn.LayerNorm(self.visual_width, eps=1e-12)
        self.mim_head = MIMHead(self.hidden_size, self.dvae_config.num_tokens)
        self.mim_fn = MIMLoss(self.dvae_config.num_tokens)


        #Uncertain Weigth
        self.sigma1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.sigma2 = nn.Parameter(torch.ones(1, requires_grad=True))

        # FIXME For lora, this method will not be executed
        if False:
            self.frozen_single_param()

    def save_pretrained(self, save_path, use_lora=False):
        if use_lora:
            # lora merge to get a updated base model
            self.bert_net.merge_and_unload()
            self.visual_net.merge_and_unload()
        state_file = os.path.join(save_path, "tmep_model.pth")
        state = {
            "bert_net": self.bert_net.state_dict(),
            "visual_net": self.visual_net.state_dict(),
            "cross_encoder": self.cross_encoder.state_dict(),
            "mlm_head": self.mlm_head.state_dict(),
            "mim_head": self.mim_head.state_dict(),
            "sim_ti": self.sim_ti.state_dict(),
            "itm_fn": self.itm_fn.state_dict()
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location=None, use_lora=False):
        state_file = os.path.join(load_path, "tmep_model.pth")
        if not os.path.exists(state_file):
            raise ValueError('state filepath is wrong.')
        state = torch.load(state_file, map_location=map_location)
        self.bert_net.load_state_dict(state["bert_net"], strict=False)
        self.visual_net.load_state_dict(state["visual_net"], strict=False)
        self.cross_encoder.load_state_dict(state["cross_encoder"], strict=False)
        self.mlm_head.load_state_dict(state["mlm_head"], strict=False)
        self.mim_head.load_state_dict(state["mim_head"], strict=False)
        self.sim_ti.load_state_dict(state["sim_ti"], strict=False)
        self.itm_fn.load_state_dict(state["itm_fn"], strict=False)
        
        if use_lora:
            # freeze all param
            for name, param in self.named_parameters():
                param.requires_grad = False
            # lora model replace
            self.bert_net = get_peft_model(self.bert_net, BERT_LORA_CONFIG)
            self.visual_net = get_peft_model(self.visual_net, VIT_LORA_CONFIG)



    # @autocast()
    def forward(self, text_ids, text_attention_masks, patch_images, patch_nums = 14 ** 2):
        if text_ids.shape[0] != 1:
            text_ids = text_ids.squeeze()
            text_attention_masks = text_attention_masks.squeeze()
        image_mask_positions = torch.zeros(int(patch_images.shape[0]), patch_nums).bool().cuda()  # Indicates which patches are masked (1) and which aren’t (0).
        
        all_states = self.visual_net(patch_images, image_mask_positions, output_hidden_states=True, return_dict=True).hidden_states
        image_embeddings = all_states[-1]
        image_embeddings = self.layernorm(image_embeddings)
        image_attns = torch.ones(image_embeddings.size()[:-1], dtype=torch.long).to(patch_images.device)

        input_ids = text_ids.clone()
        text_embeddings = self.bert_net(input_ids, attention_mask=text_attention_masks, return_dict=True).last_hidden_state
        text_attns = text_attention_masks.clone()

        text_crossout, image_crossout = self.cross_encoder(text_embeddings, image_embeddings, text_attns, image_attns)
        
        return text_crossout, image_crossout

        # FIXME For lora, below method will not be executed
        
        # ===============MLM===============================================
        prediction_scores = self.mlm_head(text_crossout)
        mlm_loss = self.mlm_fn(prediction_scores, text_labels)

        # ================ITM==============================================
        sim_t2i, sim_i2t = self.sim_ti(text_embeddings, image_embeddings)
        positive_output = text_crossout.clone()
        text_embed_all, text_attns_all, image_embed_all, image_attns_all = self.neg_data(text_embeddings, text_attns, image_embeddings, image_attns, sim_t2i, sim_i2t)
        text_crossout_neg, image_crossout_neg = self.cross_encoder(text_embed_all, image_embed_all, text_attns_all, image_attns_all)
        negative_output = text_crossout_neg.clone()
        itm_loss, prob_ret = self.itm_fn(positive_output, negative_output)

        # ================MIM==============================================
        image_tokens = self.dvae_net.get_codebook_indices(token_images).flatten(1)
        image_labels = self.image_label(image_tokens, image_mask_positions)
        image_prediction_scores = self.mim_head(image_crossout)
        mim_loss = self.mim_fn(image_prediction_scores, image_labels)

        # Uncertainty Weight
        # loss = self.uncertain_weight(itm_loss, self.sigma1) + self.uncertain_weight(mlm_loss, self.sigma2)
        loss = itm_loss+mlm_loss+mim_loss

        infer_cls = text_crossout[:, :1, :]
        infer_cls.squeeze()

        ret = {
            "infer_cls": infer_cls,
            "positive_prob": prob_ret["positive_prob"]
        }

        return loss, itm_loss, mlm_loss, mim_loss, ret


    def image_label(self, image_tokens, mask_positions):
        image_labels = image_tokens.clone()
        image_labels[~mask_positions] = -100
        return image_labels

    def uncertain_weight(self, loss, sigma):
        return 0.5 / (sigma ** 2) * loss + torch.log(1 + sigma ** 2)

    def frozen_single_param(self):
        for name, parameter in self.bert_net.named_parameters():
            parameter.requires_grad = False
        for name, parameter in self.visual_net.named_parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    @autocast()
    def predict(self, text_ids, text_attention_masks, patch_images):
        text_ids = text_ids.squeeze()
        text_attention_masks = text_attention_masks.squeeze()
        all_states = self.visual_net(patch_images, output_hidden_states=True, return_dict=True).hidden_states
        image_embeddings = all_states[-1]
        image_embeddings = self.layernorm(image_embeddings)
        image_attns = torch.ones(image_embeddings.size()[:-1], dtype=torch.long).to(patch_images.device)
        text_embeddings = self.bert_net(text_ids, attention_mask=text_attention_masks, return_dict=True).last_hidden_state
        text_crossout, image_crossout = self.cross_encoder(text_embeddings, image_embeddings, text_attention_masks, image_attns)
        infer_cls = text_crossout[:, :1, :]
        infer_cls.squeeze()
        infer_cls.squeeze()
        cross_prediction_scores = self.mlm_head(text_crossout)
        single_prediction_scores = self.mlm_head(text_embeddings)
        ret = {
            "cls": infer_cls,
            "cross_text": text_crossout,
            "cross_vision": image_crossout,
            "text_prob": cross_prediction_scores,
            "bert_text_prob": single_prediction_scores,
        }
        return ret
