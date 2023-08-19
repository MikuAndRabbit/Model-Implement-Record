import os
from typing import Any, Dict
import torch
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, cls_rep):
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3_VQA_Head(nn.Module):
    def __init__(self, input_dim: int, num_answers: int, layernorm_eps: float = 1e-12) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.answer_num = num_answers
        self.pooler = Pooler(
            input_features = input_dim, 
            output_features = input_dim, 
            norm_layer = nn.LayerNorm, 
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), 
            nn.LayerNorm(input_dim * 2, eps = layernorm_eps), 
            nn.GELU(), 
            nn.Linear(input_dim * 2, num_answers), 
        )


    def forward(self, fusion_feature):
        representation = self.pooler(fusion_feature)
        return self.head(representation)


    def save_model(self, save_folder: str, filename: str = 'beit3_vqa_head.pth'):
        checkpoint_path = os.path.join(os.path.abspath(save_folder), filename)
        saved_dict = {
            'init_params': {
                'embed_dim': self.input_dim,
                'answer_num': self.answer_num,
            },
            'weights': {
                'pooler': self.pooler.state_dict(),
                'head': self.head.state_dict(),
            }
        }
        torch.save(saved_dict, checkpoint_path)
    
    
    def from_pretrained(self, save_folder: str, map_location: str, filename: str = 'beit3_vqa_head.pth'):
        checkpoint_path = os.path.join(os.path.abspath(save_folder), filename)
        saved_dict = torch.load(checkpoint_path, map_location = map_location)
        # load weights
        weights = saved_dict['weights']
        self.pooler.load_state_dict(weights['pooler'])
        self.head.load_state_dict(weights['head'])


class Plain_VQA_Head(nn.Module):
    def __init__(self, input_dim: int, num_answers: int, layernorm_eps: float = 1e-12) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_answers = num_answers
        self.hidden_dim = input_dim * 2
        self.layernorm_eps = layernorm_eps
        
        self.vqa_classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim, eps = self.layernorm_eps),
            nn.Linear(self.hidden_dim, self.num_answers)
        )

    
    def forward(self, fusion_feature):
        return self.vqa_classifier(fusion_feature)


    def save_model(self, save_folder: str, filename: str = 'vqa_head.pth'):
        checkpoint_path = os.path.join(os.path.abspath(save_folder), filename)
        saved_dict = {
            'init_params': {
                'input_dim': self.input_dim,
                'num_answers': self.num_answers,
                'layernorm_eps': self.layernorm_eps
            },
            'weights': self.vqa_classifier.state_dict()
        }
        torch.save(saved_dict, checkpoint_path)


    def from_pretrained(self, save_folder: str, map_location: str, filename: str = 'vqa_head.pth'):
        checkpoint_path = os.path.join(os.path.abspath(save_folder), filename)
        saved_dict = torch.load(checkpoint_path, map_location = map_location)
        vqa_classifier_weights = saved_dict['weights']
        self.vqa_classifier.load_state_dict(vqa_classifier_weights)


class VQA_Fusion(nn.Module):
    def __init__(self, method: str = 'multiply') -> None:
        super().__init__()
        self.method = method
        self.activation_fn = nn.Sigmoid()

    def forward(self, input_embedding):
        text_embedding, image_embedding = input_embedding
        assert len(text_embedding) == len(image_embedding), 'The shape of text_embedding must be equal to image_embedding'
        if len(text_embedding) == 2:
            text_embedding = text_embedding.unsqueeze(0)
            image_embedding = image_embedding.unsqueeze(0)
        
        # get cls token for text & image
        text_cls = text_embedding[:, 0]
        image_cls = image_embedding[:, 0]
        assert text_cls.shape == image_cls.shape, 'The shape of text_cls must be equal to image_cls'

        # process
        fusion = None
        if self.method == 'multiply':
            fusion = torch.mul(text_cls, image_cls)
        else:
            return ValueError('Not supported method')
        
        # activation
        return self.activation_fn(fusion)
    
    def save_model(self, save_folder: str, filename: str = 'vqa_fusion.pth'):
        checkpoint_path = os.path.join(os.path.abspath(save_folder), filename)
        saved_dict = {
            'init_params': {
                'method': self.method,
            },
            'activation_fn': self.activation_fn.state_dict()
        }
        torch.save(saved_dict, checkpoint_path)


def get_vqa_head(head_type: str):
    if head_type == 'plain':
        return Plain_VQA_Head
    elif head_type == 'beit3':
        return BEiT3_VQA_Head
    return None


class VQA(nn.Module):
    def __init__(self, *, backbone_model: nn.Module, fusion_method: str, vqa_head_type: str, vqa_head_param: Dict[str, Any]) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.head_type = vqa_head_type
        self.fusion_method = fusion_method
        Head = get_vqa_head(self.head_type)
        if Head is None:
            raise ValueError('wrong value of vqa_head_type parameter')
        self.vqa_head = Head(**vqa_head_param)
        self.activation_fn = nn.Sigmoid()


    def forward(self, **backbone_input):
        # get text & image embedding
        text_embedding, image_embedding = self.backbone_model(**backbone_input)
        
        # get cls token for text & image
        text_cls = text_embedding[:, 0]
        image_cls = image_embedding[:, 0]
        assert text_cls.shape == image_cls.shape, 'The shape of text_cls must be equal to image_cls'

        # fuse text & image information
        if self.fusion_method == 'multiply':
            fusion = torch.mul(text_cls, image_cls)
        elif self.fusion_method == 'concat':
            fusion = torch.concat((text_cls, image_cls), dim = 2)
            fusion = fusion.squeeze(dim = 1)
        else:
            raise ValueError('wrong value of fusion_method parameter')
        fusion = self.activation_fn(fusion)
        
        # get output
        vqa_output = self.vqa_head(fusion)
        
        return vqa_output


    def predict(self, label2answer: Dict[int, Any], get_head_output: bool = False, **backbone_input):
        # get text & image embedding
        text_embedding, image_embedding = self.backbone_model(**backbone_input)
        
        # get cls token for text & image
        text_cls = text_embedding[:, 0]
        image_cls = image_embedding[:, 0]
        assert text_cls.shape == image_cls.shape, 'The shape of text_cls must be equal to image_cls'

        # fuse text & image information
        fusion = torch.mul(text_cls, image_cls)
        fusion = self.activation_fn(fusion)
        
        # get output
        vqa_output = self.vqa_head(fusion)

        # get answers
        sigmoid_vqa_output = self.activation_fn(vqa_output)
        indexs = torch.argmax(sigmoid_vqa_output, dim = 1)
        answers = [label2answer[int(idx.item())] for idx in indexs]
        
        if get_head_output:
            return answers, vqa_output
        else:
            return answers


    def save_model(self, save_folder: str, use_lora: bool = False):
        self.backbone_model.save_pretrained(save_folder, use_lora)
        self.vqa_head.save_model(save_folder)


    def from_pretrained(self, save_folder: str, map_location: str, use_lora: bool = False):
        self.backbone_model.from_pretrained(save_folder, map_location, use_lora)
        self.vqa_head.from_pretrained(save_folder, map_location)
