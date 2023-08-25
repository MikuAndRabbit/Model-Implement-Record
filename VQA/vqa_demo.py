import os
import json
import yaml
from typing import Any, Dict, List, Tuple
import torchvision
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertTokenizer
from model.TMEP_pretrain import TMEP_V2
from vqa import VQA
from config.yaml_config import get_config
import gradio as gr


ANSWER_PROMPT = {
    'what color is': 'The color is',
    'what colors are': 'The colors are',
    'what is the color of': 'The color is',
    'what sport is': 'It\'s',
    'what room is': 'It\'s',
    'what is it': 'It\'s',
    'what can i': 'You can',
    'where is': 'It\'s'
}


def add_defaults(func, **defaults):
    def wrapper(*args, **kwargs):
        new_kwargs = {**defaults, **kwargs}
        return func(*args, **new_kwargs)
    return wrapper


class VQA_Inference():
    def __init__(self, vqa_model: VQA, device, image_size: int | Tuple[int, int], label2answer: Dict[int, Any], vocab_filepath: str, text_max_length: int) -> None:
        # VQA model
        self.vqa_model = vqa_model
        self.device = device
        
        # image size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        
        # label2answer mapper
        self.label2answer = label2answer
        
        # text tokenizer
        self.text_max_length = text_max_length
        self.text_tokenizer_param = {
            'return_tensors': 'pt', 
            'return_attention_mask': True,
            'max_length': self.text_max_length,
            'padding': 'max_length'
        }
        self.text_tokenizer = BertTokenizer(vocab_file = vocab_filepath)
        
        # image transformer & augumentation
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.ToTensor()
        ])


    def predict(self, images_path: str | List[str], questions: str | List[str]):
        # convert to list
        if isinstance(images_path, str):
            images_path = [images_path, ]
        if isinstance(questions, str):
            questions = [questions, ]
        
        # get representation of images
        images_representation = []
        for image_path in images_path:
            image = Image.open(image_path).convert('RGB')
            image = self.transformer(image)
            images_representation.append(image)
        images_representation = torch.stack(images_representation)

        # get representation of questions
        question_tokenize_res = self.text_tokenizer(questions, **self.text_tokenizer_param)
        questions_token = question_tokenize_res['input_ids']
        questions_attention_mask = question_tokenize_res['attention_mask']
        
        # move to device
        images_representation = images_representation.to(self.device)
        questions_token = questions_token.to(self.device)
        questions_attention_mask = questions_attention_mask.to(self.device)
        
        # vqa predict input
        backbone_input = {
            'text_ids': questions_token, 
            'text_attention_masks': questions_attention_mask,
            'patch_images': images_representation
        }
        answers = self.vqa_model.predict(self.label2answer, False, **backbone_input)
        
        # make full answers
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            in_prompt = False
            for question_prefix, prompt in ANSWER_PROMPT.items():
                question = question.strip().lower()
                if question.startswith(question_prefix):
                    answers[idx] = prompt + ' ' + answer
                    in_prompt = True
                    break
            if in_prompt:
                break
            else:
                answers[idx] = 'The answer is ' + answer
        
        return answers


def load_inference_model(config_filepath: str):
    # load config
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        
    # basic model & init
    model_dict = config['model']
    device = model_dict['device']
    # tmep model
    model_tmep_dict = model_dict['tmep']
    tmep_config = get_config(model_tmep_dict['config-path'])
    tmep = TMEP_V2(tmep_config.model.bert, tmep_config.model.vit, tmep_config.model.cross_encoder, tmep_config.model.dvae)
    
    # VQA Model
    model_vqa_dict = model_dict['vqa']
    model = VQA(backbone_model = tmep, fusion_method = model_vqa_dict['fusion_method'], 
                vqa_head_type = model_vqa_dict['head_type'], vqa_head_param = model_vqa_dict['head'])
    model.from_pretrained(model_dict['pretrain_folder'], 'cpu', False)
    model = model.to(device)
    
    # answer2label mapper
    data_dict = config['data']
    with open(data_dict['label2answer_json'], 'r') as f:
       label2answer = json.load(f)
    
    # inference model
    inference_model_dict = model_dict['inference']
    inference_model = VQA_Inference(model, device, (inference_model_dict['image_size'], inference_model_dict['image_size']), label2answer, 
                                    data_dict['vocab_filepath'], inference_model_dict['text_max_length'])
    
    return inference_model


def vqa_single_question(image, question, inference_model):
    answers = inference_model.predict(image, question)
    return answers[0]


if __name__ == "__main__":
    # load model
    config_path = r''
    inference_model = load_inference_model(config_path)

    # demo interface
    vqa_demo = add_defaults(vqa_single_question, inference_model = inference_model)
    demo = gr.Interface(
        fn = vqa_demo,
        inputs = [
            gr.Image(type = "filepath", label = 'Image'),
            gr.Textbox(label = 'Question'),
        ],
        outputs = [
            gr.Textbox(label = 'Answer'),
        ],
        title = 'Visual Question Answer - V2',
    )
    demo.launch()
