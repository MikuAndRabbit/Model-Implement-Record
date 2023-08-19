import os
import json
from typing import Dict, Tuple
import torchvision
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer

class VQA2Dataset(Dataset):
    def __init__(self, standard_filepath: str, image_root: str, image_size: int | Tuple[int, int], answer2label: Dict[str, int], 
                 vocab_filepath: str, text_max_length: int, augumentation: bool = False, question_id_out: bool = False) -> None:
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.image_paths, self.questions, self.answers, self.question_ids = [], [], [], []
        self.question_id_out = question_id_out
        
        # answer mapper -> label
        self.answer2label = answer2label
        self.candidate_answers_num = len(answer2label)
        
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
        self.augumentation = augumentation
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.AutoAugment() if self.augumentation else nn.Identity(),
            torchvision.transforms.ToTensor()
        ])
        
        # prase dataset file
        with open(standard_filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data_item = json.loads(line)
            self.image_paths.append(os.path.join(image_root, data_item['image']))
            self.questions.append(data_item['question'])
            self.answers.append(data_item['answer'])
            self.question_ids.append(data_item['id'])
        assert len(self.image_paths) == len(self.questions) == len(self.answers), 'The length of image_paths, questions, answers must be equal'
    
    def __getitem__(self, index):
        # get question, answer
        question = self.questions[index]
        answer = self.answers[index]
        
        # get label
        answer_label = torch.zeros(self.candidate_answers_num)
        idx = self.answer2label.get(answer)
        if idx is not None:
            answer_label[idx] = 1
        
        # tokenize question
        token_res = self.text_tokenizer(question, **self.text_tokenizer_param)
        question_token = token_res['input_ids']
        question_attention_mask = token_res['attention_mask']
        
        # get image via PIL
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transformer(image)

        if self.question_id_out:
            # question id
            question_id = self.question_ids[index]
            return question_id, (image, question_token, question_attention_mask), answer_label
        else:
            return (image, question_token, question_attention_mask), answer_label
        
    
    def __len__(self):
        return len(self.image_paths)
