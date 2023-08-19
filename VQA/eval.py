import os
from typing import Any, Dict
import yaml
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch.nn as nn
from torch.utils.data import DataLoader
from vqa import VQA
from data.dataset import VQA2Dataset
from config.yaml_config import get_config
from model.TMEP_pretrain import TMEP_V2


EVAL_PREFIX = 'EVALUATION'


def init_logger(logger_file_folder: str, roration: str):
    # remove all
    logger.remove()
    
    # create EVAL level
    logger.level(name = 'EVAL', no = 20)
    
    # create file logger
    logger.add(sink = os.path.join(logger_file_folder, 'info.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "INFO")
    logger.add(sink = os.path.join(logger_file_folder, 'debug.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "DEBUG")
    logger.add(sink = os.path.join(logger_file_folder, 'trace.log'), rotation=roration, compression="zip", enqueue=True, level="TRACE")
    logger.add(sink = os.path.join(logger_file_folder, 'eval.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "EVAL")


def init_seeds(seed = 0, cuda_deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def vqa_eval(model: nn.Module, eval_dataloader: DataLoader, loss_fn: nn.Module, device, label2answer: Dict[int, Any]):
    total_loss, total_cnt = .0, 0
    questionid2answer = {}
    processer_bar = tqdm(total = len(eval_dataloader))
    for idx, (question_ids, (images, question_tokens, question_attention_masks), labels) in enumerate(eval_dataloader):
        # print to log
        logger.trace(f'{EVAL_PREFIX} batch index: {idx}')
        processer_bar.set_description(f'{EVAL_PREFIX} batch index: {idx}')

        # move to cuda
        images = images.to(device)
        question_tokens = question_tokens.to(device)
        question_attention_masks = question_attention_masks.to(device)
        labels = labels.to(device)

        # eval
        with torch.no_grad():
            input_param = {
                'text_ids': question_tokens, 
                'text_attention_masks': question_attention_masks,
                'patch_images': images
            }
            answers_list, output = model.predict(label2answer, True, **input_param)
        loss = loss_fn(output, labels)
        
        # statics
        total_loss += loss.cpu().item()
        total_cnt += images.shape[0]
        question_ids_list = question_ids.tolist()
        for questionid, answer in zip(question_ids_list, answers_list):
            questionid2answer[questionid] = answer
        
        # update processer bar
        processer_bar.update(1)
        processer_bar.set_postfix({'loss': loss.cpu().item()})

    # rank loss info
    eval_statistic_output = f'{EVAL_PREFIX} items: {total_cnt}, loss: {total_loss}'
    logger.info(eval_statistic_output)
    logger.log('EVAL', eval_statistic_output)
    
    return total_loss, questionid2answer


if __name__ == "__main__":
    # load config
    CONFIG_PATH = r''
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # logger setting
    eval_dict = config['eval']
    logger_dict = eval_dict['log']
    logger_file_folder = logger_dict['path']['folder']
    roration = logger_dict['rotation']
    init_logger(logger_file_folder, roration)
    
    # eval control
    device = torch.device(eval_dict['device'])
    pretrain_weights_folder = eval_dict['pretrain_weights_folder']
    
    # basic model & init
    model_dict = config['model']
    # tmep model
    model_tmep_dict = model_dict['tmep']
    tmep_config = get_config(model_tmep_dict['config-path'])
    tmep = TMEP_V2(tmep_config.model.bert, tmep_config.model.vit, tmep_config.model.cross_encoder, tmep_config.model.dvae)
        
    # VQA Model
    model_vqa_dict = model_dict['vqa']
    model = VQA(backbone_model = tmep, fusion_method = model_vqa_dict['fusion_method'], vqa_head_type = model_vqa_dict['head_type'], vqa_head_param = model_vqa_dict['head'])
    model.from_pretrained(pretrain_weights_folder, 'cpu', False)
    model = model.to(device)
    logger.debug(f'Load VQA model weights from {pretrain_weights_folder} and move model to {device}')
    
    # dataset
    data_dict = config['data']
    dataset_dict = data_dict['dataset']
    # answer2label mapper
    with open(data_dict['answer2label_json'], 'r') as f:
       answer2label = json.load(f)
    # label2answer mapper
    with open(data_dict['label2answer_json'], 'r') as f:
       label2answer = json.load(f)
    eval_dataset_dict = dataset_dict['eval']
    eval_dataset = VQA2Dataset(**eval_dataset_dict, answer2label = answer2label, question_id_out = True)
    
    # dataloader
    dataloader_dict = data_dict['dataloader']
    eval_loader_dict = dataloader_dict['eval']
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_loader_dict['bs'], num_workers = eval_loader_dict['num_workers'])
    
    # loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # evaluation
    predict_results_filepath = eval_dict['result_path']
    loss, questionid2answer = vqa_eval(model, eval_dataloader, loss_fn, device, label2answer)
    with open(predict_results_filepath, 'w') as f:
        json.dump(questionid2answer, f)
    logger.debug(f'The predicting result has beed saved to {predict_results_filepath}')
    
    # print results to log
    questionid2question = {}
    with open(eval_dataset_dict['standard_filepath'], 'r') as f:
        question_id_answer_items = f.readlines()
    for item_jsonline in question_id_answer_items:
        item = json.loads(item_jsonline)
        question_id, question, standard_answer = int(item['id']), item['question'], item['answer']
        questionid2question[question_id] = (question, standard_answer)
    
    question_ids, questions, predict_results, standard_answers = [], [], [], []
    for question_id, predict_answer in questionid2answer.items():
        question_id = int(question_id)
        question, standard_answer = questionid2question[question_id]
        question_ids.append(str(question_id))
        questions.append(question)
        predict_results.append(predict_answer)
        standard_answers.append(standard_answer)
    assert len(question_ids) == len(questions) == len(predict_results) == len(standard_answers)
    id_maxlen = max(len(s) for s in question_ids)
    ques_maxlen = max(len(s) for s in questions)
    pred_maxlne = max(len(s) for s in predict_results)
    ans_maxlen = max(len(s) for s in standard_answers)
    
    logger.info(f'Here are predicting results:')
    process_bar = tqdm(total = len(question_ids), desc = 'Print to log file')
    for id, question, predict_result, standard_answer in zip(question_ids, questions, predict_results, standard_answers):
        logger.info(f'id: {id:<{id_maxlen}}, question: {question:<{ques_maxlen}}, predict answer: {predict_result:<{pred_maxlne}}, standard answer: {standard_answer:<{ans_maxlen}}')
        process_bar.update(1)
