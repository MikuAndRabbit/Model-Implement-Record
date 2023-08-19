import argparse
from collections import deque
import random
from typing import Dict, Optional
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader
from loguru import logger
from vqa import VQA
from data.dataset import VQA2Dataset
from config.yaml_config import get_config
from model.TMEP_pretrain import TMEP_V2
import yaml
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json

TRAIN_PREFIX = f'{"TRAIN":<10}'
VAL_PREFIX = 'VALIDATION'


def print_require_grad(model: nn.Module, printer):
    infos = [(name, param.requires_grad) for name, param in model.named_parameters()]
    names, req_grads = list(zip(*infos))
    max_len = max([len(x) for x in names])
    for i in range(len(names)):
        printer(f"{names[i]:<{max_len}}   {str(req_grads[i]):<5}")


class VQA_Train():
    def __init__(self, *, model: nn.Module, rank: int, world_size: int, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], device,
                 loss_fn: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, checkpoint_pos: Optional[str], max_saved_checkpoints: Optional[int], use_lora: bool) -> None:
        # model
        self.model = model
        self.device = device
        self.use_lora = use_lora
        
        # dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # loss function & optimizer & scheduler
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # other info & statistic
        self.rank = rank
        self.world_size = world_size
        self.master_rank = (self.rank == 0)
        self.update_cnt = torch.tensor(0, dtype = torch.int64).cuda().detach()
        self.world_updates = [torch.zeros(1, dtype=torch.int64).cuda().detach() for _ in range(self.world_size)]
        self.world_train_loss = [torch.zeros(2, dtype = torch.float32).cuda().detach() for _ in range(self.world_size)]
        self.world_val_loss = [torch.zeros(2, dtype = torch.float32).cuda().detach() for _ in range(self.world_size)]
        self.epoch_loss_info = []  # (loss, batch_item_cnt)
        
        # checkpoint
        self.checkpoint_pos = os.path.abspath(checkpoint_pos) if checkpoint_pos is not None else None
        self.max_saved_checkpoints = max_saved_checkpoints
        self.saved_checkpoint_path = deque()
        self.best_train_loss = None
        self.best_train_epoch = None
        self.best_val_loss = None
        self.best_val_epoch = None
        self.saved_epoch_idx = set()


    def is_master_rank(self):
        return self.master_rank


    def save_model(self, epoch: int):
        assert self.checkpoint_pos is not None
        # make checkpoint folder
        saved_folder = os.path.join(self.checkpoint_pos, f'{epoch}_{self.update_cnt}')
        if not os.path.exists(saved_folder):
            os.mkdir(saved_folder)
        
        # save the checkpoint
        self.model.module.save_model(saved_folder, self.use_lora)
        logger.debug(f'Detailed information have been saved to {saved_folder}')
        
        # check the number of checkpoints
        self.saved_checkpoint_path.append(saved_folder)
        if self.max_saved_checkpoints is not None and len(self.saved_checkpoint_path) > self.max_saved_checkpoints:
            poped_checkpoint_folder = self.saved_checkpoint_path.popleft()
            if os.path.exists(poped_checkpoint_folder):
                os.remove(poped_checkpoint_folder)
                logger.debug(f'The number of checkpoints exceed, {poped_checkpoint_folder} have been removed!')
            else:
                logger.warning(f'Checkpoint folder {poped_checkpoint_folder} doesn\'t exist, skip.')


    @staticmethod
    # TODO need change to new method implement
    def load_model(checkpoint_path: str, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, map_location = torch.device('cuda')):
        assert checkpoint_path is not None
        checkpoint_path = os.path.abspath(checkpoint_path)
        saved_dict = torch.load(checkpoint_path, map_location = map_location)
        # load dict
        model.load_state_dict(saved_dict['model'])
        optimizer.load_state_dict(saved_dict['optimizer'])
        lr_scheduler.load_state_dict(saved_dict['lr_scheduler'])

        return saved_dict['update_cnt']


    def val(self, epoch: int):
        total_loss, total_cnt = .0, 0
        processer_bar = tqdm(total = len(self.val_dataloader), disable = not self.master_rank)
        for idx, ((images, question_tokens, question_attention_masks), labels) in enumerate(self.val_dataloader):
            # print to log
            if self.master_rank: logger.trace(f'{VAL_PREFIX} epoch: {epoch}, batch index: {idx}')
            processer_bar.set_description(f'{VAL_PREFIX} epoch: {epoch}, batch index: {idx}')
            
            # move to cuda
            images = images.to(self.device)
            question_tokens = question_tokens.to(self.device)
            question_attention_masks = question_attention_masks.to(self.device)
            labels = labels.to(self.device)
            
            # epoch setting
            self.val_dataloader.sampler.set_epoch(1)
            
            # val
            with torch.no_grad():
                input_param = {
                    'text_ids': question_tokens, 
                    'text_attention_masks': question_attention_masks,
                    'patch_images': images
                }
                output = self.model(**input_param)
            loss = self.loss_fn(output, labels)
            
            # statics
            total_loss += loss.cpu().item()
            total_cnt += images.shape[0]
            
            # update processer bar
            processer_bar.update(1)
            processer_bar.set_postfix({'loss': loss.cpu().item()})

        # rank loss info
        logger.debug(f'{VAL_PREFIX} epoch: {epoch}, rank id: {self.rank}, items: {total_cnt}, loss: {total_loss}')

        # gather all loss & cnt
        proc_loss_cnt = torch.tensor((total_loss, total_cnt), dtype = torch.float32).cuda().detach()
        torch.distributed.all_gather(self.world_val_loss, proc_loss_cnt)
        if self.master_rank:
            # get val loss & cnt
            val_loss, val_cnt = .0, 0
            for loss_cnt in self.world_val_loss:
                val_loss += loss_cnt[0]
                val_cnt += loss_cnt[1]

            # print to log
            val_cnt = int(val_cnt)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            val_statistic_output = f'epoch {epoch}, item: {val_cnt}, lr: {learning_rate}, loss: {val_loss}'
            logger.info(val_statistic_output)
            logger.log('VAL', val_statistic_output)
        
            # checkpoint
            if self.best_val_loss is None:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch
                if self.checkpoint_pos is not None and epoch not in self.saved_epoch_idx:
                    self.save_model(epoch)
                    self.saved_epoch_idx.add(epoch)
            else:
                if self.best_val_loss >= val_loss:
                    logger.debug(f'{VAL_PREFIX} previous best loss: {self.best_val_loss} ({self.best_val_epoch}) -> current best loss: {val_loss} ({epoch})')
                    self.best_train_loss = val_loss
                    self.best_val_epoch = epoch
                    if self.checkpoint_pos is not None and epoch not in self.saved_epoch_idx:
                        self.save_model(epoch)
                        self.saved_epoch_idx.add(epoch)
 

    def _train(self, epoch: int):
        proc_epoch_cnt = 0
        proc_epoch_loss = .0
        processer_bar = tqdm(total = len(self.train_dataloader), disable = not self.master_rank)
        for idx, ((images, question_tokens, question_attention_masks), labels) in enumerate(self.train_dataloader):
            # move to cuda
            images = images.to(self.device)
            question_tokens = question_tokens.to(self.device)
            question_attention_masks = question_attention_masks.to(self.device)
            labels = labels.to(self.device)
            
            # print to log
            if self.master_rank: logger.trace(f'{TRAIN_PREFIX} epoch: {epoch}, batch index: {idx}')
            processer_bar.set_description(f'{TRAIN_PREFIX} epoch: {epoch}, batch index: {idx}')
            
            # epoch setting
            self.train_dataloader.sampler.set_epoch(epoch)
            
            # get the output of model
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype = torch.bfloat16):  # type: ignore
                input_param = {
                    'text_ids': question_tokens, 
                    'text_attention_masks': question_attention_masks,
                    'patch_images': images
                }
                output = self.model(**input_param)
                loss = self.loss_fn(output, labels)
            
            # statics
            proc_epoch_loss += loss.cpu().item()
            self.update_cnt += 1
            
            # update model
            proc_epoch_cnt += images.shape[0]
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            
            # update processer bar
            processer_bar.update(1)
            processer_bar.set_postfix({'loss': loss.cpu().item()})
                    
        # rank epoch loss info
        logger.debug(f'{TRAIN_PREFIX} epoch: {epoch}, rank id: {self.rank}, items: {proc_epoch_cnt}, loss: {proc_epoch_loss}')
        
        # checkpoint
        loss_cnt_tensor = torch.tensor((proc_epoch_loss, proc_epoch_cnt), dtype = torch.float32).cuda().detach()
        torch.distributed.all_gather(self.world_train_loss, loss_cnt_tensor)
        if self.master_rank:
            # get global total loss
            total_epoch_loss, total_epoch_cnt = .0, 0
            for epoch_loss_cnt in self.world_train_loss:
                total_epoch_loss += epoch_loss_cnt[0].item()
                total_epoch_cnt += int(epoch_loss_cnt[1].item())

            # print to log
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_statistic_output = f'epoch: {epoch}, items: {total_epoch_cnt}, lr: {learning_rate}, loss: {total_epoch_loss}'
            logger.info(train_statistic_output)
            logger.log('TRAIN', train_statistic_output)
            self.epoch_loss_info.append((total_epoch_loss, total_epoch_cnt))

            # checkpoint
            if self.best_train_loss is None:
                self.best_train_loss = total_epoch_loss
                self.best_train_epoch = epoch
                if self.checkpoint_pos is not None and self.master_rank:
                    self.save_model(epoch)
                    self.saved_epoch_idx.add(epoch)
            else:
                if self.best_train_loss >= total_epoch_loss:
                    logger.debug(f'{TRAIN_PREFIX} previous best loss: {self.best_train_loss} ({self.best_train_epoch}) -> current best loss: {total_epoch_loss} ({epoch})')
                    self.best_train_loss = total_epoch_loss
                    self.best_train_epoch = epoch
                    if self.checkpoint_pos is not None and self.master_rank:
                        self.save_model(epoch)
                        self.saved_epoch_idx.add(epoch)


    def train(self, s_epoch: int, e_epoch: int, update_cnt : Optional[int] = None, need_val: bool = True, val_epoch_interval: int = 5):
        if update_cnt is not None:
            self.update_cnt = torch.tensor(update_cnt, dtype = torch.int64).cuda().detach()
        if self.master_rank: logger.info(f'Start training {s_epoch} -> {e_epoch}')
        for epoch in range(s_epoch, e_epoch + 1):
            if self.master_rank: logger.info(f'Start training of {epoch} epoch.')
            self._train(epoch)
            if need_val and epoch % val_epoch_interval == 0:
                if self.master_rank: logger.info(f'Start valuation of {epoch} epoch.')
                self.val(epoch)


def init_logger(logger_file_folder: str, roration: str):
    # remove all
    logger.remove()
    
    # create TRAIN & VAL level
    logger.level(name = 'TRAIN', no = 20)
    logger.level(name = 'VAL', no = 20)
    
    # create file logger
    logger.add(sink = os.path.join(logger_file_folder, 'info.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "INFO")
    logger.add(sink = os.path.join(logger_file_folder, 'debug.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "DEBUG")
    logger.add(sink = os.path.join(logger_file_folder, 'trace.log'), rotation=roration, compression="zip", enqueue=True, level="TRACE")
    logger.add(sink = os.path.join(logger_file_folder, 'train.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "TRAIN")
    logger.add(sink = os.path.join(logger_file_folder, 'val.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "VAL")


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


if __name__ == "__main__":
    # load config
    CONFIG_PATH = r''
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # logger setting
    train_dict = config['train']
    logger_dict = train_dict['log']
    logger_file_folder = logger_dict['path']['folder']
    roration = logger_dict['rotation']
    init_logger(logger_file_folder, roration)
    
    # bash args parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    ARGS = parser.parse_args()
    local_rank = ARGS.local_rank
    
    # DDP backend init
    distributed_dict = train_dict['distributed']
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend = distributed_dict['backend'])  # type: ignore
    device = torch.device(train_dict['device'], local_rank)

    # set a seed of random algorithm
    rank = torch.distributed.get_rank()
    init_seeds(seed = 1 + rank)
    
    # train control
    train_ctrl_dict = train_dict['control']
    s_epoch = train_ctrl_dict['s_epoch']
    e_epoch = train_ctrl_dict['e_epoch']
    freeze_pretrain = train_ctrl_dict['freeze_pretrain']
    use_lora = train_ctrl_dict['use_lora']
    
    # basic model & init
    model_dict = config['model']
    # tmep model
    model_tmep_dict = model_dict['tmep']
    tmep_config = get_config(model_tmep_dict['config-path'])
    tmep = TMEP_V2(tmep_config.model.bert, tmep_config.model.vit, tmep_config.model.cross_encoder, tmep_config.model.dvae)
    tmep.from_pretrained(model_tmep_dict['pretrain-path'], map_location = 'cpu', use_lora = use_lora)
    if use_lora:
        if local_rank == 0: logger.debug('using lora')
        if freeze_pretrain != 'cross encoder':
            for name, param in tmep.cross_encoder.named_parameters():
                param.requires_grad = True
    else:
        # freeze all
        for name, param in tmep.named_parameters():
            param.requires_grad = False
        # unfreeze some parameters
        if freeze_pretrain == 'single modal':
            if local_rank == 0: logger.debug('freeze single modal encoder')
            for name, param in tmep.cross_encoder.named_parameters():
                param.requires_grad = True
        elif freeze_pretrain == 'cross encoder':
            if local_rank == 0: logger.debug('freeze cross encoder')
            for name, param in tmep.bert_net.named_parameters():
                param.requires_grad = True
            for name, param in tmep.visual_net.named_parameters(): 
                param.requires_grad = True
        elif freeze_pretrain == 'none':
            if local_rank == 0: logger.debug('freeze nothing')
            for name, param in tmep.named_parameters():
                param.requires_grad = True
        else:
            raise ValueError('wrong value of freeze_pretrain config')
        
    if local_rank == 0:
        logger.debug('require grad of parameters in TMEP model:')
        print_require_grad(tmep, logger.debug)
    
    # VQA Model
    model_vqa_dict = model_dict['vqa']
    model = VQA(backbone_model = tmep, fusion_method = model_vqa_dict['fusion_method'], vqa_head_type = model_vqa_dict['head_type'], vqa_head_param = model_vqa_dict['head']).to(device)

    # loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # optimizer
    optimizer_dict = train_dict['optimizer']
    optimizer = AdamW(model.parameters(), betas = (optimizer_dict['beta_1'], optimizer_dict['beta_2']), 
                      eps = optimizer_dict['eps'], weight_decay = optimizer_dict['weight_decay'], lr = optimizer_dict['lr'])
    
    # schedule list
    schedule_dict = train_dict['schedule']
    # learning rate schedule
    lr_dict = schedule_dict['lr']
    lr_schedule = CosineAnnealingLR(optimizer, T_max = lr_dict['length'], eta_min = lr_dict['end'])

    # checkpoint
    checkpoint_dict = train_dict['checkpoint']
    checkpoint_pos = checkpoint_dict['folder']
    max_saved_checkpoints = checkpoint_dict['max_size']

    '''
    TODO load model
    update_cnt = 0
    if train_ctrl_dict['continue_train']:
        param_averager, kl_schedule, temperature_schedule, update_cnt = DVAE_Train.load_model(train_ctrl_dict['checkpoint'], model, optimizer, lr_schedule)
    '''
    
    # model DDP
    model = DDP(model, device_ids = [local_rank], output_device = local_rank, find_unused_parameters = True) #! notice find_unused_parameters
    logger.debug(f'model of rank {rank} at {model.device}')

    # dataset
    data_dict = config['data']
    dataset_dict = data_dict['dataset']
    # answer2label mapper
    with open(data_dict['answer2label_json'], 'r') as f:
       answer2label = json.load(f)
    # train
    train_dataset_dict = dataset_dict['train']
    train_dataset_dict['answer2label'] = answer2label
    train_dataset = VQA2Dataset(**train_dataset_dict)
    # val
    val_dataset_dict = dataset_dict['val']
    val_dataset_dict['answer2label'] = answer2label
    val_dataset = VQA2Dataset(**val_dataset_dict)
    
    # dataloader
    dataloader_dict = data_dict['dataloader']
    # train
    train_loader_dict = dataloader_dict['train']
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size = train_loader_dict['bs'], sampler = train_sampler, num_workers = train_loader_dict['num_workers'])
    # val
    val_loader_dict = dataloader_dict['train']
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size = val_loader_dict['bs'], sampler = val_sampler, num_workers = val_loader_dict['num_workers'])
    
    # trainer
    trainer = VQA_Train(
        model = model,
        rank = rank,
        world_size = torch.distributed.get_world_size(),
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        lr_scheduler = lr_schedule,
        checkpoint_pos = checkpoint_pos,
        max_saved_checkpoints = max_saved_checkpoints,
        device = device,
        use_lora = use_lora
    )
    trainer.train(s_epoch, e_epoch, 0, True, 1)
