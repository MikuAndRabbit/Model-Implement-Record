from typing import List, Optional, Tuple
from data import DVAE_Dataset
from modeling_discrete_vae import DiscreteVAE
from utils import EMA, cosine_scheduler
import torch
import torch.nn as nn
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from loguru import logger
from collections import deque
from tqdm import tqdm
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import random


def get_all_image_paths(folder_path: str, image_suffixes: List[str] = ['jpg', 'jpeg', 'png'], break_through: bool = False) -> List[str]:
    res = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if break_through or file.split('.')[-1] in image_suffixes:
                res.append(os.path.join(root, file))
        break
    return res


class DVAE_Train():
    def __init__(self, *, model: nn.Module, rank: int, world_size: int, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], optimizer: Optimizer, lr_scheduler: LRScheduler, 
                 param_averager: EMA, kl_schedule: np.ndarray, temperature_schedule: np.ndarray, checkpoint_pos: Optional[str], max_saved_checkpoints: Optional[int]) -> None:
        # model
        self.model = model
        # dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # optimizer & scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.param_averager = param_averager
        # schedule
        self.kl_schedule = kl_schedule
        self.temperature_schedule = temperature_schedule
        self.kl_schedule_len = len(self.kl_schedule)
        self.temperature_schedule_len = len(self.temperature_schedule)
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


    def get_current_schedule(self, break_through: bool = False, **kl_temperature):
        if break_through:
            assert len(kl_temperature) == 2
            return kl_temperature['kl'], kl_temperature['temperature']
        torch.distributed.all_gather(self.world_updates, self.update_cnt)
        update_cnt = 0
        for cnt in self.world_updates:
            update_cnt += cnt.item()
        kl_index = min(self.kl_schedule_len - 1, update_cnt)
        temperature_index = min(self.temperature_schedule_len - 1, update_cnt)
        logger.debug(f'update cnt {update_cnt} <- {self.rank} process')
        return self.kl_schedule[kl_index], self.temperature_schedule[temperature_index]


    def save_model(self, epoch: int, **others):
        assert self.checkpoint_pos is not None
        # make saved dict
        saved_dict = {
            'model': self.model.module.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'param_averager': (self.param_averager.shadow, self.param_averager.mu),
            'kl_schedule': self.kl_schedule,
            'temperature_schedule': self.temperature_schedule,
            'update_cnt': self.update_cnt
        }
        saved_dict.update(others)
        
        # save the checkpoint
        saved_path = os.path.join(self.checkpoint_pos, f'{epoch}_{self.update_cnt}.pth')
        torch.save(saved_dict, saved_path)
        logger.debug(f'Detailed information have been saved to {saved_path}')
        
        # check the number of checkpoints
        self.saved_checkpoint_path.append(saved_path)
        if self.max_saved_checkpoints is not None and len(self.saved_checkpoint_path) > self.max_saved_checkpoints:
            poped_checkpoint_path = self.saved_checkpoint_path.popleft()
            if os.path.exists(poped_checkpoint_path):
                os.remove(poped_checkpoint_path)
                logger.debug(f'The number of checkpoints exceed, {poped_checkpoint_path} have been removed!')


    @staticmethod
    def load_model(checkpoint_path: str, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, map_location = torch.device('cuda')):
        assert checkpoint_path is not None
        checkpoint_path = os.path.abspath(checkpoint_path)
        saved_dict = torch.load(checkpoint_path, map_location = map_location)
        # load dict
        model.load_state_dict(saved_dict['model'])
        optimizer.load_state_dict(saved_dict['optimizer'])
        lr_scheduler.load_state_dict(saved_dict['lr_scheduler'])
        # EMA
        shadow, mu = saved_dict['param_averager']
        param_averager = EMA(mu, shadow)
        return param_averager, saved_dict['kl_schedule'], saved_dict['temperature_schedule'], saved_dict['update_cnt']


    def val(self, epoch: int):
        total_loss, total_cnt = .0, 0
        for idx, images in tqdm(enumerate(self.val_dataloader)):
            if self.master_rank: logger.trace(f'Valuation    epoch: {epoch}, batch index: {idx}')
            # epoch setting
            self.val_dataloader.sampler.set_epoch(1)
            # superparameter
            kl, temperature = self.get_current_schedule(break_through = True, kl = 0., temperature = 1.)
            with torch.no_grad():
                loss = self.model(img = images, temperature = temperature, kl_div_loss_weight = kl, return_loss = True)
            total_loss += loss.cpu().item()
            total_cnt += images.shape[0]
        
        # rank loss info
        logger.debug(f'Valuation    epoch: {epoch}, rank id: {self.rank}, items: {total_cnt}, loss: {total_loss}')
        
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
                    logger.debug(f'Valuation    previous best loss: {self.best_val_loss} ({self.best_val_epoch}) -> current best loss: {val_loss} ({epoch})')
                    self.best_train_loss = val_loss
                    self.best_val_epoch = epoch
                    if self.checkpoint_pos is not None and epoch not in self.saved_epoch_idx:
                        self.save_model(epoch)
                        self.saved_epoch_idx.add(epoch)
            

    def _train(self, epoch: int):
        proc_epoch_cnt = 0
        proc_epoch_loss = .0
        for idx, images in tqdm(enumerate(self.train_dataloader)):
            if self.master_rank: logger.trace(f'Train    epoch: {epoch}, batch index: {idx}')
            # epoch setting
            self.train_dataloader.sampler.set_epoch(epoch)
            # superparameter            
            kl, temperature = self.get_current_schedule(break_through = True, kl = 0., temperature = 1.)
            proc_epoch_cnt += images.shape[0]
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # type: ignore
                loss = self.model(img = images, temperature = temperature, kl_div_loss_weight = kl, return_loss = True)
            # statics
            proc_epoch_loss += loss.cpu().item()
            self.update_cnt += 1
            # update model
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.param_averager.update(name, param.data)
                    
        # rank epoch loss info
        logger.debug(f'Train    epoch: {epoch}, rank id: {self.rank}, items: {proc_epoch_cnt}, loss: {proc_epoch_loss}')
        
        # checkpoint
        loss_cnt_tensor = torch.tensor((proc_epoch_loss, proc_epoch_cnt), dtype = torch.float32).cuda().detach()
        torch.distributed.all_gather(self.world_train_loss, loss_cnt_tensor)
        if self.master_rank:
            # get total loss
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
                    logger.debug(f'Train    previous best loss: {self.best_train_loss} ({self.best_train_epoch}) -> current best loss: {total_epoch_loss} ({epoch})')
                    self.best_train_loss = total_epoch_loss
                    self.best_train_epoch = epoch
                    if self.checkpoint_pos is not None and self.master_rank:
                        self.save_model(epoch)
                        self.saved_epoch_idx.add(epoch)


    def train(self, s_epoch: int, e_epoch: int, update_cnt : Optional[int] = None, need_val: bool = True, val_epoch_interval: int = 5):
        if update_cnt is not None:
            self.update_cnt = torch.tensor(update_cnt, dtype = torch.int64).cuda().detach()
        if self.master_rank: logger.info(f'Start training {s_epoch} - {e_epoch}')
        for epoch in range(s_epoch, e_epoch + 1):
            if self.master_rank: logger.info(f'Start training of {epoch} epoch.')
            self._train(epoch)
            if need_val and epoch % val_epoch_interval == 0:
                if self.master_rank: logger.info(f'Start valuation of {epoch} epoch.')
                self.val(epoch)


def init_seeds(seed=0, cuda_deterministic=True):
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
    CONFIG_PATH = r'/gly/guogb/lym/code/dVAE/config.yaml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # logger setting
    logger_dict = config['log']
    logger.remove()
    logger.level(name = 'TRAIN', no = 20)
    logger.level(name = 'VAL', no = 20)
    logger_file_folder = logger_dict['path']['folder']
    roration = logger_dict['rotation']
    logger.add(sink = os.path.join(logger_file_folder, 'info.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "INFO")
    logger.add(sink = os.path.join(logger_file_folder, 'debug.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "DEBUG")
    logger.add(sink = os.path.join(logger_file_folder, 'trace.log'), rotation=roration, compression="zip", enqueue=True, level="TRACE")
    logger.add(sink = os.path.join(logger_file_folder, 'train.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "TRAIN")
    logger.add(sink = os.path.join(logger_file_folder, 'val.log'), rotation=roration, compression="zip", enqueue=True, filter=lambda record: record["level"].name == "VAL")

    # bash args parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    ARGS = parser.parse_args()
    local_rank = ARGS.local_rank

    # DDP backend init
    train_dict = config['train']
    distributed_dict = train_dict['distributed']
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend = distributed_dict['backend'])  # type: ignore
    device = torch.device(train_dict['device'], local_rank)
    
    # set a seed of random algorithm
    rank = torch.distributed.get_rank()
    init_seeds(seed = 1 + rank)
    
    # basic model & init
    model_dict = config['model']
    model = DiscreteVAE(**model_dict).to(device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    # optimizer
    optimizer_dict = train_dict['optimizer']
    optimizer = AdamW(model.parameters(), betas = (optimizer_dict['beta_1'], optimizer_dict['beta_2']), 
                      eps = optimizer_dict['eps'], weight_decay = optimizer_dict['weight_decay'], lr = optimizer_dict['lr'])
    
    # schedule list
    schedule_dict = train_dict['schedule']
    # kl
    kl_schedule_dict = schedule_dict['kl']
    kl_schedule = cosine_scheduler(base_value = kl_schedule_dict['start'], final_value = kl_schedule_dict['end'], 
                                   max_length = kl_schedule_dict['length'])
    # temperature
    temperature_schedule_dict = schedule_dict['temperature']
    temperature_schedule = cosine_scheduler(base_value = temperature_schedule_dict['start'], final_value = temperature_schedule_dict['end'], 
                                            max_length = temperature_schedule_dict['length'])
    # learning rate
    lr_dict = schedule_dict['lr']
    lr_schedule = CosineAnnealingLR(optimizer, T_max = lr_dict['length'], eta_min = lr_dict['end'])

    # device & checkpoint
    checkpoint_dict = train_dict['checkpoint']
    checkpoint_pos = checkpoint_dict['folder']
    max_saved_checkpoints = checkpoint_dict['max_size']
    
    # train control
    train_ctrl_dict = train_dict['control']
    s_epoch = train_ctrl_dict['s_epoch']
    e_epoch = train_ctrl_dict['e_epoch']

    # load model
    update_cnt = 0
    if train_ctrl_dict['continue_train']:
        param_averager, kl_schedule, temperature_schedule, update_cnt = DVAE_Train.load_model(train_ctrl_dict['checkpoint'], model, optimizer, lr_schedule)
    
    # model DDP
    model = DDP(model, device_ids = [local_rank], output_device = local_rank)
    
    # param
    if not train_ctrl_dict['continue_train']:
        decay_coefficient = schedule_dict['param']['decay_coefficient']
        param_averager = EMA(mu = decay_coefficient)
    # param register
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_averager.register(name, param.data)

    # data
    # train
    train_data_dict = train_dict['data']
    train_dataset = DVAE_Dataset(get_all_image_paths(train_data_dict['folder']), model_dict['image_size'], False)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size = train_data_dict['bs'], sampler = train_sampler, num_workers = train_data_dict['num_workers'])
    # val
    val_dict = config['val']
    val_data_dict = val_dict['data']
    val_dataset = DVAE_Dataset(get_all_image_paths(val_data_dict['folder']), model_dict['image_size'], False)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size = val_data_dict['bs'], sampler = val_sampler, num_workers = val_data_dict['num_workers'])
    
    # Trainer
    trainer = DVAE_Train(model = model, rank = torch.distributed.get_rank(), world_size = torch.distributed.get_world_size(), train_dataloader = train_dataloader, val_dataloader = val_dataloader, optimizer = optimizer,
                         lr_scheduler = lr_schedule, param_averager = param_averager, kl_schedule = kl_schedule, temperature_schedule = temperature_schedule,
                         checkpoint_pos = checkpoint_pos, max_saved_checkpoints = max_saved_checkpoints)
    trainer.train(s_epoch, e_epoch, update_cnt, val_dict['need'], val_dict['interval'])
