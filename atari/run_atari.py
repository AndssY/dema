import csv
import logging
import os
import sys
import time
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from configs.config import MambaConfig, Ds4Config, VmConfig
from models.mamba import DMamba
from models.dt import GPT
from models.ds4 import DS4
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
sys.path.append("..")
sys.path.append(os.pardir)


def _to_str(num):
    if num >= 1e6:
        return f'{(num / 1e6):.2f} M'
    else:
        return f'{(num / 1e3):.2f} k'


def param_to_module(param):
    if param == '':
        return 'Top Level Model'
    module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        module_str = f'{modules.get(module, "N/A")}' if module != 'Top Level Model' else 'Top Level Model'
        print(' ' * 8, f'{key:10}: {_to_str(count)} | {module_str}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(' ' * 8, f'... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = np.insert(done_idxs, 0, 0)
        
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, 1


class TrajectoryDataset(Dataset):
    def __init__(self, data, pct_traj, actions, done_idxs, rtgs, timesteps):
        self.pct_traj = pct_traj
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = np.insert(done_idxs, 0, 0)
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.done_idxs) - 1

    def get_total_steps(self):
        return int(self.done_idxs[-1])

    def __getitem__(self, idx):
        start_id = self.done_idxs[idx]
        end_id = self.done_idxs[idx+1]
        
        states = torch.tensor(np.array(self.data[start_id:end_id]), dtype=torch.float32).reshape(end_id-start_id, -1)
        states = states / 255.
        actions = torch.tensor(self.actions[start_id:end_id], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[start_id:end_id], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[start_id:start_id+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

class FileRedirector(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "a", encoding="utf-8")
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.file.flush()

def experiment(exp_prefix,variant):
    device = variant.get('device', 'cuda')
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    env_name = variant['env']
    model_class = variant['model_class']
    group_name = f'{exp_prefix}-{env_name}'
    exp_prefix = f'{current_time}-{group_name}'
    exp_path = './' + exp_prefix
    exp_path = os.path.join(variant['save_prefix'], exp_path)

    # print(exp_path)
    os.makedirs(exp_path, exist_ok=True)
    save_path = exp_path + '/models/'
    os.makedirs(save_path, exist_ok=True)

    variant['exp_path'] = exp_path
    sys.stdout = FileRedirector(f'{exp_path}/out.log')
    sys.stderr = FileRedirector(f'{exp_path}/out.log')
    print(f"Exporting log to {exp_path}")
    print('Now time is ', current_time)

    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(variant['num_buffers'], 
        variant['num_steps'], variant['env'], variant['data_dir_prefix'], variant['trajectories_per_buffer'])
    
    if variant['model_class'] == 'mamba':
        if variant['mamba_type'] == 'rnn':
            train_dataset = TrajectoryDataset(obss, variant['pct_traj'], actions, done_idxs, rtgs, timesteps)
            conf = MambaConfig(train_dataset.vocab_size, train_dataset.block_size,
                            n_layer=6, max_timestep=max(timesteps),**variant)
            conf.final_tokens = train_dataset.get_total_steps()
        elif variant['mamba_type'] == 'tr':
            train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
            conf = MambaConfig(train_dataset.vocab_size, train_dataset.block_size, final_tokens=2*len(train_dataset)*args.context_length*3,
                            n_layer=6, max_timestep=max(timesteps),**variant)
        
        model = DMamba(conf)

    elif variant['model_class'] == 'dt':
        train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
        conf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                        n_layer=6, n_head=8, n_embed=128, model_class=args.model_class, max_timestep=max(timesteps))
        # filtered_variant = {k: v for k, v in variant.items() if k not in conf}
        conf = GPTConfig(
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size,
            n_layer=6,
            n_head=8,
            n_embed=128,
            max_timestep=max(timesteps),
            **variant
        )
        conf.betas=(0.9, 0.95)
        conf.inner_it=1
        conf.warmup_tokens=512*20
        conf.final_tokens=2*len(train_dataset)*args.context_length*3
        model = GPT(conf)
    elif variant['model_class'] == 'ds4':
        train_dataset = TrajectoryDataset(obss, variant['pct_traj'], actions, done_idxs, rtgs, timesteps)
        conf = Ds4Config(train_dataset.vocab_size, **variant)
        conf.final_tokens = train_dataset.get_total_steps()
        conf.betas=(0.9, 0.95)
        conf.warmup_tokens=512*20
        conf.final_tokens=2*len(train_dataset)*args.context_length*3
        model = DS4(conf)
    else:
        raise NotImplementedError
    conf.save_path = exp_path

    report_parameters(model)
    print("ARGS ON RUN")
    for k in dir(conf):
        # Filter out special and private attributes
        if not k.startswith("_") and not callable(getattr(conf, k)):
            print(f"{k:20} :: {getattr(conf, k)}")
    print("ARGS END")

    # initialize a trainer instance and kick off training
    # epochs = args.epochs
    # tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
    #                     lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
    #                     num_workers=4, seed=args.seed, model_class=args.model_class, game=args.game, max_timestep=max(timesteps))
    trainer = Trainer(model, train_dataset, None, conf)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--env', type=str, default='Breakout')
    parser.add_argument('--save_prefix', type=str, default='test')
    parser.add_argument('--data_dir_prefix', type=str, default='/home/.d4rl/atari_data/dqn_replay/')
    parser.add_argument('--device', type=str, default='cuda')

    # model
    parser.add_argument('--model_class', type=str, default='mamba') # ds4, mamba, vmamba, dt
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--mamba_type', type=str, default='tr') # tr, rnn
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_tokens', type=int, default=3000*32)

    # training
    parser.add_argument('--context_length', type=int, default=30)
    # parser.add_argument('--block_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pct_traj', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--inner_it', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--grad_norm_clip', type=float, default=0.8)

    # eval
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--load_model', type=str, default='none')

    args = parser.parse_args()
    set_seed(vars(args)['seed'])
    experiment('dmamba', variant=vars(args))
