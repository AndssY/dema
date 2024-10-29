"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import time

from tqdm import tqdm
from mamba_ssm.utils.generation import InferenceParams
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def sample2(logits, temperature=1, top_k=3, is_prob=True):
    # logits: [B, L, D]
    # Scale logits by temperature
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    logits = logits / temperature
    # Optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # Apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # Sample from the distribution or take the most likely
    if is_prob:
        ix = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(logits.size(0), logits.size(1), 1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    # Return sampled indexes
    return ix

def top_k_logits(logits, k):
    B, L, D = logits.size()
    # Compute top k values and their indices along the last dimension
    v, ix = torch.topk(logits, k, dim=-1)
    # Clone logits to not modify the original values 
    out = logits.clone()
    # Create a broadcastable shape for comparison
    broadcast_shape = list(out.shape)
    broadcast_shape[-1] = 1
    # Values that don't make the cut are set to -inf
    out[out < v[..., [-1]]] = -float('Inf')
    return out

def normalized_atari_score(env, score):
    gamer_score = {'Breakout':30, 'Qbert':13455, 'Pong':15, 'Seaquest':42055, 'Asterix':8503, 'Frostbite':4335, 'Assault':742, 'Gopher':2412}
    random_score = {'Breakout':2, 'Qbert':164, 'Pong':-21, 'Seaquest':68, 'Asterix':210, 'Frostbite':65, 'Assault':222, 'Gopher':258}
    return (score-random_score[f'{env}'])/(gamer_score[f'{env}']-random_score[f'{env}'])

def padding_collate(batch):
    data, actions, rtgs, timesteps = zip(*batch)
    num_trajectories = len(data)
    len_state = [xx.shape[0] for xx in data]
    max_len = max(len_state)

    padded_data = pad_sequence([torch.tensor(d, dtype=torch.float32) for d in data],
                                batch_first=True,
                                padding_value=0)
    padded_rtgs = pad_sequence([torch.tensor(r, dtype=torch.float32) for r in rtgs],
                                batch_first=True,
                                padding_value=-1000).unsqueeze(-1)
    padded_actions = pad_sequence([torch.tensor(action, dtype=torch.long) for action in actions],
                                    batch_first=True,
                                    padding_value=3)
    new_timesteps = torch.arange(max_len).unsqueeze(0).repeat(num_trajectories, 1)
    masks = torch.zeros((num_trajectories, max_len), dtype=bool)

    for i, _ in enumerate(padded_data):
        masks[i, :len_state[i]] = 1  # Mark valid data points
    # print(torch.max(padded_data))
    if torch.max(padded_data) > 1:
        padded_data = padded_data / 255.
    return padded_data, padded_actions, padded_rtgs, new_timesteps, masks

def seconds_format(time_cost: int):
    """
    :param time_cost: 
    :return: 
    """
    min = 60
    hour = 60 * 60
    day = 60 * 60 * 24
    if not time_cost or time_cost < 0:
        raise TypeError
    elif time_cost < min:
        return '%sS' % time_cost
    elif time_cost < hour:
        return '%sM%sS' % (divmod(time_cost, min))
    elif time_cost < day:
        cost_hour, cost_min = divmod(time_cost, hour)
        return '%sH%s' % (cost_hour, seconds_format(cost_min))
    else:
        cost_day, cost_hour = divmod(time_cost, day)
        return '%sD%s' % (cost_day, seconds_format(cost_hour))

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, model):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = model.module if hasattr(model, "module") else model
        print("saving path ", self.config.save_path + f'/models/{epoch}.pkl')
        torch.save(raw_model.state_dict(), self.config.save_path + f'/models/{epoch}.pkl')

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        train_start = time.time()
        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            if config.mamba_type == 'tr':
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=4)
            elif config.mamba_type == 'rnn':
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,collate_fn=padding_collate,
                                    num_workers=config.num_workers)       
            count = 0
            duration = []
            losses = []                         
            for it, (x, y, r, t, mask) in enumerate(loader):

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                if config.mamba_type == 'tr':
                    mask = mask.to(self.device)
                elif config.mamba_type == 'rnn':
                    cu_seqlens = mask.to(self.device)
                for nit in range(1, config.inner_it+1): 
                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        start_time = time.time()
                        # logits, loss = model(x, y, r)
                        if config.mamba_type == 'tr':
                            logits, loss = model(x, y, y, r, t, mask)
                        elif config.mamba_type == 'rnn':
                            logits, loss = model(x, y, y, r, t, mask=None, running=True, cu_seqlens=cu_seqlens)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        end_time = time.time()
                        count += 1
                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        if nit==1 or nit % 5 == 0:  # report progress
                            duration.append(end_time - start_time)
                            print(f"epoch {epoch + 1} iter {it} niter {nit}: train loss {loss.item():.5f}. lr {lr:e}, time {sum(duration)/len(duration)}")  

            if not is_train:
                test_loss = float(np.mean(losses))
                print("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        
        best_return = -float('inf')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.tokens = 0 # counter used for learning rate decay
        start_time = time.time()
        for epoch in range(config.epochs):
            start2_time = time.time()
            run_epoch('train', epoch_num=epoch)
            now_time = time.time()
            print(f'epoch {epoch+1} training cost time ', seconds_format(now_time-start2_time))
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                assert 'dt' in self.config.model_class
                eval_return = self.get_returns_dt(0)
            elif self.config.model_type == 'reward_conditioned':
                tr = {'Breakout':[90,150,180], "Seaquest":[1150,1450,1750], "Qbert":[8000,12000,14000], "Pong":[20], "Frostbite":[1200,1450,1700], "Assault":[1200,1500,1800], "Gopher":[6500], "Asterix":[900,1000,1100]}
                if self.config.mamba_type == 'tr':
                    eval_return = self.get_returns_dt(tr[f'{self.config.env}']) 
                elif self.config.mamba_type == 'rnn':
                    eval_return = self.get_returns_mamba(tr[f'{self.config.env}'])
            else:
                raise NotImplementedError()
            print(f'epoch {epoch+1} evaluating cost time ', seconds_format(time.time()-now_time))
            self.save_checkpoint(epoch, model)
        print('total training time is ', seconds_format(time.time()-start_time))
    def get_returns_dt(self, rets):
        self.model.train(False)
        for ret in rets:
            args=Args(self.config.env.lower(), self.config.seed)
            env = Env(args)
            env.eval()

            T_rewards, T_Qs = [], []
            done = True
            for i in range(self.config.num_eval_episodes):
                state = env.reset()
                state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                rtgs = [ret]
                # first state is from env, first rtg is target return, and first timestep is 0
                sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=torch.tensor([0],dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

                j = 0
                all_states = state
                actions = [0]
                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False
                    action = sampled_action.cpu().numpy()[0,-1]
                    actions += [sampled_action]
                    state, reward, done = env.step(action)
                    reward_sum += reward
                    j += 1

                    if done:
                        T_rewards.append(reward_sum)
                        break

                    state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                    all_states = torch.cat([all_states, state], dim=0)

                    rtgs += [rtgs[-1] - reward]
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                    # timestep is just current timestep
                    sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
            env.close()
            # print(T_rewards)
            eval_return = sum(T_rewards)/(self.config.num_eval_episodes)
            print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return

    def get_returns_mamba(self, rets):
        self.model.train(False)
        for ret in rets:
            args=Args(self.config.env.lower(), self.config.seed)
            env = Env(args)
            env.eval()

            T_rewards, T_Qs = [], []
            done = True
            for i in range(self.config.num_eval_episodes):
                state = env.reset()
                self.model.infer_params = InferenceParams(max_batch_size=1, max_seqlen=108e3)
                state = state.type(torch.float32).to('cuda')
                rtgs = [ret]
                action = torch.randint(0, self.config.vocab_size, (1,), device='cuda', dtype=torch.long)
                j = 0
                # all_states = state
                actions = []
                print_actions = []
                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False
                        self.model.infer_params = InferenceParams(max_batch_size=1, max_seqlen=108e3)
                        state = state.type(torch.float32).to('cuda')
                    if hasattr(self.model, "module"):
                        action_logits = self.model.module.get_action(
                            state,
                            actions=torch.tensor(action, dtype=torch.long).to('cuda').reshape(1, 1),
                            rtgs=torch.tensor(rtgs[-1], dtype=torch.float32).to('cuda').reshape(1, 1, 1),
                        )
                    else:
                        action_logits = self.model.get_action(
                            state,
                            actions=torch.tensor(action, dtype=torch.long).to('cuda').reshape(1, 1),
                            rtgs=torch.tensor(rtgs[-1], dtype=torch.float32).to('cuda').reshape(1, 1, 1),
                        ) 
                    sampled_action = sample2(action_logits)
                    action = int(sampled_action.cpu().numpy()[0])
                    actions += [sampled_action]
                    state, reward, done = env.step(action)
                    print_actions += [action]
                    state = state.type(torch.float32).to('cuda')
                    reward_sum += reward
                    j += 1

                    if done:
                        T_rewards.append(reward_sum)
                        break

                    state = state.to('cuda')

                    # all_states = torch.cat([all_states, state], dim=0)

                    rtgs += [rtgs[-1] - reward]
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                    # timestep is just current timestep
            env.close()
            print(T_rewards)
            eval_return = sum(T_rewards)/(self.config.num_eval_episodes)
            print("target return: %d, eval return: %d, normalized score: %s" % (ret, eval_return, normalized_atari_score(self.config.env, eval_return)))
            # print('game length is ',len(print_actions), 'game action is ', print_actions)


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(10)
        # return self.env.render()
    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
