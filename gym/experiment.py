import copy
from functools import partial
import json
import time

import d4rl
import gym
import numpy as np
import torch
import wandb
import os
import argparse
import pickle
import random
import sys
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_rtg_parallel
from decision_transformer.models.ssm import Mamba_mujoco
from decision_transformer.training.seq_trainer import SequenceTrainer
import matplotlib.pyplot as plt

import sys
from envpool.registration import make_gym

class FileRedirector(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.file.flush()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def _to_str(num):
    if num >= 1e6:
        return f'{(num / 1e6):.2f} M'
    else:
        return f'{(num / 1e3):.2f} k'


def param_to_module(param):
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
        print(' ' * 8, f'{key:10}: {_to_str(count)} | {modules[module]}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(' ' * 8, f'... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters


def experiment(
        exp_prefix,
        variant,
):

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
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
    print('NOW TIME IS ', current_time)
    print("ARGS ON RUN")
    for k in variant.keys():
        print(f"{k:20}:: {variant[k]}")
    print("ARGS END")
    
    if variant['eval_parallel']:
        make_env = partial(make_gym, num_envs=variant['num_eval_episodes'], seed=0)
    else:
        make_env = partial(gym.make)

    if env_name == 'hopper':
        env = make_env('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 2700, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = make_env('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 9000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = make_env('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 3750, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data_mujoco/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    nor_states = (states - state_mean) / state_std
    # state_bound = [np.min(nor_states), np.max(nor_states)]
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        goals = None
        return s, a, r, d, rtg, timesteps, mask, goals

    def get_batch_recurrent(batch_size=256):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        max_len = max([traj_lens[sorted_inds[batch_inds[i]]] for i in range(batch_size)])
        max_len = max(max_len, 3)
        # print(f"LOG max len for batch: {max_len}")
        startpoint = 0
        new_len = int(1.0 * pct_traj * max_ep_len)
        if pct_traj < 1 and max_len > new_len:
            startpoint = max_len - new_len
            max_len = new_len
            # print(f"LOG max len for batch (new): {max_len}")

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        goals = []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if pct_traj < 1:
                si = random.randint(0, max(0, len(traj['observations']) - max_len))
            else:
                si = 0

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([s[-1], np.zeros((1, max_len - tlen, state_dim))], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([a[-1], np.zeros((1, max_len - tlen, act_dim))], axis=1)
            r[-1] = np.concatenate([r[-1], np.zeros((1, max_len - tlen, 1))], axis=1)
            d[-1] = np.concatenate([d[-1], np.ones((1, max_len - tlen)) * 2], axis=1)
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, max_len - tlen, 1))], axis=1) / scale
            timesteps[-1] = np.concatenate([timesteps[-1], np.zeros((1, max_len - tlen))], axis=1)
            mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1))
            got_goals = traj.get('infos/goal', None)
            if got_goals is not None:
                goals.append((got_goals[5, :] - state_mean[:2]) / state_std[:2])
            else:
                goals.append(got_goals)

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        if goals[0] is None:
            goals = None
        else:
            goals = torch.from_numpy(np.concatenate([z.reshape((1, 2)) for z in goals], axis=0)).to(dtype=torch.float32,
                                                                                                    device=device)
        return s, a, r, d, rtg, timesteps, mask, goals

    def eval_episodes(target_rew, parallel=False):
        parallel = bool(parallel)
        def fn(model):
            returns, lengths, all_average_diff, all_last_action_diff = [], [], [], []
            # ids = np.arange(variant['num_eval_episodes'])
            # reward = np.zeros(variant['num_eval_episodes'])
            # lengths = np.zeros(variant['num_eval_episodes'])
            if parallel == True:
                with torch.no_grad():
                    if model_type == 'dt' or 'mamba' in model_type:
                        returns, lengths = evaluate_episode_rtg_parallel(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            num_envs=variant['num_eval_episodes'],
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            )
                    else:
                        return NotImplemented
            else:
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        if model_type == 'dt' or 'mamba' in model_type:
                            if variant['run_type'] != 'train':
                                env.seed(variant['seed']+num_eval_episodes)
                                env.action_space.seed(variant['seed']+num_eval_episodes)

                            ret, length = evaluate_episode_rtg(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=scale,
                                target_return=target_rew / scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                                model_type=variant['model_type'],
                                attention=variant['attention'],
                                start_time=variant['start_time'],
                                end_time=variant['end_time']
                            )
                        else:
                            ret, length = evaluate_episode(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                target_return=target_rew / scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                    returns.append(ret)
                    lengths.append(length)
            print(returns)
            print(lengths)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'mamba':
        model = Mamba_mujoco(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            embedding_dim=variant['embedding_dim'],
            d_model=variant['d_model'],
            n_layer=variant['n_layer'],
            ssm_cfg=variant['ssm_cfg'],
            rms_norm=variant['rms_norm'],
            residual_in_fp32=variant['residual_in_fp32'],
            fused_add_norm=variant['fused_add_norm'],
            dropout = variant['dropout'],
            reward_type = variant['reward_type'],
            time_embd = variant['time_embd']
        )
    elif model_type == 'dt':
        from decision_transformer.models.decision_transformer import DecisionTransformer
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embedding_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embedding_dim'],
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    report_parameters(model)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    model_initial = copy.deepcopy(model)

    model_final = None
    if variant['load_model'] != "none" and variant['run_type'] != 'train':
        
        file_modl = variant['load_model']
        loadedfile = torch.load(file_modl)
        if model_type == 'mamba':
            model.load_state_dict(loadedfile['model'])
        elif model_type == 'dt':
            model.load_state_dict(loadedfile['model_state_dict'])
        print(f"Using model from {file_modl}, model type is {model_type}")
        optimizer = loadedfile['optimizer']
        for g in optimizer.param_groups:
            g['lr'] = variant['learning_rate']
        model_final = copy.deepcopy(model)
        try:
            del model_final.embed_timestep
        except:
            pass
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    model = model.to(device=device)
    print(f'training model type is {type(model)}')
    if model_type == 'mamba' or model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar, variant['eval_parallel']) for tar in env_targets],
        )
    if log_to_wandb and variant['run_type'] != 'render':
        wandb.init(
            name=exp_path,
            group=group_name,
            project=f"{variant['run_type']}-{variant['model_type']}",
            dir=exp_path,
            config=variant
        )

    if variant['run_type'] == 'train':
        best_result = [-10000 for _ in env_targets] 
        now_best_idx = [0 for i in range(len(best_result))]
        for iter in range(variant['max_iters']):
            outputs, res = trainer.train_iteration(
                num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
            if len(res) > 0:
                for i in range(len(env_targets)):
                    if res[i] > best_result[i]:
                        now_best_idx[i] = iter
                        best_result[i] = max(best_result[i], res[i])
                torch.save({
                    # 'model': model,
                    'model': model.state_dict(),
                    'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler': scheduler,
                }, f"{save_path}/model_iter{iter}.pkl")
            if log_to_wandb:
                wandb.log(outputs)
        print('=' * 80)
        print('the best result during training is ', best_result)
        print('the best normalized score is ', [d4rl.get_normalized_score(f"{variant['env']}-medium-v2", i)
                                                for i in best_result])
        print('=' * 80)
        torch.save({
            # 'model': model,
            'model': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler': scheduler,
        }, f"{save_path}/final_model_iter{iter}.pkl")
    # cannot reproduct the training result due to the seed control of envpool
    if variant['eval_parallel']:
        make_env = partial(gym.make)
        if env_name == 'hopper':
            env = make_env('Hopper-v3')
            max_ep_len = 1000
            env_targets = [3600, 1800]  # evaluation conditioning targets
            scale = 1000.  # normalization for rewards/returns
        elif env_name == 'halfcheetah':
            env = make_env('HalfCheetah-v3')
            max_ep_len = 1000
            env_targets = [12000, 6000]
            scale = 1000.
        elif env_name == 'walker2d':
            env = make_env('Walker2d-v3')
            max_ep_len = 1000
            env_targets = [5000, 2500]
            scale = 1000.
        elif env_name == 'reacher2d':
            from decision_transformer.envs.reacher_2d import Reacher2dEnv
            env = Reacher2dEnv()
            max_ep_len = 100
            env_targets = [76, 40]
            scale = 10.
        else:
            raise NotImplementedError
        best_idx = max(now_best_idx)
        variant['load_model'] = f"{save_path}/model_iter{best_idx}.pkl"
        
        file_modl = variant['load_model']
        loadedfile = torch.load(file_modl)
        model.load_state_dict(loadedfile['model'])
        print(f"Using model from {file_modl}, model type is {model_type}")
        optimizer = loadedfile['optimizer']
        for g in optimizer.param_groups:
            g['lr'] = variant['learning_rate']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        model = model.to(device=device)
        print(f'training model type is {type(model)}')
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar, 1) for tar in env_targets],
        )
        trainer.eval_step(print_logs=True, env=variant['env'])

    elif variant['run_type'] == 'evaluation':
        trainer.eval_step(print_logs=True, env=variant['env'])       

    elif variant['run_type'] == 'render':
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--reward_type', type=str, default='normal') # K_term, normal
    parser.add_argument('--time_embd', type=bool, default=False)
    parser.add_argument('--attention', type=str, default='step') # trig, step
    parser.add_argument('--start_time', type=int, default=20) # trig, step
    parser.add_argument('--end_time', type=int, default=800) # trig, step
    # must be with run_type==evaluation
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--run_type', type=str, default='evaluation')  # train, evaluation, render
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str,
                        default='mamba')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--load_model', type=str, default='none')
    parser.add_argument('--save_prefix', type=str, default='attention')
    # dt, s4d, bc, mamba
    # model config for mamba
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)  # not used
    parser.add_argument('--residual_in_fp32', type=bool, default=True)
    parser.add_argument('--rms_norm', type=bool, default=True)
    parser.add_argument('--fused_add_norm', type=bool, default=True)
    parser.add_argument('--ssm_cfg', type=json.loads, default={"d_state":64, "expand":2})
    parser.add_argument('--dropout', type=float, default=0.1)
    # train
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--eval_parallel', type=int, default=0)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()
    if args.run_type != 'train':
        set_seed(vars(args)['seed'])
    for _ in range(args.repeat_num):
        experiment('dmamba', variant=vars(args))
