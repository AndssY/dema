import os
import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.ssm import Mamba_mujoco

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
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

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)


    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

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
    )
    report_parameters(model)

    model = model.to(device=device)
    report_parameters(model)

    dummy_states = torch.randn(1, K, state_dim, dtype=torch.float32).to(device)
    dummy_acts = torch.randn(1, K, act_dim, dtype=torch.float32).to(device)
    dummy_rewards = torch.randn(1, K, 1, dtype=torch.float32).to(device)
    dummy_rtg = torch.randn(1, K, 1, dtype=torch.float32).to(device)
    dummy_timesteps = torch.randint(0, 1000, (1, K), dtype=torch.long).to(device)

    # dummy_attn_masks = torch.bernoulli(torch.full((1, K), 0.5)).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model.forward(
                dummy_states, dummy_acts, dummy_rewards, dummy_rtg, dummy_timesteps)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model.forward(
                dummy_states, dummy_acts, dummy_rewards, dummy_rtg, dummy_timesteps)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium-expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--run_type', type=str, default='train')  # train, evaluation, render
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str,
                        default='mamba')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--load_model', type=str, default='none')
    # dt, s4d, bc, mamba
    # model config for mamba
    parser.add_argument('--d_model', type=int, default=256)  # 没用了
    parser.add_argument('--embedding_dim', type=int, default=128)  # 变成了d_model
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)  # not used
    parser.add_argument('--residual_in_fp32', type=bool, default=True)
    parser.add_argument('--rms_norm', type=bool, default=True)
    parser.add_argument('--fused_add_norm', type=bool, default=True)
    parser.add_argument('--ssm_cfg', type=dict, default={})
    parser.add_argument('--activation_function', type=str, default='relu')  # not used? # TODO 跟踪一下
    parser.add_argument('--dropout', type=float, default=0.1)  # not used? mamba里也有类似的
    # train
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_steps_per_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)

    args = parser.parse_args()
    experiment('dmamba2', variant=vars(args))
