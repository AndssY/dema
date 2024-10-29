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

def swap_two_model_params(idx, swap_names, model_parameters):

    now_model_parameters = copy.deepcopy(model_parameters[idx])
    for swap_name in swap_names:
        now_model_parameters[swap_name] = model_parameters[1-idx][swap_name]
    return now_model_parameters

def change_model_params(idx, names, model_parameters, method='random'):
    now_model_parameters = copy.deepcopy(model_parameters[idx])
    for name in names:
        shape = now_model_parameters[name].shape
        if method == 'random':
            random_matrix = torch.rand(shape)
            # random_matrix_normal
            now_model_parameters[name] = random_matrix
        elif method == 'zero':
            zeros_matrix = torch.zeros_like(now_model_parameters[name])
            now_model_parameters[name] = zeros_matrix
    return now_model_parameters
def experiment(
        exp_prefix,
        variant,
):

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    env_names = [variant['env'], variant['env_b']]
    dataset = [variant['dataset'], variant['dataset_b']]

    model_type = variant['model_type']
    group_name = f'{exp_prefix}'
    exp_prefix = f'{current_time}-{group_name}'
    exp_path = './' + exp_prefix
    exp_path = os.path.join(variant['save_prefix'], exp_path)
    print(exp_path)
    os.makedirs(exp_path, exist_ok=True)

    variant['exp_path'] = exp_path
    sys.stdout = FileRedirector(f'{exp_path}/out.log')
    sys.stderr = FileRedirector(f'{exp_path}/out.log')
    print(f"Exporting log to {exp_path}")
    print('Now time is ', current_time)
    print("ARGS ON RUN")
    for k in variant.keys():
        print(f"{k:20}:: {variant[k]}")
    print("ARGS END")
    

    make_env = partial(gym.make)
    env, max_ep_len,env_targets,scale = [],[],[],[]
    
    for env_name in env_names:
        if env_name == 'hopper':
            env.append(make_env('Hopper-v3'))
            max_ep_len.append(1000)
            env_targets.append([3600, 2700, 1800])  # evaluation conditioning targets
            scale.append(1000.)  # normalization for rewards/returns
        elif env_name == 'halfcheetah':
            env.append(make_env('HalfCheetah-v3'))
            max_ep_len.append(1000)
            env_targets.append([12000, 9000, 6000])
            scale.append(1000.)
        elif env_name == 'walker2d':
            env.append(make_env('Walker2d-v3'))
            max_ep_len.append(1000)
            env_targets.append([5000, 3750, 2500])
            scale.append(1000.)
        elif env_name == 'reacher2d':
            from decision_transformer.envs.reacher_2d import Reacher2dEnv
            env.append(Reacher2dEnv())
            max_ep_len.append(100)
            env_targets.append([76, 40])
            scale.append(10.)
        else:
            raise NotImplementedError

    state_dim = [i.observation_space.shape[0] for i in env]
    act_dim = [i.action_space.shape[0] for i in env]

    state_mean, state_std = [], []
    for i in range(2):
        dataset_path = f'data_mujoco/{env_names[i]}-{dataset[i]}-v2.pkl'
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
        state_mean.append(np.mean(states, axis=0))
        state_std.append(np.std(states, axis=0) + 1e-6)

        # num_timesteps = sum(traj_lens)

        # print('=' * 50)
        # print(f'Starting new experiment: {env_name} {dataset}')
        # print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        # print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        # print('=' * 50)

    K = variant['K'] 
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    # num_timesteps = max(int(pct_traj * num_timesteps), 1)


    def get_batch(batch_size=256, max_len=K):
        pass

    def get_batch_recurrent(batch_size=256):
        pass

    def eval_episodes(target_rew, idx=0):

        def fn(model):
            returns, lengths, all_average_diff, all_last_action_diff = [], [], [], []
            # ids = np.arange(variant['num_eval_episodes'])
            # reward = np.zeros(variant['num_eval_episodes'])
            # lengths = np.zeros(variant['num_eval_episodes'])
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt' or 'mamba' in model_type:
                        # if variant['run_type'] != 'train':
                        #     env.seed(variant['seed']+num_eval_episodes)
                        #     env.action_space.seed(variant['seed']+num_eval_episodes)

                        ret, length = evaluate_episode_rtg(
                            env[idx],
                            state_dim[idx],
                            act_dim[idx],
                            model,
                            max_ep_len=max_ep_len[idx],
                            scale=scale[idx],
                            target_return=target_rew / scale[idx],
                            mode=mode,
                            state_mean=state_mean[idx],
                            state_std=state_std[idx],
                            device=device,
                            model_type=variant['model_type'],
                            mamba_type=variant['mamba_type']
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
    model = []
    model_parameters = []
    load_model = [variant['load_model'], variant['load_model_b']]
    for i in range(2):
        if model_type == 'mamba':
            model.append(Mamba_mujoco(
                state_dim=state_dim[i],
                act_dim=act_dim[i],
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
                time_embd = variant['time_embd'],
                type_input = variant['type_input']
            ))
        elif model_type == 'dt':
            from decision_transformer.models.decision_transformer import DecisionTransformer
            model.append(DecisionTransformer(
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
            ))

        report_parameters(model[i])
        if load_model[i] != 'none' and load_model[i] != 'None':
            file_modl = load_model[i]
            loadedfile = torch.load(file_modl)
            if model_type == 'mamba':
                # model[i].load_state_dict(loadedfile['model'])
                model_parameters.append(loadedfile['model'])
            elif model_type == 'dt':
                # model[i].load_state_dict(loadedfile['model_state_dict'])
                model_parameters.append(loadedfile['model_state_dict'])
            print(f"Using model_parameters_{i} from {file_modl}, model type is {model_type}")
            optimizer = loadedfile['optimizer']
            for g in optimizer.param_groups:
                g['lr'] = variant['learning_rate']
        else:
            model_parameters.append(None)
            print(f"Using model_parameters_{i} initially, model type is {model_type}")



    total_keys = [i for i in model_parameters[0].keys()]
    total_swap_keys = [s for s in total_keys if (('.mixer' in s) and ('in_proj' not in s) and ('out_proj' not in s))]

    if variant['type'] == 'single_swap':
        variant['swap_idx'] = -1
        for idx in range(2):
            variant['idx'] = idx
            model[idx].load_state_dict(model_parameters[idx])
            model[idx] = model[idx].to(device=device)

            trainer = SequenceTrainer(
                model=model[idx],
                optimizer=None,
                batch_size=None,
                get_batch=None,
                scheduler=None,
                loss_fn=None,
                eval_fns=[eval_episodes(tar, idx) for tar in env_targets],
            )

            print(f'baseline model_{idx} load sucessfully!')
            print(f'evaluation environment is {env_names[idx]}')
            trainer.eval_step(print_logs=True, env=variant['env'])
            print('')

        variant['swap_idx'] = variant['swap_idx'] + 1 
        for head in range(0,len(total_swap_keys),2):
            key1, key2 = total_swap_keys[head], total_swap_keys[head+1]
            for idx in range(2):
                variant['idx'] = idx
                now_model_parameters = copy.deepcopy(model_parameters[idx])
                now_model_parameters[key1], now_model_parameters[key2] = model_parameters[1-idx][key1], model_parameters[1-idx][key2]

                model[idx].load_state_dict(now_model_parameters)
                print(f'model_{idx} get swapped layer {total_swap_keys[head]}, {total_swap_keys[head+1]}, load sucessfully!')
                print(f'evaluation environment is {env_names[idx]}')
                
                trainer = SequenceTrainer(
                    model=model[idx],
                    optimizer=None,
                    batch_size=None,
                    get_batch=None,
                    scheduler=None,
                    loss_fn=None,
                    eval_fns=[eval_episodes(tar, idx) for tar in env_targets],
                )
                trainer.eval_step(print_logs=True, env=variant['env'])
            variant['swap_idx'] = variant['swap_idx'] + 1
    elif variant['type'] == 'attention_swap':
        s6_layer_name = [s for s in total_keys if (('.mixer' in s) and ('in_proj' not in s) and ('out_proj' not in s))]

        print('targets', env_targets)
        for idx in range(2):
            variant['idx'] = idx

            model[idx].load_state_dict(model_parameters[idx])
            print(' ')
            print(f'Using initial model_{idx}, load sucessfully!')
            print(f'evaluation environment is {env_names[idx]}')
            trainer = SequenceTrainer(
                model=model[idx],
                optimizer=None,
                batch_size=None,
                get_batch=None,
                scheduler=None,
                loss_fn=None,
                eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
            )
            trainer.eval_step(print_logs=True, env=env_names[idx])

            for si in range(variant['n_layer']):
                # change single s6 layer
                swap_names = [i for i in s6_layer_name if f".{si}." in i]
                new_param = swap_two_model_params(idx, swap_names, model_parameters)
 
                model[idx].load_state_dict(new_param)
                print(' ')
                print(f'model_{idx} get swapped layer {swap_names}, load sucessfully!')
                print(f'evaluation environment is {env_names[idx]}')
                trainer = SequenceTrainer(
                    model=model[idx],
                    optimizer=None,
                    batch_size=None,
                    get_batch=None,
                    scheduler=None,
                    loss_fn=None,
                    eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
                )
                trainer.eval_step(print_logs=True, env=env_names[idx])
            new_param = swap_two_model_params(idx, s6_layer_name, model_parameters)
            model[idx].load_state_dict(new_param)
            print(' ')
            print(f'model_{idx} get all attention swapped layer {s6_layer_name}, load sucessfully!')
            print(f'evaluation environment is {env_names[idx]}')
            trainer = SequenceTrainer(
                model=model[idx],
                optimizer=None,
                batch_size=None,
                get_batch=None,
                scheduler=None,
                loss_fn=None,
                eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
            )
            trainer.eval_step(print_logs=True, env=env_names[idx])
    elif variant['type'] == 'random':
        s6_layer_name = [s for s in total_keys if (('.mixer' in s) and ('in_proj' not in s) and ('out_proj' not in s))]

        print('targets', env_targets)
        for idx in range(2):
            variant['idx'] = idx

            for si in range(variant['n_layer']):
                # change single s6 layer
                swap_names = [i for i in s6_layer_name if f".{si}." in i]
                new_param = change_model_params(idx, swap_names, model_parameters)
 
                model[idx].load_state_dict(new_param)
                print(' ')
                print(f'model_{idx} layer {swap_names} gets normal random, load sucessfully!')
                print(f'evaluation environment is {env_names[idx]}')
                trainer = SequenceTrainer(
                    model=model[idx],
                    optimizer=None,
                    batch_size=None,
                    get_batch=None,
                    scheduler=None,
                    loss_fn=None,
                    eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
                )
                trainer.eval_step(print_logs=True, env=env_names[idx])
            new_param = change_model_params(idx, s6_layer_name, model_parameters)
            model[idx].load_state_dict(new_param)
            print(' ')
            print(f'model_{idx} get all attention swapped layer {s6_layer_name} got normal random, load sucessfully!')
            print(f'evaluation environment is {env_names[idx]}')
            trainer = SequenceTrainer(
                model=model[idx],
                optimizer=None,
                batch_size=None,
                get_batch=None,
                scheduler=None,
                loss_fn=None,
                eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
            )
            trainer.eval_step(print_logs=True, env=env_names[idx])
    elif variant['type'] == 'zero':
        s6_layer_name = [s for s in total_keys if (('.mixer' in s) and ('in_proj' not in s) and ('out_proj' not in s))]
        # print(s6_layer_name)
        print('targets', env_targets)
        for idx in range(2):
            variant['idx'] = idx

            for si in range(variant['n_layer']):
                # change single s6 layer
                swap_names = [i for i in s6_layer_name if f".{si}." in i]
                new_param = change_model_params(idx, swap_names, model_parameters, method='zero')
 
                model[idx].load_state_dict(new_param)
                print(' ')
                print(f'model_{idx} layer {swap_names} gets zero, load sucessfully!')
                print(f'evaluation environment is {env_names[idx]}')
                trainer = SequenceTrainer(
                    model=model[idx],
                    optimizer=None,
                    batch_size=None,
                    get_batch=None,
                    scheduler=None,
                    loss_fn=None,
                    eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
                )
                trainer.eval_step(print_logs=True, env=env_names[idx])
            new_param = change_model_params(idx, s6_layer_name, model_parameters, method='zero')
            model[idx].load_state_dict(new_param)
            print(' ')
            print(f'model_{idx} all attention swapped layer {s6_layer_name} gets zero, load sucessfully!')
            print(f'evaluation environment is {env_names[idx]}')
            trainer = SequenceTrainer(
                model=model[idx],
                optimizer=None,
                batch_size=None,
                get_batch=None,
                scheduler=None,
                loss_fn=None,
                eval_fns=[eval_episodes(tar, idx) for tar in env_targets[idx]],
            )
            trainer.eval_step(print_logs=True, env=env_names[idx])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--env_b', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset_b', type=str, default='medium')
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--reward_type', type=str, default='normal') # K_term, normal
    parser.add_argument('--time_embd', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--type_input', type=str, default='B3LD')  # B3LD,BL3D
    parser.add_argument('--type', type=str, default='attention_swap')  # single_swap, attention_swap
    parser.add_argument('--mamba_type', type=str, default='tr')  # tr, rnn
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='mamba')
    parser.add_argument('--load_model', type=str, default='') # path
    parser.add_argument('--load_model_b', type=str, default='') # path
    parser.add_argument('--save_prefix', type=str, default='merge')
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
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)

    args = parser.parse_args()

    experiment('dmamba', variant=vars(args))
