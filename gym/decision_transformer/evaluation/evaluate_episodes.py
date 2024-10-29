from mamba_ssm.utils.generation import InferenceParams
import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action2(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        model_type=None,
        mamba_type='tr'
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    model.inference_params = InferenceParams(max_seqlen=3000, max_batch_size=1)
    for t in range(max_ep_len):

        if model_type == 'mamba':
            if mamba_type == 'tr':
                action = model.get_action2(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            elif mamba_type == 'rnn':
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),            
                )    
        elif model_type == 'dt':
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg_parallel(
        env,
        state_dim,
        act_dim,
        model,
        num_envs=100,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    # this code got from https://github.com/sail-sg/envpool/blob/main/envpool/atari/atari_pretrain_test.py#L37
    ids = np.arange(num_envs)
    new_ids = np.arange(num_envs)
    rewards = torch.zeros(num_envs).to(device)
    lengths = torch.zeros(num_envs)
    state = env.reset()
    all_done = np.zeros(num_envs)

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    states = torch.from_numpy(state).reshape(num_envs, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, 1, act_dim), device=device, dtype=torch.float32)
    # rewards = torch.zeros(ids, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    target_return = target_return.repeat(num_envs, 1)
    timesteps = timesteps.repeat(num_envs, 1)

    model.inference_params = InferenceParams(max_seqlen=3000, max_batch_size=num_envs)

    for t in range(max_ep_len):
        action = model.get_action2(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            None,
            target_return.to(dtype=torch.float32),
            None,
        )
        actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
        # rewards = torch.cat([rewards, torch.zeros(ids, device=device)])

        actions[:, -1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action, ids)
        new_ids = info["env_id"]
        all_done = np.logical_or(all_done, done)
        reward = torch.tensor(reward).to(device)

        new_ids = ids[~all_done]

        cur_state = torch.from_numpy(state).to(device=device).reshape(num_envs, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)

        if mode != 'delayed':
            pred_return = target_return[:,-1] - (reward/scale)
        else:
            pred_return = target_return[:,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        
        rewards[new_ids] += reward[new_ids]
        lengths[new_ids] += 1
        
        if len(new_ids)==0:
            break
    # print(rewards)
    # print(lengths)
    return rewards.detach().cpu().numpy(), lengths.detach().cpu().numpy()