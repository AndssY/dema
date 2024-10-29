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
        attention='step',
        start_time=1,
        end_time=200,
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
    # p_list = []
    collected_data  = {}
    fused_attn_matrix=[]
    
    import pickle
    import matplotlib.pyplot as plt
    import os
    import datetime
    now = datetime.datetime.now()

    all_layer_attentions = np.zeros((3, end_time-start_time, model.d_model*model.ssm_cfg['expand'], model.max_length*3))
    collected_data  = {}

    if attention == 'trig':
        model.backbone.compute_attn_matrix=True
        folder_name = now.strftime('%Y%m%d_%H%M%S') + f'{env.spec.id}' + f'_{start_time}' + 'trig_attention'
        os.makedirs(folder_name, exist_ok=True)
        print(f'Folder "{folder_name}" created successfully!')
    elif attention == 'step':
        model.backbone.compute_attn_matrix=False
        folder_name = now.strftime('%Y%m%d_%H%M%S') + f'{env.spec.id}' + f'_{start_time}-{end_time}' + 'step_attention'
        os.makedirs(folder_name, exist_ok=True)
        print(f'Folder "{folder_name}" created successfully!')
    else:
        raise NotImplementedError
    for t in range(max_ep_len):

        if model_type == 'mamba':
            action = model.get_action2(
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

        # for test
        if done:
            if t < start_time:
               raise ValueError('env run time is less than the start time of collection of trig/step attention.')
            elif t < end_time:
                print('INFO: env run time is less than the end time of collection of step attention.')  
                collected_data[f'step_attn_matrices'] = all_layer_attentions[:, :t]
                with open(f'{folder_name}/all_attn_data.pkl', 'wb') as file:
                    pickle.dump(collected_data, file)
                print(f'Data of time {start_time} to {t} saved successfully!')
            else:
                collected_data[f'step_attn_matrices'] = all_layer_attentions
                with open(f'{folder_name}/all_attn_data.pkl', 'wb') as file:
                    pickle.dump(collected_data, file)
                print(f'Data of time {start_time} to {end_time} saved successfully!')                
            break

        if t == start_time and attention == 'trig':

            collected_data  = {}
            fused_attn_matrix=[]

            for selected_layer in range(3):
                selected_channel = 30
                # Extract and normalize attention matrices
                attn_matrix = model.backbone.layers[selected_layer].mixer.xai.abs()
                normalize_attn_mat = lambda attn_mat : (attn_mat.abs() - torch.min(attn_mat.abs())) / (torch.max(attn_mat.abs()) - torch.min(attn_mat.abs()))
                attn_matrix_normalize = normalize_attn_mat(attn_matrix)
                # Plot each attention matrix
                fig, axs = plt.subplots(1, 6, figsize=(20,10))
                collected_data[f'attn_matrices_{selected_layer}'] = attn_matrix
                for i in range(6):
                    print('11', attn_matrix.shape)
                    axs[i].imshow(attn_matrix.cpu().detach().numpy()[0, selected_channel+i, :, :])
                    axs[i].axis('off')
                # plt.colorbar(im)
                plt.savefig(f'{folder_name}/attn_matrix_layer{selected_layer}.png')
                print(f'attn_matrix_layer{selected_layer}.png saved successfully!')
                plt.clf()
                print('attn_matrix', attn_matrix.shape)
                mean_attn_matrix = np.mean(attn_matrix.cpu().detach().numpy(), axis=1)
                print('mean_attn_matrix',mean_attn_matrix.shape)
                fused_attn_matrix.append(mean_attn_matrix[0])
                im = plt.imshow(mean_attn_matrix[0], interpolation='nearest')
                collected_data[f'mean_attn_matrices_{selected_layer}'] = mean_attn_matrix[0]
                plt.axis('off')
                plt.colorbar(im)
                plt.savefig(f'{folder_name}/mean_attn_matrix_layer{selected_layer}.png')
                print(f'mean_attn_matrix_layer{selected_layer}.png saved successfully!')
                plt.clf()
            print('fused_attn_matrix', len(fused_attn_matrix), fused_attn_matrix[0].shape)
            # p = np.concatenate(fused_attn_matrix, axis=0)
            p = np.mean(fused_attn_matrix, axis=0)
            # print(p.shape)
            im = plt.imshow(p, interpolation='nearest')
            collected_data['p'] = p
            plt.axis('off')
            plt.colorbar(im)
            plt.savefig(f'{folder_name}/heatmap_fused_layer.png')
            print(f'heatmap_fused_layer.png saved successfully!')
            with open(f'{folder_name}/all_attn_data.pkl', 'wb') as file:
                pickle.dump(collected_data, file)
            print('Data saved successfully!')
            exit(0)
        if start_time <= t < end_time and attention == 'step':
            for selected_layer in range(3):
                selected_channel = 30 # 
                attn_matrix = model.backbone.layers[selected_layer].mixer.xai.abs().cpu().detach().numpy()
                # mean_attn_matrix = np.mean(attn_matrix.cpu().detach().numpy(), axis=1)[0] # (1,60) -> (60,)
                all_layer_attentions[selected_layer,t-start_time,:] = attn_matrix

    return episode_return, episode_length

def evaluate_episode_rtg_parallel2(
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



        action = model.get_action2(
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