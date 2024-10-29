import numpy as np
import torch
import d4rl
import time


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

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, 
                 scheduler=None, eval_fns=None, reward_scale = None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.reward_scale = reward_scale
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()

        for _ in range(num_steps):
            start_time = time.time()
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        logs['time/training'] = seconds_format(time.time() - train_start)

        eval_start = time.time()
        self.model.eval()
        res = []
        if iter_num % 1 == 0:
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v
                    if 'return_mean' in k:
                        res.append(v)
            logs['time/total'] = seconds_format(time.time() - self.start_time)
            logs['time/train'] = seconds_format(eval_start - self.start_time)
            logs['time/evaluation'] = seconds_format(time.time() - eval_start)
            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

            return logs, res
        else:
            logs['time/total'] = seconds_format(time.time() - self.start_time)
            logs['time/train'] = seconds_format(eval_start - self.start_time)
            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            return logs, res

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        # s,a:(B,L,X), mask, rtg:(B,L)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def eval_step(self, print_logs=False, env=None):
        logs = dict()
        eval_start = time.time()
        self.model.eval()
        res = []
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
                if 'return_mean' in k:
                    res.append(v)
        logs['time/evaluation'] = seconds_format(time.time() - eval_start)
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]
        if print_logs:
            print('=' * 80)
            result = []
            for k, v in logs.items():
                if 'return_mean' in k:
                    print(f'{k}: {v}')
                    print(f'{k}: {d4rl.get_normalized_score(f"{env}-medium-v2", v)}')
                    result.append(v)
                # print('the best result during training is ', best_result)
                # print('the best normalized score is ', [d4rl.get_normalized_score(f"{variant['env']}-medium-v2", i)
                #                                         for i in best_result])
                else:
                    print(f'{k}: {v}')
            print('=' * 80)
        return result