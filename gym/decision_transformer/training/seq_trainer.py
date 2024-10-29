import numpy as np
import torch

from decision_transformer.models.ssm import Mamba_mujoco
from decision_transformer.training.trainer import Trainer
import sys
# from decision_transformer.models.ssm import S4D_mujoco_wrapper_v3

class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, goals = self.get_batch(self.batch_size)
        # s,a:(B,L,X), rtg:(B,L,1), mask, timesteps,mask:(B,L)
        action_target = torch.clone(actions)

        # target_goal = None
        # to_pass_rtg = rtg[:,:-1]

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask)
        if type(self.model) == Mamba_mujoco:
            attention_mask = attention_mask[...,1:]
            action_target = action_target[:,1:,...]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
    
    def step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, goals = self.get_batch(self.batch_size)
        # s,a:(B,L,X), rtg:(B,L,1), mask, timesteps,mask:(B,L)
        action_target = torch.clone(actions)

        # target_goal = None
        # to_pass_rtg = rtg[:,:-1]

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps)
        if type(self.model) == Mamba_mujoco:
            attention_mask = attention_mask[...,1:]
            action_target = action_target[:,1:,...]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), action_preds, action_target