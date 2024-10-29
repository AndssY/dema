"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from functools import partial
import math
import logging
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_ref
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.utils.generation import InferenceParams

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import opt_einsum as oe

import sys
import os
import inspect
currentdir = '/decision-transformer/s4_modules'
# print(currentdir)
# s4dir = os.path.join(os.path.dirname(currentdir), "s4_modules")
sys.path.insert(0, currentdir)
from s4_module import S4
contract = oe.contract



class S4_Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, H,
                 l_max, d_state, measure, dt_min, dt_max, rank, trainable, lr, weight_norm
                 ,s4mode='nplr'):
        super().__init__()
        self.h = H
        self.n = d_state
        self.config =config
        #self.beforeblock = nn.BatchNorm1d(self.h) if self.config.layer_norm_s4 else nn.Identity()
        self.beforeblock = nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity()
        self.afterblock = nn.Sequential(

            nn.GELU(),
            nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity(),
            nn.Linear(self.h, self.h),
            nn.GELU(),
            #nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            #nn.Linear(self.h, 3 * self.h),
            #nn.GELU(),
            #nn.Linear(3 * self.h, self.h),
            #nn.Tanh(),
        )
       #self.afterblock = nn.Sequential(
       #     nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
       #     nn.GELU(),
       #     nn.Linear(self.h, 3 * self.h),
       #     nn.GELU(),
       #     nn.Linear(3 * self.h, self.h),
            #nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            #nn.Linear(self.h, 3 * self.h),
            #nn.GELU(),
            #nn.Linear(3 * self.h, self.h),
            #nn.Tanh(),
        #    nn.Dropout(self.config.dropout) if self.config.dropout>0 else nn.Identity(),
        #)
        self.l_max = l_max
        if self.config.single_step_val:
            l_max = None
        self.s4_mod_in = S4(H, l_max=l_max, d_state=d_state, measure=measure,
            dt_min=dt_min, dt_max=dt_max, rank=rank, trainable=trainable, lr=lr,
            weight_norm=weight_norm, linear=True, mode=s4mode, precision=self.config.precision, n_ssm=self.config.n_ssm,
        )
        #### TEST
        #self.normalizer = nn.LayerNorm(self.h)

    def forward(self, u):
        #y = u.transpose(-1, -2)
        y = u
        y = self.beforeblock(y)
        y = y.transpose(-1, -2)
        if "seq" in self.config.base_model:
            self.s4_mod_in.setup_step()
            s4_state =  self.s4_mod_in.default_state(y.shape[0]).to(device=y.device)
            out = []
            for i in range(y.shape[2]):
                yt, s4_state = self.s4_mod_in.step(y[:,:,i], s4_state)
                out.append(yt.unsqueeze(-1))
            y = torch.cat(out, dim=-1)
            next_state = s4_state
        else:
            y, next_state = self.s4_mod_in(y)
        if self.config.train_noise>0:
            y = y + self.config.train_noise * torch.randn(y.shape, device=y.device)
        y = y.transpose(-1, -2)
        y = self.afterblock(y)
        if self.config.s4_resnet:
            y = y + u
        #### TEST
        #y = self.normalizer(y+u.transpose(-1, -2))
        return y, next_state

    def step(self, u, state):
        #print(f"LOGZZ u1 {u.shape}")
        y = self.beforeblock(u)
        #print(f"LOGZZ u2 {y.shape}")
        inner_output, new_state = self.s4_mod_in.step(y, state)
        #print(f"LOGZZ u3 {inner_output.shape}")
        inner_output = self.afterblock(inner_output)
        #print(f"LOGZZ u4 {inner_output.shape}")
        if self.config.s4_resnet:
            inner_output = inner_output + u
        #print(f"LOGZZ u5 {inner_output.shape}")
        #### TEST
        #inner_output = self.normalizer(inner_output+u)
        return inner_output, new_state


class DS4(nn.Module):
    def __init__(self, config):
        super().__init__()    
        self.config = config
        self.h = config.d_model
        self.n = config.d_state
        self.s4_mode = config.kernel_mode
        self.model_type = config.model_type
        self.input_emb_size = config.n_embed
        # self.drop = nn.Dropout(config.dropout)
        self.s4_weight_decay = config.s4_weight_decay
        self.batch_mode = False

        if self.config.layer_norm_s4:
            self.input_norm_layer = nn.LayerNorm(self.h)
            self.output_norm_layer = nn.LayerNorm(self.h)
        if self.config.dropout>0:
            self.dropoutlayer = nn.Dropout(self.config.dropout)
        if self.config.activation is not None:    
            self.input_proj2 = nn.Linear(self.h, self.h)



        self.l_max = None

        if self.config.inner_type == 'conv':
            self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                            nn.Flatten(), nn.Linear(3136, config.n_embed), nn.Tanh())
        elif self.config.inner_type == 'transformer':
            raise NotImplemented
        elif self.config.inner_type == 'mamba':
            raise NotImplemented
        elif self.config.inner_type == 'vmamba':
            raise NotImplemented
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embed), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embed), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.input_projection = nn.Linear(config.n_embed*3, self.h)
        self.beforeblock = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.h, self.h),
            nn.Dropout(self.config.dropout),
        )
        self.s4_amount = self.config.n_layer
        trainable = None if self.config.s4_trainable else False
        if self.config.base_model == "lin":
            raise NotImplemented
        elif self.config.base_model == "rnn":
            raise NotImplemented
        else:
            self.s4_mods = nn.ModuleList([S4_Block(self.config, H=self.h, l_max=None, d_state=self.n, measure=config.measure,
                                                   dt_min=config.dt_min, dt_max=config.dt_max, rank=config.rank, trainable=trainable, lr=config.ssm_lr,
                                                   weight_norm=config.weight_norm, s4mode=self.s4_mode) for _ in
                                          range(self.s4_amount)])
        if self.config.discrete > 0:
            #self.output_projection = nn.Linear(self.h, self.action_dim * self.config.discrete, bias=False)
            self.output_projection = nn.Linear(self.h, (self.vocab_size + self.state_dim ) * self.config.discrete, bias=False)
            self.output_projection_rtg = nn.Linear(self.h, 1)
        else:
            self.output_projection = nn.Linear(self.h, config.vocab_size, bias=False)

    def pre_val_setup(self):
        for mod in self.s4_mods:
            mod.s4_mod_in.setup_step()
        return

    # need to validate it in migpt.utils, if need to change
    def reset_state(self, device): # TODO
        self.curr_state = torch.zeros((1, self.h, self.n)).to(device=device, dtype=torch.cfloat)

    ##added optimizer to match the DT original structure need to edit:
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        s4_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # parameters that need to be configured:
        # 'kernel.krylov.C', 'output_linear.weight', 'kernel.krylov.w', 'kernel.krylov.B', 'D', 'kernel.krylov.log_dt'
        # original:
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        #S4_kernel_modules = (krylov)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or 'norm' in fpn or 'embeddings' in fpn or 'emb' in fpn or 'ln_f' in fpn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        #no_decay.add('pos_emb')
        #no_decay.add('global_pos_emb')
        #for r in ["s4_mod.kernel.krylov.C", "s4_mod.output_linear.weight", "s4_mod.kernel.krylov.w", "s4_mod.kernel.krylov.B", "s4_mod.D", "s4_mod.kernel.krylov.log_dt"]:
        #    if self.s4_weight_decay > 0:
        #        decay.add(r)
        #    else:
        #        no_decay.add(r)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        union_params = decay | no_decay | s4_decay
        for d1, d2 in [(decay, no_decay), (decay, s4_decay), (s4_decay, decay)]:
            inter_params = decay & no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(s4_decay))], "weight_decay": self.s4_weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def get_initial_state(self, batchsize, device='cpu'):
        if not self.config.single_step_val:
            return None

        return [mod.s4_mod_in.default_state(batchsize).to(device=device) for mod in self.s4_mods]

    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, mask=None, running=False, cache=None, **kwargs):
        del_r = 1
        batch_size = actions.shape[0]
        sequence_len = actions.shape[1]
        if running:
            del_r = 0        
        states = states[:,del_r:,...].reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        # actions = actions[:,:sequence_len-del_r,...].reshape(-1, 1)
        # rtgs = rtgs[:, :sequence_len-del_r, ...].reshape(-1, 1).type(torch.float32)
        
        state_embeddings = self.state_encoder(states) # (batch * (len-1)), n_embed)
        state_embeddings = state_embeddings.view(-1, self.config.n_embed) # (batch, len-1, n_embed)
        action_embed = self.action_embeddings(actions[:,:sequence_len-del_r,...].reshape(-1,1))
        reward_embed = self.ret_emb(rtgs[:, :sequence_len-del_r, ...].reshape(-1, 1).type(torch.float32))
        action_embed = action_embed.reshape(batch_size * (sequence_len - del_r), -1)
        reward_embed = reward_embed.reshape(batch_size * (sequence_len - del_r), -1)        
        
        u = torch.cat([state_embeddings, action_embed, reward_embed], dim=-1)
        u = self.input_projection(u).reshape(batch_size, sequence_len-del_r, self.h)
        ret_y = self.beforeblock(u)

        for mod in self.s4_mods:
            ret_y, _ = mod(ret_y)
        ret_temp = self.output_projection(ret_y.reshape(-1,self.h))
        if self.config.discrete > 0:
            raise NotImplemented
        logits = ret_temp.reshape(batch_size, sequence_len - del_r, self.config.vocab_size)

        loss = None
        if targets is not None:
            mask = mask[:, del_r:]
            targets = targets[:, del_r:]
            b_mask = mask.unsqueeze(-1)
            masked_logits = torch.masked_select(logits, b_mask.bool()).reshape(-1, self.config.vocab_size)
            masked_targets = torch.masked_select(targets, b_mask.bool())            
            loss = F.cross_entropy(masked_logits, masked_targets)
        return logits, loss

    def get_action(self, states, actions, rtgs=None, timesteps=None, s4_states=None, running=False, targets=None, **kwargs):
        states = states.view(1, 4, 84, 84).type(torch.float32)
        actions = actions.view(1, 1)
        rtgs = rtgs.view(1, 1, 1).type(torch.float32)       

        state_embed = self.state_encoder(states)
        state_embed = state_embed.view(1, 1, self.config.n_embed)
        action_embed = self.action_embeddings(actions.type(torch.long)).view(1, 1, self.config.n_embed)
        reward_embed = self.ret_emb(rtgs.type(torch.float32)).view(1, 1, self.config.n_embed)
        print(state_embed.shape, action_embed.shape, reward_embed.shape)
        u = torch.cat([state_embed, action_embed, reward_embed], dim=-1)
        u = self.input_projection(u)
        ret_y = self.beforeblock(u)
        output_states = []
        for i, mod in enumerate(self.s4_mods):
            input_state = None
            if s4_states is not None:
                input_state = s4_states[i]
            print('1', ret_y.shape)
            print('2', input_state.shape)
            ret_y, new_state = mod.step(ret_y, input_state)
            print('3', ret_y.shape)
            print('4', new_state.shape)
            output_states.append(new_state)

        ret_act = self.output_projection(ret_y).unsqueeze(1)
        return ret_act[0, -1, :], output_states


if __name__ == '__main__':
    import random

    class Ds4Config():
        model_type = 'reward_conditioned'
        inner_type = 'conv'
        dropout = 0
        activation = None
        layer_norm_s4 = False
        d_state = 64
        d_model = 128
        n_embed = 128
        n_layer = 6


        # SSM layer
        single_step_val = 1
        len_corr = 1
        kernel_mode = 'nplr'
        measure = 'legs'
        dt_min = 0.001
        dt_max = 0.1
        rank = 1
        s4_trainable = True
        s4_weight_decay = 0.0
        ssm_lr = None
        len_corr = False
        stride = 1
        
        weight_norm = False

        use_state = False
        setup_c = False
        s4_resnet = False
        s4_onpolicy = False
        s4_layers = 6

        track_step_err = False
        recurrent_mode = False
        n_ssm = 1
        precision = 1
        train_noise = 0
        base_model = "s4"
        discrete = 0
        s4_ant_multi_lr = None
        def __init__(self, vocab_size, **kwargs):
            self.vocab_size = vocab_size
            for k,v in kwargs.items():
                setattr(self, k, v)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    action_size = 4
    # state_size = 3
    seq_len = 5000
    device = 'cuda'

    config = Ds4Config(vocab_size=action_size+1)

    model = DS4(config)
    model = model.to(device=device)
    model.eval()
    u = torch.randn((1, seq_len, 4, 84, 84)).to(device)
    action = torch.randint(0, action_size + 1, (1, seq_len)).long().to(device)
    # print(action)
    rtg = torch.randn((1,seq_len, 1)).to(device)
    # timesteps = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    out1, _ = model(states=u, actions=action, rtgs=rtg, running=True)
    out2 = torch.zeros(out1.shape).to(device)
    # model.infer_params.reset(max_batch_size=1, max_seqlen=seq_len)
    for x in range(seq_len):
        z = model.get_action(states=u[:,x,...], actions=action[:,x], rtgs=rtg[:,x,:])
        out2[:,x,:] = z

    print("#"*15 +f"Diff out1| out2" + "#"*15)
    print(f"Sizes: out1 {out1.shape} | out2 {out2.shape}")
    totnumbers = (out1.shape[1] * out1.shape[2])
    print(f"Diff  L2: {torch.sum(torch.pow(out1 - out2, 2))}")
    print(f"Diff  L1: {torch.sum(torch.abs(out1 - out2))}")
    print(f"first L1: {torch.sum(torch.abs(out1 - out2)[0,0,:])}")
    print(f"last  L1: {torch.sum(torch.abs(out1 - out2)[0,-1,:])}")

    print(f"Average Diff  L2: {torch.sum(torch.pow(out1 - out2, 2)) / totnumbers}")
    print(f"Average Diff  L1: {torch.sum(torch.abs(out1 - out2)) / totnumbers}")
    print(f"Average first L1: {torch.sum(torch.abs(out1 - out2)[0,0,:]) / out1.shape[2]}")
    print(f"Average last  L1: {torch.sum(torch.abs(out1 - out2)[0,-1,:]) / out1.shape[2]}")
    print(f"Diff enlarged:\n{out1 - out2}")
    print(f"#"*100)
    print(f"Out1:\n{out1}")
    print(f"Out2:\n{out2}")