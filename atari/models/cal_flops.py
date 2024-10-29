from __future__ import annotations
# this code is from https://github.com/johnma2006/mamba-minimal/tree/master
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
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm.models.mixer_seq_simple import _init_weights
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

class Mamba(nn.Module):
    def __init__(self, 
                 d_model=64,
                 n_layer=3,
                 vocab_size=256):
        """Full Mamba model."""
        super().__init__()
        # self.args = args
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.embedding = nn.Linear(vocab_size, d_model)
        self.layers = nn.ModuleList([ResidualBlock(d_model) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        # logits = self.lm_head(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        # self.args = args
        self.mixer = MambaBlock(d_model)
        self.norm = RMSNorm(d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, conv_bias=True, bias=False, dt_rank="auto"):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * self.d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        # print(self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class DMamba(nn.Module):
    def __init__(self, config):
        super().__init__()    
        self.config = config
        self.model_type = config.model_type

        # input embedding stem
        self.n_embed = config.n_embed
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embed))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embed))
        self.drop = nn.Dropout(config.dropout)

        factory_kwargs = {"device": None, "dtype": None}

        # self.infer_params = InferenceParams(max_batch_size=1, max_seqlen=1)
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


        self.backbone = Mamba(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.n_embed,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
            )
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

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
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, mask=None, running=False):
        del_r = 1
        batch_size = actions.shape[0]
        sequence_len = actions.shape[1]
        if running:
            del_r = 0        
        states = states[:,del_r:,...].reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        actions = actions[:,:sequence_len-del_r,...].reshape(-1, sequence_len-del_r, 1)
        rtgs = rtgs[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, 1).type(torch.float32)
        
        state_embeddings = self.state_encoder(states) # (batch * (len-1)), n_embed)
        state_embeddings = state_embeddings.view(-1, sequence_len-del_r, self.config.n_embed) # (batch, len-1, n_embed)
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, len-1, n_embed)

            token_embeddings = torch.zeros((batch_size, (sequence_len-del_r)*3, self.config.n_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = action_embeddings
            token_embeddings[:,1::3,:] = rtg_embeddings
            token_embeddings[:,2::3,:] = state_embeddings
        else:
            raise NotImplementedError()
        
        u = self.drop(token_embeddings) 
        y = self.backbone(u)
        logits = self.head(y)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 2::3, :] # only keep predictions from state_embeddings
        else:
            raise NotImplementedError()
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            mask = mask[:, del_r:]
            targets = targets[:, del_r:]
            b_mask = mask.unsqueeze(-1)
            masked_logits = torch.masked_select(logits, b_mask.bool()).reshape(-1, self.config.vocab_size)
            masked_targets = torch.masked_select(targets, b_mask.bool())            
            loss = F.cross_entropy(masked_logits, masked_targets)
        return logits, loss

    def get_action(self, states, actions, rtgs=None, timesteps=None):
        # states: (4, 84, 84)
        # actions: (1, 1)
        # rtgs: (1, 1, 1)
        states = states.view(1, 4, 84, 84).type(torch.float32)
        actions = actions.view(1, 1)
        rtgs = rtgs.view(1, 1, 1).type(torch.float32)
        
        state_embeddings = self.state_encoder(states) # (1, n_embed)
        state_embeddings = state_embeddings.view(1, 1, self.config.n_embed) # (1, 1, n_embed)
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (1, 1, n_embed)

            token_embeddings = torch.zeros((1, 3, self.config.n_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,0,:] = action_embeddings
            token_embeddings[:,1,:] = rtg_embeddings
            token_embeddings[:,2,:] = state_embeddings
        else:
            raise NotImplementedError()
        
        x = self.drop(token_embeddings)
        outputs = torch.zeros(1, x.shape[1], self.config.d_model, dtype=x.dtype, device=x.device)
        for i in range(x.shape[1]):
            ret_y = self.backbone(x[:, i:i+1], inference_params=self.infer_params)
            outputs[:, i, :] = ret_y
            self.infer_params.seqlen_offset += 1
        logits = self.head(outputs)
        return logits[:, 2, :]
        # (1, vocab_size)


if __name__ == '__main__':
    import random

    class MambaConfig:
        """ base GPT config, params common to all GPT versions """
        model_type = 'reward_conditioned'


        n_embed = 128
        d_model = 128 # D
        ssm_cfg = {'d_state': 64, 'expand': 2} # N
        inner_type = 'conv'
        dropout = 0.1
        n_layer = 3
        n_head = 1
        d_conv = 4
        # expand = 2
        norm_epsilon = 1e-5
        initializer_cfg = None
        rms_norm = True
        residual_in_fp32 = True
        fused_add_norm = True
        # embd_pdrop = 0.1
        # resid_pdrop = 0.1
        # norm_layer = nn.LayerNorm
        recurrent_mode = False
        def __init__(self, vocab_size, **kwargs):
            self.vocab_size = vocab_size
            for k,v in kwargs.items():
                setattr(self, k, v)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    batch_size = 128
    action_size = 4
    # state_size = 3
    seq_len = 30
    device = 'cuda'

    config = MambaConfig(vocab_size=action_size+1)

    model = DMamba(config)
    model = model.to(device=device)
    model.eval()
    u = torch.randn((batch_size, seq_len, 4, 84, 84)).to(device)
    action = torch.randint(0, action_size + 1, (batch_size, seq_len)).long().to(device)
    # print(action)
    rtg = torch.randn((batch_size,seq_len, 1)).to(device)
    # timesteps = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    # out1, _ = model(states=u, actions=action, rtgs=rtg, running=True)


    ####################################
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    flops = FlopCountAnalysis(model, (u, action, None, rtg))
    print("FLOPs: ", flops.by_module())
    print("FLOPs: ", flops.total())
    # print(parameter_count_table(model))
    exit(0)
    #####################################