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
import sys
currentdir = '/home/daiyang22/decision-transformer/'
sys.path.insert(0, currentdir)
try:
    from VMamba.classification.models import VSSM
except:
    pass
import numpy as np


def build_vssm_model(n_embed):
    model = VSSM(
        patch_size=4, 
        in_chans=4, 
        num_classes=0, 
        depths=[2], 
        dims=[n_embed], 
        # ===================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0,
        ssm_init="v0",
        forward_type="v2",
        # ===================
        mlp_ratio=4.0,
        mlp_act_layer="GELU",
        mlp_drop_rate=0.0,
        # ===================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        downsample_version="v2",
        patchembed_version="v1",
        gmlp=False,
        use_checkpoint=False,
        head=True
    )
    return model


class DMamba(nn.Module):
    def __init__(self, config):
        super().__init__()    
        self.config = config
        self.model_type = config.model_type
        self.reward_type = config.reward_type # 'normal', 'K'
        self.type_input = config.type_input
        # input embedding stem
        self.n_embed = config.n_embed
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embed))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embed))
        self.drop = nn.Dropout(config.dropout)

        factory_kwargs = {"device": None, "dtype": None}

        self.infer_params = InferenceParams(max_batch_size=1, max_seqlen=1)
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
           self.state_encoder = build_vssm_model(config.n_embed)
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embed), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embed), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)


        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.n_embed if config.type_input=='B3LD' else 3*config.n_embed,
            # dropout_val=config.dropout,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=config.initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            **factory_kwargs,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
            )
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

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

    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, mask=None, running=False, cu_seqlens=None):
        del_r = 1
        batch_size = actions.shape[0]
        sequence_len = actions.shape[1]
        if running or sequence_len==1:
            del_r = 0        
        states = states[:,del_r:,...].reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        actions = actions[:,:sequence_len-del_r,...].reshape(-1, sequence_len-del_r, 1)
        rtgs = rtgs[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, 1).type(torch.float32)
        
        state_embeddings = self.state_encoder(states) # (batch * (len-1)), n_embed)
        state_embeddings = state_embeddings.view(batch_size, sequence_len-del_r, self.config.n_embed) # (batch, len-1, n_embed)
        if self.reward_type == 'K' and self.type_input == 'B3LD':
            if actions is not None and self.model_type == 'reward_conditioned': 
                rtgs = rtgs[:, 0, ...]
                rtg_embeddings = self.ret_emb(rtgs)
                action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, len-1, n_embed)

                token_embeddings = torch.zeros((batch_size, (sequence_len-del_r)*2+1, self.config.n_embed), dtype=torch.float32, device=state_embeddings.device)

                token_embeddings[:,0,:] = rtg_embeddings
                token_embeddings[:,1::2,:] = action_embeddings
                token_embeddings[:,2::2,:] = state_embeddings
            else:
                raise NotImplementedError()            
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, len-1, n_embed)

            if self.type_input == 'B3LD':
                token_embeddings = torch.zeros((batch_size, (sequence_len-del_r)*3, self.config.n_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:,::3,:] = action_embeddings
                token_embeddings[:,1::3,:] = rtg_embeddings
                token_embeddings[:,2::3,:] = state_embeddings
            elif self.type_input == 'BL3D':
                token_embeddings = torch.cat([action_embeddings, rtg_embeddings, state_embeddings], dim=-1)
        else:
            raise NotImplementedError()
        
        u = self.drop(token_embeddings) 
        if cu_seqlens is not None and self.config.mamba_type == 'rnn':
            y = self.backbone(u, cu_seqlens=cu_seqlens)
        else:
            y = self.backbone(u)
        logits = self.head(y)

        if actions is not None and self.model_type == 'reward_conditioned':
            if self.reward_type == 'K' and self.type_input == 'B3LD':
                logits = logits[:, 2::2, :] # only keep predictions from state_embeddings
            elif self.type_input == 'B3LD':
                logits = logits[:, 2::3, :]
            elif self.type_input == 'BL3D':
                logits = logits
        else:
            raise NotImplementedError()
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print('mask', mask)
            if (mask==0).any():
                mask = mask[:, del_r:]
                targets = targets[:, del_r:]
                b_mask = mask.unsqueeze(-1)
                masked_logits = torch.masked_select(logits, b_mask.bool()).reshape(-1, self.config.vocab_size)
                masked_targets = torch.masked_select(targets, b_mask.bool())            
                loss = F.cross_entropy(masked_logits, masked_targets)
            else:
                targets = targets[:, del_r:]    
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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
        ssm_cfg = {'d_state': 32, 'expand': 2} # N
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

    action_size = 4
    # state_size = 3
    seq_len = 5000
    device = 'cuda'

    config = MambaConfig(vocab_size=action_size+1)

    model = DMamba(config)
    model = model.to(device=device)
    model.eval()
    u = torch.randn((1, seq_len, 4, 84, 84)).to(device)
    action = torch.randint(0, action_size + 1, (1, seq_len)).long().to(device)
    # print(action)
    rtg = torch.randn((1,seq_len, 1)).to(device)
    # timesteps = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    out1, _ = model(states=u, actions=action, rtgs=rtg, running=True)
    out2 = torch.zeros(out1.shape).to(device)
    model.infer_params.reset(max_batch_size=1, max_seqlen=seq_len)
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