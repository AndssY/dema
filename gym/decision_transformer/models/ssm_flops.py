from __future__ import annotations
from dataclasses import field
from functools import partial
import math
import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn


import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


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
        


if __name__ == '__main__':
    import sys
    import os
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print(currentdir)
    s4dir = os.path.join(os.path.dirname(currentdir), "s4_module")
    sys.path.insert(0, s4dir)
    from model import TrajectoryModel
else:
    from decision_transformer.models.model import TrajectoryModel
# contract = oe.contract

class Mamba_mujoco(TrajectoryModel):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        embedding_dim: int,
        d_model: int,
        n_layer: int,
        dropout: int,
        ssm_cfg=None,
        rms_norm: bool=True,
        residual_in_fp32: bool=True,
        fused_add_norm: bool=True,
        initializer_cfg=None,
        device=None,
        dtype=None,
        max_length=1000,
        max_ep_len=4096,
        action_tanh=True,
        reward_type='normal',
        time_embd=False,
        type_input='B3LD',
    ):
        TrajectoryModel.__init__(self, state_dim, act_dim, max_length=max_length)
        self.max_length = max_length
        # config = MambaConfig()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.input_emb_size = embedding_dim
        self.d_model = d_model
        self.dropout = dropout
        self.reward_type = reward_type
        self.time_embd=time_embd
        self.type_input = type_input
        n_layer = n_layer
        ssm_cfg = ssm_cfg
        rms_norm = rms_norm
        residual_in_fp32 = residual_in_fp32
        fused_add_norm = fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}
        if self.time_embd:
            self.embed_timestep = nn.Embedding(max_ep_len, self.input_emb_size)
        self.ret_emb = nn.Linear(1, self.input_emb_size)
        self.state_encoder = nn.Linear(self.state_dim, self.input_emb_size)
        self.action_embeddings = nn.Linear(self.act_dim, self.input_emb_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.d_model, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.d_model, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(self.d_model, 1)

        self.backbone = Mamba(
            d_model=self.d_model,
            n_layer=n_layer,
            vocab_size=self.input_emb_size if type_input=='B3LD' else 3*self.input_emb_size,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        # self.inference_params = InferenceParams(max_seqlen=3000, max_batch_size=1)
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, states, actions, rewards, rtg, timesteps=None, running=False, cache=None, goals=None, target_goal=None,
                attention_mask=None, position_ids=None, inference_params=None, num_last_tokens=0):
        del_r = 1
        batch_size = states.shape[0]
        sequence_len = actions.shape[1]
        if running or sequence_len==1:
            del_r = 0
        state_embeddings = self.state_encoder(states[:,del_r:,...].reshape(-1, sequence_len-del_r, self.state_dim).type(torch.float32).contiguous())
        action_embeddings = self.action_embeddings(actions[:,:sequence_len-del_r,...].reshape(-1, sequence_len-del_r, self.act_dim))
        returns_embeddings = self.ret_emb(rtg[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, 1).type(torch.float32))
        # action_embeddings = action_embeddings.reshape(batch_size * (sequence_len - del_r), -1)
        # returns_embeddings = returns_embeddings.reshape(batch_size * (sequence_len - del_r), -1)
        
        if self.time_embd:
            time_embeddings = self.embed_timestep(timesteps[:, :sequence_len-del_r])
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings
        # print(state_embeddings.shape,action_embeddings.shape,returns_embeddings.shape)
        if self.reward_type == 'K' and self.type_input == 'B3LD':
            returns_embeddings = returns_embeddings[:, 0]
            u = torch.zeros((batch_size, (sequence_len-del_r)*2+1, self.input_emb_size), dtype=torch.float32, device=state_embeddings.device)
            # print('token',token_embeddings.shape)
            u[:,0,:] = returns_embeddings
            u[:,1::2,:] = action_embeddings
            u[:,2::2,:] = state_embeddings
        elif self.type_input == 'B3LD':
            u = torch.stack(
                (returns_embeddings, action_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*(sequence_len-del_r), self.input_emb_size)
        elif self.type_input == 'BL3D':
            u = torch.cat([returns_embeddings, action_embeddings, state_embeddings], dim=-1)
        y = self.backbone(u)
        # if num_last_tokens > 0:
        #     y = y[:, -num_last_tokens:]
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

        if self.reward_type == 'K' and self.type_input == 'B3LD':
            y = y[:, 2::2, :]
            action_preds = self.predict_action(y)
        elif self.type_input == 'B3LD':
            y = y.reshape(batch_size, sequence_len-del_r, 3, self.d_model).permute(0, 2, 1, 3)
            # return_preds = self.predict_return(y[:, 2])  # predict next return given state and action
            # state_preds = self.predict_state(y[:, 2])    # predict next state given state and action
            action_preds = self.predict_action(y[:,-1])  # predict next action given state
        elif self.type_input == 'BL3D':
            action_preds = self.predict_action(y).reshape(batch_size, sequence_len - del_r, -1)
        return None, action_preds, None

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, position_ids=None, num_last_tokens=0, **kwargs):
        if len(states.shape)==3:
            bs = states.shape[0]
        else:
            bs = 1
        states = states.reshape(bs, -1, self.state_dim)[:,-1]
        actions = actions.reshape(bs, -1, self.act_dim)[:,-1]
        returns_to_go = returns_to_go.reshape(bs, -1, 1)[:,-1]
        # timesteps = timesteps.reshape(1, -1)[:,-1

        state_embeddings = self.state_encoder(states).reshape(bs, -1, self.input_emb_size)
        action_embeddings = self.action_embeddings(actions).reshape(bs, -1, self.input_emb_size)
        returns_embeddings = self.ret_emb(returns_to_go).reshape(bs, -1, self.input_emb_size)
        if self.time_embd:
            time_embeddings = self.embed_timestep(timesteps[:, -1]).reshape(1, -1)
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings
        if self.type_input == 'B3LD':
            u = torch.stack(
                (returns_embeddings, action_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(bs, 3, self.input_emb_size)
        elif self.type_input == 'BL3D':
            u = torch.cat([returns_embeddings, action_embeddings, state_embeddings], dim=-1)
        outputs = torch.zeros(bs, u.shape[1], self.d_model, dtype=u.dtype, device=u.device)
        for i in range(u.shape[1]):
            ret_y = self.backbone(u[:, i:i+1], inference_params=self.inference_params)
            outputs[:, i] = ret_y[:, 0]
            self.inference_params.seqlen_offset += 1
        if self.type_input == 'B3LD':
            outputs = outputs.reshape(bs, 1, 3, self.d_model).permute(0, 2, 1, 3)
        action_preds = self.predict_action(outputs[:,-1])
        # TODO BL3D
        return action_preds[:, -1]

    def get_action2(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        if len(states.shape)==3:
            bs = states.shape[0]
        else:
            bs = 1
        states = states.reshape(bs, -1, self.state_dim)
        actions = actions.reshape(bs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(bs, -1, 1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, running=True, **kwargs)

        return action_preds[:,states.shape[1]-1]


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

if __name__ == '__main__':
    
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    batch_size = 64
    action_size = 3
    state_size = 11
    K = 20
    device = 'cuda'
    model = Mamba_mujoco(
        state_dim=state_size,
        act_dim=action_size,
        max_length=K,
        max_ep_len=1000,
        embedding_dim=128,
        d_model=128,
        n_layer=3,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        dropout=0
    ).to(device)
    model.eval()

    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    state = torch.rand(batch_size, K, state_size).to(device)
    action = torch.rand(batch_size, K, action_size).to(device)
    rtg = torch.rand(batch_size, K, 1).to(device)
    timesteps = torch.randint(high=100,size=(batch_size, K)).to(device)
    flops = FlopCountAnalysis(model, (state,action,rtg,timesteps))
    print("FLOPs: ", flops.by_module())
    print(parameter_count_table(model))
    exit(0)