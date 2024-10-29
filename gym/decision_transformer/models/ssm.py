from dataclasses import field
from functools import partial
import math
import random
import numpy as np
import torch
import torch.nn as nn

# from decision_transformer.models.model import TrajectoryModel
from mamba_ssm.models.mixer_seq_simple import MixerModel
# from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.ops.triton.layernorm import RMSNorm
from mamba_ssm.utils.generation import InferenceParams
# import opt_einsum as oe


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

        self.backbone = MixerModel(
            d_model=self.d_model,
            n_layer=n_layer,
            vocab_size=self.input_emb_size if type_input=='B3LD' else 3*self.input_emb_size,
            # dropout_val=self.dropout,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.inference_params = InferenceParams(max_seqlen=3000, max_batch_size=1)
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

    action_size = 12
    state_size = 13
    seq_len = 200
    device = 'cuda'
    model = Mamba_mujoco(
        state_dim=state_size,
        act_dim=action_size,
        max_length=seq_len,
        max_ep_len=1000,
        embedding_dim=128,
        d_model=56,
        n_layer=2,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        dropout=0
    ).to(device)
    model.eval()
    u = torch.randn((1,seq_len,state_size+action_size)).to(device)
    rtg = torch.randn((1,seq_len,1)).to(device)
    timesteps = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    _, out1, _ = model(u[:, :, 0:state_size], u[:, :, state_size:], None, rtg, timesteps, running=True)

    out2 = torch.zeros(1,seq_len,action_size).to(device)
    model.inference_params = InferenceParams(max_seqlen=3000, max_batch_size=1)
    for x in range(seq_len):
        z = model.get_action(u[:,x,0:state_size], u[:,x,state_size:], None, rtg[:,x,:], timesteps[:,x])
        out2[:, x, :] = z
    out2 = out2
        # infer.seqlen_offset += 1
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
    #print(f"Diff enlarged:\n{out1 - out2}")
    print(f"#"*100)
    print(f"Out1:\n{out1}")
    print(f"Out2:\n{out2}")