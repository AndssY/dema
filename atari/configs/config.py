class MambaConfig:
    # warmup_tokens = 375e6
    model_type = 'reward_conditioned'
    reward_type = 'normal'
    type_input = 'B3LD'
    n_embed = 128
    d_model = 128 # D
    ssm_cfg = {'d_state': 128, 'expand': 2} # N
    inner_type = 'conv'
    dropout = 0.1
    n_layer = 6
    n_head = 1
    d_conv = 4
    expand = 2
    norm_epsilon = 1e-5
    initializer_cfg = None
    rms_norm = True
    residual_in_fp32 = True
    fused_add_norm = True
    # embd_pdrop = 0.1
    # resid_pdrop = 0.1
    # norm_layer = nn.LayerNorm
    recurrent_mode = False
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


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