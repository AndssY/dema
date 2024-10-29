
# Is Mamba Compatible with Trajectory Optimization in Offline Reinforcement Learning?

## Instructions

1. Python 3.8.19 \
`conda create -n your_env_name python==3.8`
2. Activate Env \
`conda activate your_env_name`
3. install the requirements of [dt](https://github.com/kzl/decision-transformer.git) or `pip install -r requirements.txt`
4. upgrade torch(`torch==1.2.0+cu121`) and install `mamba` from our source \
cd `mamba` and `pip install -e .`
5. move to subdir and run script respectively.

## Notice

1. Every time you switch to or switch out of the `cu_seqlens` branch, you need to reinstall mamba `pip install -e .`, this is because the `cu_seqlens` branch has modified the C files, so recompilation is necessary.


## Acknowledgements

This repository is based on [decision-transformer](https://github.com/kzl/decision-transformer.git), [Mamba](https://github.com/state-spaces/mamba.git), [Mamba-minimal](https://github.com/johnma2006/mamba-minimal) and [HiddenMambaAttn](https://github.com/AmeenAli/HiddenMambaAttn?tab=readme-ov-file). Thanks for their wonderful works.