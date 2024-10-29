import gym
import d4rl
import numpy as np
# dat_name = ['expert-v2','medium-expert-v2','medium-v2','medium-replay-v2','random-v2']
dat_name = ['expert-v2','medium-v2','medium-replay-v2']
env_name = ['hopper-','walker2d-','halfcheetah-']
for i in env_name:
    for j in dat_name:
        env = gym.make(i+j)
        dataset = env.get_dataset()
        print(i+j,np.mean(dataset['rewards']))
        print(i+j,len(dataset['rewards']))