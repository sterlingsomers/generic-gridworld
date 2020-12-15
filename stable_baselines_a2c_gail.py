import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, GAIL
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset

from A2C_static_graph import A2C as A2C_mine
from envs.generic_env_v2 import GenericEnv
from envs.core_v2 import *

# Parallel environments
#env = make_vec_env('CartPole-v1', n_envs=4)
# env = gym.make("CartPole-v1")
env = GenericEnv()
# envs = Mn(gym.make('CartPole-v0'), './data', video_callable=lambda episode_id: episode_id % 10 == 0,
#           force=True)
# envs = Mn(env, './data', video_callable=lambda episode_id: episode_id % 5 == 0,
#           force=True)  # record every 10 episodes
Goal(env, entity_type='goal', color='green')
NetworkAgent(env, entity_type='agent', color='aqua')

m = A2C_mine(model_filepath='./analysis/getogoal.pb')

def expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    value, expertprobs, fc2, conv1, conv2, conv3, fc2_logit_W, logits_pre_bias, conv1W = m.test(
        data= np.expand_dims(_obs,axis=0))
    action = np.argmax(expertprobs,axis=1)[0]
    return action#env.action_space.sample()

# generate_expert_traj(expert, 'expert_cartpole', env, n_timesteps=int(1e5), n_episodes=1000)
# model = A2C(CnnPolicy, env, verbose=1, tensorboard_log="./a2c_stable_baselines/")

# data will be saved in a numpy archive named `expert_cartpole.npz`
dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                        traj_limitation=-1, batch_size=32) # -1 means all trajectories
# print('Pretrain')
# model.pretrain(dataset, n_epochs=500)

model = GAIL('MlpPolicy', env, dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_pendulum")

# Test the pre-trained model
# env = model.get_env()
obs = env.reset()
obs = obs['img']

reward_sum = 0.0
for step in range(200):
    print('Step=',step)
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    obs = obs['img']
    reward_sum += reward
    env.render()
    if done:
            print(reward_sum)
            reward_sum = 0.0
            obs = env.reset()
            obs = obs['img']

env.close()
model.save("gail_stable_baselines")

# model.learn(total_timesteps=25000)
# model.save("a2c_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = A2C.load("a2c_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()