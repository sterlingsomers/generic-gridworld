import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gym
import numpy as np
import pandas as pd
import  pickle
import sys
syspath = '/home/konstantinos/Documents/Projects/general_grid/common'
sys.path.append(syspath)#'/Users/constantinos/Documents/Projects/genreal_grid/common')

from gym.wrappers import TimeLimit
from common.wrappers import CoordsOnlyWrapper, ImageOnlyWrapper, FeaturesWrapper

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, GAIL
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset

from A2C_static_graph import A2C as A2C_mine
from envs.generic_env_v2 import GenericEnv
from envs.core_v2 import *

from sklearn.neighbors import KNeighborsClassifier

import os
print('Path=',os.getcwd())

# Parallel environments
#env = make_vec_env('CartPole-v1', n_envs=4)
# env = GenericEnv()
# Goal(env, entity_type='goal', color='green')
# NetworkAgent(env, entity_type='agent', color='aqua')

env = GenericEnv()
Goal(env, entity_type='goal', color='green', position='specific', position_coords=(5, 7))
NetworkAgent(env, entity_type='agent', color='aqua' , position='specific', position_coords=(3,1))
ChasingBlockingAdvisary(env, entity_type='advisary',
                        color='red', obs_type='data', position='specific', position_coords=(6,8))#position='near-goal')
# red dark_orange
env.seed(0)
# Correct order of wrappers
env = FeaturesWrapper(env)
# env = ImageOnlyWrapper(env)
env = TimeLimit(env)
env._max_episode_steps = 200

# m = A2C_mine(model_filepath='../analysis/networkb.pb')#getogoal.pb')
# dm = pickle.load(open('/home/konstantinos/Documents/Projects/general_grid_v2/models/2020_Nov12_time13-17_agent_knn5_nenvs20_numdates50000.dm', 'rb'))


def creat_ibl_human_clf():

    def softmax(weights):  # Should be applied on a vector and NOT a matrix!
        """Compute softmax values for each sets of matching scores in x."""
        # weights = 5*weights
        e_x = np.exp(-weights)
        s = e_x / e_x.sum(1).reshape(weights.shape[0], 1)
        return s

    # Load data
    print('Create classifier...')
    participant = 1
    # Have to put absolute path as the algo doesnt start from general grid folder
    filename = '/home/konstantinos/Documents/Projects/general_grid_v2/data/net_vs_pred/' + str(participant) + \
               '_participant.pdr'
    df = pd.read_pickle(filename)

    data = df['features'].values
    data = np.vstack(data)
    print(data.shape)

    y_fc = df['action_label'].values
    clf_neigh = KNeighborsClassifier(n_neighbors=100, weights=softmax)
    clf_neigh.fit(data, y_fc) # fit with all the data!!!
    return clf_neigh

clf_neigh = creat_ibl_human_clf()

def expert_ibl_human(obs):

    probs = clf_neigh.predict_proba([obs])
    probs_ = probs[:,[2,0,4,1,3]].copy()
    action = np.argmax(probs_)
    return np.asscalar(action)


def expert_ibl(obs):
    epsilon = 0
    knn = 5
    nenvs = 1
    obs = np.stack([obs]).astype(dtype=np.float32)
    action = dm.choose_action(obs, epsilon, knn, nenvs)
    action = np.asscalar(action)
    return action

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

d = generate_expert_traj(expert_ibl_human, save_path='./expert_traj/predator_ibl_human', env=env, n_episodes=500) #
# should be 0 if

# d = generate_expert_traj(expert, save_path='./expert_traj/predator_networkb', env=env, n_episodes=100,
#                          image_folder='./images') # should be 0 if
# model is trained! Else it will start training as well

dataset = ExpertDataset(expert_path='./expert_traj/predator_ibl_human.npz',
                        traj_limitation=-1, batch_size=32)  # -1 means all trajectoriesdataset.plot()
dataset.plot()
print('job done')
# model = A2C(CnnPolicy, env, verbose=1, tensorboard_log="./a2c_stable_baselines/")

# data will be saved in a numpy archive named `expert_cartpole.npz`
# dataset = ExpertDataset(expert_path='../expert_cartpole.npz',
#                         traj_limitation=-1, batch_size=32) # -1 means all trajectories
# # print('Pretrain')
# # model.pretrain(dataset, n_epochs=500)
#
# model = GAIL('MlpPolicy', env, dataset, verbose=1)
# # Note: in practice, you need to train for 1M steps to have a working policy
# model.learn(total_timesteps=1000)
# model.save("gail_pendulum")
#
# # Test the pre-trained model
# # env = model.get_env()
# obs = env.reset()
# obs = obs['img']
#
# reward_sum = 0.0
# for step in range(200):
#     print('Step=',step)
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     # obs = obs['img']
#     reward_sum += reward
#     env.render()
#     if done:
#             print(reward_sum)
#             reward_sum = 0.0
#             obs = env.reset()
#             # obs = obs['img']
#
# env.close()
# model.save("gail_stable_baselines")

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