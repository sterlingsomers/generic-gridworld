from datetime import datetime
import argparse
import time
import os
import numpy as np
import pickle
import pandas as pd
from baselines import logger
from baselines.bench import Monitor

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from baselines.common import set_global_seeds
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import gym
from gym.wrappers import TimeLimit
from common.wrappers import FeaturesWrapper

import tensorflow as tf
from tqdm import tqdm

from envs.generic_env_v2 import GenericEnv
from envs.core_v2 import *

from IBL import IBL

now = datetime.now()

def softmax(weights): # Should be applied on a vector and NOT a matrix!
  """Compute softmax values for each sets of matching scores in x."""
  # weights = 5*weights
  e_x = np.exp(-weights)
  s = e_x / e_x.sum(1).reshape(weights.shape[0],1)
  return s

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo/'+now.strftime("%Y%m%d-%H%M%S") + "/")
    parser.add_argument('--savedir', help='save directory', default='trained_models/ibl')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(3e3), type=int)
    parser.add_argument('--maxsteps', default=int(1e3))
    parser.add_argument('--save_agent', default=False)
    parser.add_argument('--training', default=True)
    parser.add_argument('--participant', default=int(0))
    # parser.add_argument('--images', default=True)
    return parser.parse_args()


def main(args):
    # Load data
    print('Create classifier...')
    participant = args.participant
    filename = './data/net_vs_pred/' + str(participant) + '_participant.pdr'
    df = pd.read_pickle(filename)

    data = df['features'].values
    data = np.vstack(data)
    print(data.shape)

    y_fc = df['action_label'].values
    clf_neigh = KNeighborsClassifier(n_neighbors=100, weights=softmax)
    clf_neigh.fit(data, y_fc) # fit with all the data!!!
    probs = clf_neigh.predict_proba(data)
    probs_ = probs[:,[2,0,4,1,3]].copy() #swap axis from ['DOWN', 'LEFT', 'NOOP', 'RIGHT', 'UP'] to ['NOOP',...]

    # Clean duplicates
    print('Clean duplicates...')
    dat, ind = np.unique(data, axis=0, return_index=True)
    sort = np.sort(ind) # We need to sort as the array will be a mess so we cannot find the relevant instnc in actions
    unique_dat = data[sort]
    pr = probs_[sort]

    # Create env
    print('Create env...')
    env = GenericEnv()
    Goal(env, entity_type='goal', color='green', position='specific', position_coords=(6,7))
    NetworkAgent(env, entity_type='agent', color='aqua', position='specific', position_coords=(1,1))
    ChasingBlockingAdvisary(env, entity_type='advisary',
                            color='red', obs_type='data', position='near-goal')  # red dark_orange
    env.seed(0)
    # Correct order of wrappers
    env = FeaturesWrapper(env)
    env = TimeLimit(env)
    env._max_episode_steps = 200

    #Notes: Synchronous updates need to run for an amount of time (e.g. T timesteps) even if the env is done! So you
    # need the handling episode to store the results in tb. In sync updates you CANNOT use while done as you need to
    # run all envs till time=T

    mem_capacity = 20000 # 5 actions, for static ~4000 states but because of higher level feats we will get fewer
    # states.
    knn = 30
    FEATS_DIM = data.shape[1]
    ACTIONS = env.action_space.n
    print('Create IBL...')
    dm = IBL(capacity=mem_capacity, num_feats=FEATS_DIM, num_actions=ACTIONS, neighbors=knn, temp=1)
    dm.add(unique_dat, pr)
    # dm = pickle.load(open('./models/2020_Dec15_time00-42_agent_knn5_nenvs1_steps1500000.dm', 'rb'))

    # dm = pickle.load(open('./models/2020_Dec14_time21-28_agent_knn30_nenvs1_steps500000.dm', 'rb'))
    # x_fc = data
    # y_fc = df['action_label'].values
    # x_train_fc, x_test_fc, y_train_fc, y_test_fc = train_test_split(x_fc, y_fc, test_size=0.20, random_state=42)
    #
    # clf_neigh = KNeighborsClassifier(n_neighbors=100, weights=softmax)
    # clf_neigh.fit(x_train_fc, y_train_fc) # fit with all the data!!!
    # y_pred_fc = clf_neigh.predict(x_test_fc)
    # print('kNN Accuracy: %.4f' % clf_neigh.score(x_test_fc, y_test_fc))
    # probs = clf_neigh.predict_proba(data)
    # probs_ = probs[:,[2,0,4,1,3]].copy() #swap axis from ['DOWN', 'LEFT', 'NOOP', 'RIGHT', 'UP'] to ['NOOP',...]
    # dm.add(data[:knn,:], probs_[:knn,:]) # This adds everything without checking for same instances!!!

    # Find which instances arent already in
    # indx = []
    # for t in range(knn,data.shape[0]):
    #     capacity = dm.curr_capacity
    #     dm.update(data[t,:].reshape(1,FEATS_DIM), np.argmax(probs_[t,:]), probs_[t,:].max())
    #     if dm.curr_capacity == capacity+ 1:
    #         indx.append(t)
    # # Find if row 104 exists (duplicate). Classifier will return same probs for same obs so you can safely remove probs
    # # as well

    # np.where((data==data[104]).all(axis=1))[0]

    if args.training:
        epsilon_start = 1.0  # 1.0
        epsilon_min = 0.01  # 0.005
        # decay should be the number of total updates so you get from e_init to e_final within the number of updates
        epsilon_decay = args.maxsteps
        epsilon_rate = ((epsilon_start - epsilon_min) / epsilon_decay)
        epsilon = epsilon_start
        gamma = args.gamma
        num_envs = 1
        step = 0
        iteration = 0
        print('MAIN PROCESS')
        initial_state = env.reset()
        latest_obs = initial_state
        # for update in range(num_updates):
        writer = tf.summary.FileWriter(args.logdir)
        pbar = tqdm(total=args.maxsteps)
        while step < args.maxsteps:
            # print('Update:', update)

            mb_actions = []
            mb_obs = []
            mb_rewards = []
            mb_done = []

            # if update>=10000: epsilon = 0.20
            # else: epsilon = max(epsilon_min, epsilon - epsilon_rate) # after zeroing epsilon exploration will be 0.5% (epsilon=epsilon_rate)
            epsilon = max(epsilon_min, epsilon - epsilon_rate)
            # epsilon = 0.005
            epsilon = 0.0
            episode_length = 0
            while True:  # In multiple envs that reset is automatically done we HAVE TO DO nstep reward! So MAX_STEPS
                episode_length += 1
                latest_obs = np.stack([latest_obs]).astype(dtype=np.float32)

                action = dm.choose_action(latest_obs, epsilon, knn, num_envs)
                action = np.asscalar(action)
                obs, r, done, info = env.step(action)
                # print(done)
                step += 1
                pbar.update(1)

                mb_obs.append(latest_obs.copy())
                mb_actions.append(action)
                mb_rewards.append(r)
                mb_done.append(done)
                dm.tm += 0.01  # TODO: This might be wrong here! If this changes then when you go back in time all

                if done:
                    latest_obs = env.reset()
                    break
                else:
                    latest_obs = obs

            tm_temp = dm.tm

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(mb_rewards)**(len(mb_rewards)))])
                               , iteration)
            # [num_envs x num steps]
            mb_obs = np.reshape(mb_obs, newshape=[num_envs] + [-1] + list(env.observation_space.shape)) # [steps x obs]
            mb_actions= np.array(mb_actions).reshape(num_envs,-1)
            mb_rewards= np.array(mb_rewards).reshape(num_envs,-1)
            mb_done= np.array(mb_done).reshape(num_envs,-1)

            # UPDATES
            # disc_rewards = np.zeros([num_envs, mb_rewards.shape[1]])
            # R = 0
            # for t in range(mb_rewards.shape[1] - 1, -1, -1):
            #     dm.tm -= 0.01
            #     R = mb_rewards[:, t] + gamma * R * (1 - mb_done[:, t])
            #     disc_rewards[:, t] = R  # [nenvs x maxtimesteps] # nenvs = batch
            #     dm.update(mb_obs[:, t], mb_actions[:, t], R)  # , modify=True) # TODO: Might be easier to do it outside the
                # loop
            dm.tm = tm_temp
            iteration += 1
        writer.close()
        pbar.close()
        # elapsed_time = time.time() - start_time
        # print('\n')
        # if elapsed_time >= 60:
        #     print("--- %s minutes ---" % (np.round(elapsed_time / 60, 2)))
        # else:
        #     print("--- %s seconds ---" % (np.round(elapsed_time, 2)))
        if args.save_agent:
            path = './models/'
            now = datetime.now()
            timestamp = str(now.strftime("%Y_%b%d_time%H-%M"))
            type = '.dm'
            agent_name = path + timestamp + '_' + 'agent_' + 'knn' + str(knn) + '_' + 'nenvs' + str(
                num_envs) + '_' + 'steps' + str(args.maxsteps) + type
            pickle_in = open(agent_name, 'wb')
            # '/Users/constantinos/Documents/Projects/MFEC_02orig/models/agent.dm', 'wb')
            pickle.dump(dm, pickle_in)
        try:
            env.close()
        except AttributeError:
            pass

if __name__ == '__main__':
    args = argparser()
    main(args)