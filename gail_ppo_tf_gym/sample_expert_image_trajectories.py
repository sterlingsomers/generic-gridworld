import argparse
import gym
import numpy as np
import pickle
#from network_models.policy_net import Policy_net
import tensorflow as tf

# import sys
# syspath = '/home/konstantinos/Documents/Projects/general_grid/common'
# sys.path.append(syspath)
from common.wrappers import  DiscreteToBoxWrapper
from envs.generic_env_v2 import GenericEnv
from envs.core_v2 import *
from A2C_static_graph import A2C

import os
os.chdir('/Users/constantinos/Documents/Projects/genreal_grid')
# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='filename of model to test', default='trained_models/ppo/model.ckpt')
    parser.add_argument('--iteration', default=50, type=int)

    return parser.parse_args()


def main(args):
    MAX_STEPS = 200
    env = DiscreteToBoxWrapper(GenericEnv())
    # envs = Mn(gym.make('CartPole-v0'), './data', video_callable=lambda episode_id: episode_id % 10 == 0,
    #           force=True)
    # envs = Mn(env, './data', video_callable=lambda episode_id: episode_id % 5 == 0,
    #           force=True)  # record every 10 episodes
    Goal(env, entity_type='goal', color='green')
    NetworkAgent(env, entity_type='agent', color='aqua')
    #env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space['img']
    # Policy = pickle.load(open('./models/dm_getgoal.dm','rb')) #TODO: Instead use A2C frozen graph and pickle obs and actions
    Policy = A2C(model_filepath='./analysis/getogoal.pb')
    #Policy = Policy_net('policy', env)
    #saver = tf.train.Saver()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, args.model)
    obs = env.reset()
    obs = obs['img']#['features']
    mb_obs = []
    mb_actions = []
    for iteration in range(args.iteration):  # episode
        print('ITERATION:', iteration)
        observations = []
        actions = []
        run_steps = 0
        done=False

        while True:# or (t<MAX_STEPS):
            print('-> step ', run_steps)
            run_steps += 1
            # prepare to feed placeholder Policy.obs
            mb_obs.append(obs)
            obs = np.stack([obs]).astype(dtype=np.float32)

            act= Policy.test(obs)#Policy.choose_action(obs, epsilon=0, knn=5, nenvs=1)#act(obs=obs, stochastic=True)
            act = np.argmax(act[1])#np.asscalar(act)

            observations.append(obs)

            actions.append(act)
            mb_actions.append(act)

            next_obs, reward, done, info = env.step(act)
            next_obs = next_obs['img']#['features']

            if done:
                print(run_steps)
                obs = env.reset()
                obs = obs['img']#['features']
                break
            else:
                obs = next_obs

        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        # open_file_and_save('gail_ppo_tf_gym/expert_trajectory/observations.csv', observations)
        # open_file_and_save('gail_ppo_tf_gym/expert_trajectory/actions.csv', actions)
        #TODO: Fix the saving this is wrong as you save only the steps of one iteration
        # actions = np.array(actions).astype(dtype=np.int32)

    dict = {'observations': np.array(mb_obs),
            'actions': np.array(mb_actions).astype(dtype=np.int32)
            }
    pickle_in = open(
        '/Users/constantinos/Documents/Projects/genreal_grid/gail_ppo_tf_gym/expert_trajectory/a2c_expert.tj', 'wb')
    pickle.dump(dict, pickle_in)


if __name__ == '__main__':
    args = argparser()
    main(args)