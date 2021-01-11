import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
os.getcwd()
import numpy as np

import argparse
# import gym
from gym.wrappers import TimeLimit
from common.wrappers import CoordsOnlyWrapper, ImageOnlyWrapper, FeaturesWrapper

from stable_baselines import A2C, PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from envs.generic_env_v2 import GenericEnv
from envs.core_v2 import *

from datetime import datetime
now = datetime.now()


algorithm_description = 'ppo_feats_predator'#'a2c_coords_predator'#'a2c_coords_get2goal'
#Notes: NO NEED to change WRAPPER when changing task. Simply the missing entitiy will have zero coords
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log summaries directory', default='./log/train/' + algorithm_description)# +
                        #now.strftime("%Y%m%d-%H%M%S") + "/")
    parser.add_argument('--savedir', help='save model directory', default='./trained_models/' + algorithm_description)
    parser.add_argument('--num_envs', default=int(40))
    parser.add_argument('--maxsteps', default=int(1e6))
    parser.add_argument('--images', default=False)
    parser.add_argument('--gail', default=False)
    parser.add_argument('--training', default=False)
    parser.add_argument('--bc', default=True)
    return parser.parse_args()

def make_single_env(seed, images=True):
    env = GenericEnv()
    Goal(env, entity_type='goal', color='green' , position='specific', position_coords=(5, 7))
    NetworkAgent(env, entity_type='agent', color='aqua' , position='specific', position_coords=(3,1))
    ChasingBlockingAdvisary(env, entity_type='advisary',
                            color='red', obs_type='data', position='specific', position_coords=(6, 8))
    #position='near-goal')  # red dark_orange
    env.seed(seed)
    # Correct order of wrappers
    if images:
        env = ImageOnlyWrapper(env)
    else:
        env = FeaturesWrapper(env)
    env = TimeLimit(env)
    env._max_episode_steps = 200
    return Monitor(env, algorithm_description)

def make_custom_env(num_env, seed, images=True, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank):
        def _thunk():
            env = GenericEnv()
            Goal(env, entity_type='goal', color='green')#, position='specific', position_coords=(6, 7))
            NetworkAgent(env, entity_type='agent', color='aqua')#, position='specific', position_coords=(2,2))
            ChasingBlockingAdvisary(env, entity_type='advisary',
                                    color='red', obs_type='data', position='near-goal')  # red dark_orange
            env.seed(seed + rank)
            # Correct order of wrappers
            if images:
                env = ImageOnlyWrapper(env)
            else:
                env = CoordsOnlyWrapper(env)
            env = TimeLimit(env)
            env._max_episode_steps = 200
            # Monitor should take care of reset!
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return Monitor(env, algorithm_description)
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main(args):
    if args.training:
        env = make_custom_env(args.num_envs, 0, images=args.images)
        if os.path.exists(args.savedir + '.pkl'):
            model = A2C.load(args.savedir + '.pkl',tensorboard_log=args.logdir)
            model.set_env(env)
            model.learn(total_timesteps=args.maxsteps, reset_num_timesteps=False)
        else:
            model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=args.logdir)
            model.learn(total_timesteps=args.maxsteps)
    elif args.gail:
        print('Running GAIL')
        dataset = ExpertDataset(expert_path='./expert_traj/predator_ibl_human.npz',
                            traj_limitation=-1, batch_size=32) # -1 means all trajectories
        # dataset.plot() # plot returns
        env = make_single_env(0,images=args.images)
        model = GAIL('MlpPolicy', env, dataset, verbose=1, tensorboard_log=args.logdir, timesteps_per_batch=32)
        model.learn(total_timesteps=args.maxsteps)

    elif args.bc:
        # dataset = ExpertDataset(expert_path='./expert_traj/predator_ibl_human.npz',
        #                     traj_limitation=-1, batch_size=128)
        # env = make_single_env(0, images=args.images) # Switch one and off this line to compare
        env = make_custom_env(args.num_envs, 0, images=args.images)
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=args.logdir)  # Pretrain the PPO2 model
        # model.pretrain(dataset, n_epochs=1000) # Switch one and off this line to compare
        # As an option, you can train the RL agent#
        model.learn(int(args.maxsteps))
        # Test the pre-trained model
        #env = model.get_env() # Only for ONE ENV
        env = make_single_env(0, images=args.images)
        obs = env.reset()
        reward_sum = 0.0
        epilength = []
        for t in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            # env.render() # DO IT ONLY FOR ONE ENV!!!
            if done:
                print(reward_sum)
                reward_sum = 0.0
                epilength.append(t)
                obs = env.reset()
        env.close()
    model.save(args.savedir, algorithm_description)

if __name__ == '__main__':
    args = argparser()
    main(args)