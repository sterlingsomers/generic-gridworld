import os
import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *

from baselines import logger
from baselines.bench import Monitor
# from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import gym

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = envs.generic_env.GenericEnv(map='small-empty',features=[{'class':'feature','type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
            predator = ChasingBlockingAdvisary(env, entity_type='advisary', color='red', obs_type='data')
            # ai = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
            env.seed(seed + rank) # DONT SEED (OR USE SAME SEED) IF YOU WANT TO REPLICATE RESULTS
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    eager_exec = True
    num_envs = 2
    env_name = 'FireGrid' # CartPole-v0, MsPacman-v4
    # env = gym.make('CartPole-v0')
    # env = gameEnv(partial=False, size=9)
    env = make_custom_env(env_name, num_envs, 1)
    print('GO')