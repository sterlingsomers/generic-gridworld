import os
# print(os.getcwd())
# os.chdir('path) # CHANGE
import time
from datetime import datetime
import gym
from envs.generic_env import GenericEnv
import logging
import scipy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from baselines import logger
from baselines.bench import Monitor
# from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import os
import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions, pad='same'):
        super().__init__('cnn_policy')
        # no tf.get_variable(), just simple Keras API
        self.conv1 = kl.Conv2D(32, 4, 2, activation='relu', padding=pad)
        self.conv2 = kl.Conv2D(32, 3, 1, activation='relu', padding=pad)
        self.flatten = kl.Flatten()
        self.fc = kl.Dense(256, activation='relu')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits') # Just a layer without the activation tf.nn.softmax function
        self.value = kl.Dense(1, name='value')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # tf.reshape(inputs,(-1, tf.shape(inputs[1:])))
        x = tf.cast(inputs, tf.float32) / 255.
        x = tf.convert_to_tensor(x)
        # separate hidden layers from the same input tensor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        hidden_logs = self.fc(x)
        hidden_vals = self.fc(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return action, np.squeeze(value, axis=-1) # squeeze reduces the redundant dim e.g. (num_envs x 1) to (num_envs,)

class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.0007),#ko.RMSprop(lr=0.0007), # USE ADAM?,
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def discount(self, x):
        return scipy.signal.lfilter([1], [1, - self.params['gamma']], x[::-1], axis=0)[::-1]

    def train_batch(self, envs, n_steps=32, updates=1000):
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.create_file_writer(logdir)
        writer.set_as_default()

        num_envs = env.num_envs

        latest_obs = envs.reset()
        for update in range(updates):
            mb_actions = np.zeros((num_envs,) + (n_steps,),dtype=np.int32)
            mb_obs = np.zeros(((num_envs,) + (n_steps,) + env.observation_space.shape), dtype=np.float32)
            mb_values = np.zeros((envs.num_envs, n_steps + 1), dtype=np.float32)
            mb_rewards = np.zeros((envs.num_envs, n_steps), dtype=np.float32)
            mb_done = np.zeros((envs.num_envs, n_steps), dtype=np.int32)
            for n in range(n_steps):
                mb_actions[:, n], mb_values[:, n] = self.model.action_value(latest_obs)
                # print('|step:', n, '|actions:', action_ids)  # (MINE) If you put it after the envs.step the SUCCESS appears at the envs.step so it will appear oddly
                mb_obs[:, n] = latest_obs.copy()
                print('### Update:', update, '### nstep:', n)
                latest_obs, mb_rewards[:, n], mb_done[:, n], info = envs.step(mb_actions[:, n])

                indx = 0  # env count
                for t in mb_done[:, n]:
                    if t == True:  # if done=true report score
                        # # Put reward in scores
                        epis_reward = info[indx]['episode']['r']
                        epis_length = info[indx]['episode']['l']
                        last_step_r = mb_rewards[indx, n]
                        # print('\renv {:d} Done with Reward collected {:.2f} after {:d} steps'.format(indx, np.sum(mb_rewards[indx]), mb_rewards[indx].size ) )
                        print(">>>>>>>>>>>>>>>update %d ended. Score %f | Total Steps %d" % (
                            update, epis_reward, epis_length))
                        # self._handle_episode_end(epis_reward, epis_length,
                        #                          last_step_r)  # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                    indx = indx + 1  # finished envs count

            _, mb_values[:, -1] = self.model.action_value(latest_obs)
            n_step_advantage = self.general_nstep_adv_sequential(
                mb_rewards,
                mb_values,
                mb_done,
                # filled and unfilled inputs should be separated and grouped
                discount=0.95,
                lambda_par= 1.0, # change lambda to  something between 1 and 0 else you do not get the nstep reward calculation
                nenvs=num_envs,
                maxsteps=n_steps
            )
            returns = n_step_advantage + mb_values[:, :-1]

            # convert obs, targets dims to [batch*maxsteps x image] and [batch*maxsteps x restdims] # WE HAVE TO (but we do not know where keras computes the input dims cauz if we give it with the timesteps it returns erros for wrong dim inputs
            mb_obs = mb_obs.reshape(((-1,) + envs.observation_space.shape)) # -1 for batch*maxsteps
            returns = returns.reshape(-1) # [64,1]
            # a trick to input actions and advantages through same API
            acts_and_advs = np.stack((mb_actions.reshape((-1,)), n_step_advantage.reshape((-1,)))).transpose() # [64,2]
            # train_on_batch: Runs a single gradient update on a single batch of collected data.
            # train_on_batch might consume too much CPU!!! YES ITS AN ISSUE!!!
            losses = self.model.train_on_batch(mb_obs, [acts_and_advs, returns]) # [total loss, pg + entropy, value,metrics=None]. The logits loss needs selected actions and the bellman error needs returns
            # DO NOT CHECK GRADS AFTER THE PREVIOUS LINE: MODEL PARAMS HAVE BEEN UPDATED SO PREDICTIONS WILL BE DIFFICULT
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            # if update % 20 == 0: # every 20 updates we plot
            tf.summary.scalar('value_loss', 0.5, step=update)
            writer.flush()

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                # env.render()
                env.renderEnv()
        return ep_reward

    # TODO: Import from util.py
    def general_nstep_adv_sequential(self,
            mb_rewards: np.ndarray,
            mb_values: np.ndarray,
            mb_done: np.ndarray,
            discount: float,
            lambda_par: float,
            nenvs: int,
            maxsteps: int):
        gae = np.zeros((nenvs, maxsteps))
        batch = np.arange(nenvs)
        ind = np.array(
            np.nonzero(mb_done))  # [batch x maxsteps]. first row indicate the batch and second the index where done=1
        for b in batch:
            told = 0
            adv = []
            indices = ind[1, np.where(ind[0] == b)[0]] + 1  # take the appropriate index in which done=1 per batch b
            indices = np.insert(indices, 0, 0, axis=0)  # insert a 0 in front so the first index will be 0 to ...
            if indices[-1] != (maxsteps):
                indices = np.insert(indices, indices.size, maxsteps, axis=0)
            for t in range(indices.size):
                # print('t=', t, 'told=', told)
                # print('ind_t=', indices[t], 'ind_t-1=', indices[told])
                d = mb_done[b, indices[told]:indices[t]]
                # print('done', d)
                r = mb_rewards[b, indices[told]:indices[t]]
                r = r.reshape([1, r.size])
                # print('reward', r)
                v = mb_values[b, indices[told]:indices[t] + 1]
                # print('values', v)
                batch_size, timesteps = r.shape  # [1,18]
                delta = r + discount * v[1:] * (1 - d) - v[:-1]
                delta_rev = delta[:, ::-1]
                adjustment = (discount * lambda_par) ** np.arange(timesteps, 0, -1)
                advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]
                adv = np.concatenate([adv, advantage[0]])
                told = t
            gae[b] = adv
        return gae


    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, value) # for debugging with the eager execution (not with th egraph) do self.params['value'] * kls.mean_squared_error(returns, values).numpy()

    def _logits_loss(self, acts_and_advs, logits): # logits should be [numsteps*batch x num_actions]
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities. "Sparse" means that we can use actions as integers and not one-hot encoded as in the non sparse version.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)  # wiht Keras you can define the function and call it later (l:126)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32) # convert actions back to integers so they can be used by the env
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)  # Ai*log(pi)= sample_weight*CE, dim = [1]
        # entropy loss can be calculated via CE over itself. For probs calculation: tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)  # pi*log(pi)
        # entropy_loss = tf.reduce_mean(entropy_loss)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy'] * entropy_loss

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = envs.generic_env.GenericEnv(map='small-empty',features=[{'class':'feature','type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
            predator = ChasingBlockingAdvisary(env, entity_type='advisary', color='red', obs_type='data')
            network_agent = NetworkAgent(env, color='aqua')
            env.seed(seed + rank) # DONT SEED (OR USE SAME SEED) IF YOU WANT TO REPLICATE RESULTS
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    eager_exec = True
    num_envs = 1
    env_name = 'FireGrid' # CartPole-v0, MsPacman-v4
    # env = gym.make('CartPole-v0')
    # env = gameEnv(partial=False, size=9)
    env = make_custom_env(env_name, num_envs, 1)
    print('ok')


    if eager_exec==True:
        '''Comment for graph execution'''
        print("Eager Execution:", tf.executing_eagerly())
        model = Model(num_actions=env.action_space.n)
        agent = A2CAgent(model)

        start_time = time.time()
        # rewards_history = agent.train(env)
        # agent.train(env)
        agent.train_batch(env)
        print("--- %s seconds ---" % (time.time() - start_time)) # (MINE)
        print("Finished training.")
        # print("Total Episode Reward: %d out of 200" % agent.test(env, True))
    else:
        '''Uncomment for graph execution'''
        # With the graph from ~22mins it does ~8mins for 1000 updates on 32 steps
        with tf.Graph().as_default():
            print("Eager Execution:", tf.executing_eagerly())  # False

            model = Model(num_actions=env.action_space.n)
            agent = A2CAgent(model)

            start_time = time.time()
        #     rewards_history = agent.train(env)
        #     agent.train(env)
            agent.train_batch(env)
            print("--- %s seconds ---" % (time.time() - start_time))  # (MINE)
            print("Finished training, testing...")
        #     print("Total Episode Reward: %d out of 200" % agent.test(env))

    # plt.style.use('seaborn')
    # plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.show()