from collections import namedtuple
# from pysc2.lib import actions

# import pygame
import numpy as np
import sys
from actorcritic.agent import ActorCriticAgent, ACMode
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.util import general_n_step_advantage, combine_first_dimensions, \
    dict_of_lists_to_list_of_dicst, general_nstep_adv_sequential
import tensorflow as tf
from absl import flags
# from time import sleep
from actorcritic.policy import FullyConvPolicy, MetaPolicy, RelationalPolicy,\
    FullyConv3DPolicy, FullyConvPolicyAlt, FactoredPolicy, FactoredPolicy_PhaseI, FactoredPolicy_PhaseII#, LSTM

PPORunParams = namedtuple("PPORunParams", ["lambda_par", "batch_size", "n_epochs"])


class Runner(object):
    def __init__(
            self,
            envs,
            agent: ActorCriticAgent,
            n_steps=5,
            discount=0.99,
            do_training=True,
            ppo_par: PPORunParams = None,
            n_envs=1,
            policy_type = None
    ):
        self.envs = envs
        self.n_envs = n_envs
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.ppo_par = ppo_par
        self.batch_counter = 0
        self.episode_counter = 0
        self.score = 0.0
        # self.policy_type = FullyConvPolicy if ( (policy_type == 'FullyConv') or (policy_type == 'Relational')) else MetaPolicy
        if policy_type == 'FullyConv':
            self.policy_type = FullyConvPolicy
        elif policy_type == 'Relational':
            self.policy_type = RelationalPolicy
        elif policy_type == 'MetaPolicy':
            self.policy_type = MetaPolicy
        elif policy_type == 'FullyConv3D':
            self.policy_type = FullyConv3DPolicy
        elif policy_type == 'AlloAndAlt':
            self.policy_type = FullyConvPolicyAlt
        elif (policy_type == 'Factored'):
            self.policy_type = FactoredPolicy
        elif (policy_type == 'FactoredPolicy_PhaseI'):
            self.policy_type = FactoredPolicy_PhaseI
        elif (policy_type == 'FactoredPolicy_PhaseII'):
            self.policy_type = FactoredPolicy_PhaseII
        else: print('Unknown Policy')

        assert self.agent.mode in [ACMode.A2C, ACMode.PPO]
        self.is_ppo = self.agent.mode == ACMode.PPO
        if self.is_ppo:
            assert ppo_par is not None
            # assert n_steps * envs.n_envs % ppo_par.batch_size == 0
            # assert n_steps * envs.n_envs >= ppo_par.batch_size
            assert n_steps * self.envs.num_envs % ppo_par.batch_size == 0
            assert n_steps * self.envs.num_envs >= ppo_par.batch_size
            self.ppo_par = ppo_par

    def reset(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def reset_demo(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process([obs])

    def _log_score_to_tb(self, score, length): # log score to tensorboard
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        summary.value.add(tag='sc2/episode_length', simple_value=length)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def first_nonzero(self, arr, axis):#, invalid_val=self.n_steps):
        mask = arr != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), self.n_steps)

    def _handle_episode_end(self, timestep, length, last_step_r):
        self.score = timestep
        # print("\rMean Rewards: {:.2f} Episode: {:d}    ".format(
        #     np.mean(self.ep_rewards[-100:]), self.ep_counter), end="")
        # print("\rMean Rewards: {:.2f} Episode: {:d}    ".format(
        #     self.score, self.episode_counter), end="") # To be persistent on the screen it needs to be ran on a main loop, so you print once and just updates the score
        # print(">>>>>>>>>>>>>>>episode %d ended. Score %f | Total Steps %d | Last step Reward %f" % (self.episode_counter, self.score, length, last_step_r))
        print(">>>>>>>>>>>>>>>episode %d ended. Score %f | Total Steps %d" % (
        self.episode_counter, self.score, length))
        self._log_score_to_tb(self.score, length) # logging score to tensorboard
        self.episode_counter += 1 # Is not used for stopping purposes judt for printing. You train for a number of batches (nsteps+training no matter reset)
        #self.reset() # Error if Monitor doesnt have the option to reset without an env to be done (THIS RESETS ALL ENVS!!! YOU NEED remot.send(env.reset) to reset a specific env. Else restart within the env

    def _train_ppo_epoch(self, full_input):
        total_obs = self.n_steps * self.envs.num_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        if self.policy_type == MetaPolicy: # We take out the if from the loop so you choose trainer BEFORE getting into the batch loop
            for b in batches:
                self.agent.train_recurrent(b)
        else:
            for b in batches:
                self.agent.train(b)

    def _train_ppo_recurrent_epoch(self, full_input, rnn_state):
        # HE SHUFFLES SO BE CAREFUL!!! RECHECK IT: rnn_state might need to get in the full_input
        total_obs = self.n_steps * self.envs.num_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        for b in batches:
            self.agent.train_recurrent(b, rnn_state) # IMPORTANT : όταν κανεις training δεν χρειαζεσαι την rnn_State, ξεκινας απο το 0 και αθτη παιρνη την μορφή πουπρεπει να εχει

    def run_batch(self):
        #(MINE) MAIN LOOP!!!
        # The reset is happening through Monitor (except the first one of the first batch (is in hte run_agent)
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)

        latest_obs = self.latest_obs # (MINE) =state(t)

        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, value_estimate = self.agent.step(latest_obs)
            # print('|step:', n, '|actions:', action_ids)  # (MINE) If you put it after the envs.step the SUCCESS appears at the envs.step so it will appear oddly
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            #actions_pp = self.action_processer.process(action_ids) # Actions have changed now need to check: BEFORE: actions.FunctionCall(actions.FUNCTIONS.no_op.id, []) NOW: actions.FUNCTIONS.no_op()
            obs_raw = self.envs.step(action_ids)
            #obs_raw.reward = reward
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)
            #print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))
            mb_rewards[:, n] = [t for t in obs_raw[1]]
            mb_done[:, n] = [t for t in obs_raw[2]]

            #Check for all t (timestep/observation in obs_raw which t has the last state true, meaning it is the last state
            # IF MAX_STEPS OR GOAL REACHED
            if obs_raw[2].any(): # At least one env has done=True
                for i in (np.argwhere(obs_raw[2])): # Run the loop for ONLY the envs that have finished
                    indx = i[0] # i will contain in [] the index of the env that has finished
                    epis_reward = obs_raw[3][indx]['episode']['r']
                    epis_length = obs_raw[3][indx]['episode']['l']
                    last_step_r = obs_raw[1][indx]
                    self._handle_episode_end(epis_reward, epis_length, last_step_r)
                    # self.latest_obs[indx] = self.envs
            # You can use as below for obs_raw[4] which is success of failure
            #print(obs_raw[2])
            # indx=0 # env count
            # for t in obs_raw[2]: # Monitor returns additional stuff such as epis_reward and epis_length etc apart the obs, r, done, info
            #     #obs_raw[2] = done = [True, False, False, True,...] each element corresponds to an env
            #     if t == True: # done=true
            #         # Put reward in scores
            #         epis_reward = obs_raw[3][indx]['episode']['r']
            #         epis_length = obs_raw[3][indx]['episode']['l']
            #         last_step_r = obs_raw[1][indx]
            #         self._handle_episode_end(epis_reward, epis_length, last_step_r) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                indx = indx + 1 # finished envs count
            # for t in obs_raw:
            #     if t.last():
            #         self._handle_episode_end(t)

        #print(">> Avg. Reward:",np.round(np.mean(mb_rewards),3))
        mb_values[:, -1] = self.agent.get_value(latest_obs) # We bootstrap from last step. If not terminal! although he doesnt use any check here
        # R, G = calc_rewards(self, mb_rewards, mb_values[:,:-1], mb_values[:,1:], mb_done, self.discount)
        # R = R.reshape((self.n_envs, self.n_steps)) # self.n_envs returns 1, self.envs.num_envs
        # G = G.reshape((self.n_envs, self.n_steps))
        # He replaces the last value with a bootstrap value. E.g. s_t-->V(s_t),s_{lastnstep}-->V(s_{lastnstep}), s_{lastnstep+1}
        # and we replace the V(s_{lastnstep}) with V(s_{lastnstep+1})
        n_step_advantage = general_nstep_adv_sequential(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0, # change lambda to  something between 1 and 0 else you do not get the nstep reward calculation
            nenvs=self.n_envs,
            maxsteps=self.n_steps
        )
        # n_step_advantage = general_n_step_advantage(
        #     mb_rewards,
        #     mb_values,
        #     self.discount,
        #     mb_done,
        #     lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        # )
        # disc_ret = calculate_n_step_reward(
        #     mb_rewards,
        #     self.discount,
        #     mb_values,
        # )
        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays CHECK THIS OUT!!!YOU HAVE ALREADY THE TIMESTEP DIM!!!
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),#, G.transpose()
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()# R.transpose()# (we left out the last element) if you add to the advantage the value you get the target for your value function training. Check onenote in cmu_gym/Next Steps
        }
        #(MINE) Probably we combine all experiences from every worker below
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        # Below comment it out for LSTM
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()} # The function takes in nsteps x nenvs x [dims] and combines nsteps*nenvs x [dims].

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1 # It is used only for printing reasons as the outer while loop takes care to stop the number of batches
        print('Update %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_factored_batch(self):
        #(MINE) MAIN LOOP!!!
        # The reset is happening through Monitor (except the first one of the first batch (is in hte run_agent)
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_values_goal = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_values_fire = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_rewards_goal = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_rewards_fire = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)

        latest_obs = self.latest_obs # (MINE) =state(t)
        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, value_estimate_goal, value_estimate_fire, value_estimate = self.agent.step_factored(latest_obs)
            # print('|step:', n, '|actions:', action_ids)  # (MINE) If you put it after the envs.step the SUCCESS appears at the envs.step so it will appear oddly
            if np.random.random() < 0.20: action_ids = np.array([self.envs.action_space.sample() for _ in range(self.envs.num_envs)])
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_values_goal[:, n] = value_estimate_goal
            mb_values_fire[:, n] = value_estimate_fire
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            #actions_pp = self.action_processer.process(action_ids) # Actions have changed now need to check: BEFORE: actions.FunctionCall(actions.FUNCTIONS.no_op.id, []) NOW: actions.FUNCTIONS.no_op()
            obs_raw = self.envs.step(action_ids)
            #obs_raw.reward = reward
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)
            #print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))
            mb_rewards[:, n] = [t for t in obs_raw[1]]
            temp = [t for t in obs_raw[3]]
            mb_rewards_goal[:,n] = [d['goal'] for d in temp]
            mb_rewards_fire[:, n] = [d['fire'] for d in temp]
            mb_done[:, n] = [t for t in obs_raw[2]]

            # IF MAX_STEPS OR GOAL REACHED
            if obs_raw[2].any(): # At least one env has done=True
                for i in (np.argwhere(obs_raw[2])): # Run the loop for ONLY the envs that have finished
                    indx = i[0]
                    epis_reward = obs_raw[3][indx]['episode']['r']
                    epis_length = obs_raw[3][indx]['episode']['l']
                    last_step_r = obs_raw[1][indx]
                    self._handle_episode_end(epis_reward, epis_length, last_step_r)

            # indx=0 # env count
            # for t in obs_raw[2]: # Monitor returns additional stuff such as epis_reward and epis_length etc apart the obs, r, done, info
            #     #obs_raw[2] = done = [True, False, False, True,...] each element corresponds to an env
            #     if t == True: # done=true
            #         # Put reward in scores
            #         epis_reward = obs_raw[3][indx]['episode']['r']
            #         epis_length = obs_raw[3][indx]['episode']['l']
            #         last_step_r = obs_raw[1][indx]
            #         self._handle_episode_end(epis_reward, epis_length, last_step_r) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
            #     indx = indx + 1 # finished envs count
            # for t in obs_raw:
            #     if t.last():
            #         self._handle_episode_end(t)

        #print(">> Avg. Reward:",np.round(np.mean(mb_rewards),3))
        mb_values_goal[:, -1], mb_values_fire[:, -1], mb_values[:, -1] = self.agent.get_factored_value(latest_obs) # We bootstrap from last step. If not terminal! although he doesnt use any check here
        # R, G = calc_rewards(self, mb_rewards, mb_values[:,:-1], mb_values[:,1:], mb_done, self.discount)
        # R = R.reshape((self.n_envs, self.n_steps)) # self.n_envs returns 1, self.envs.num_envs
        # G = G.reshape((self.n_envs, self.n_steps))
        # He replaces the last value with a bootstrap value. E.g. s_t-->V(s_t),s_{lastnstep}-->V(s_{lastnstep}), s_{lastnstep+1}
        # and we replace the V(s_{lastnstep}) with V(s_{lastnstep+1})
        n_step_advantage_goal = general_nstep_adv_sequential(
            mb_rewards_goal,
            mb_values_goal,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0,
            nenvs=self.n_envs,
            maxsteps=self.n_steps
        )
        n_step_advantage_fire = general_nstep_adv_sequential(
            mb_rewards_fire,
            mb_values_fire,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0,
            nenvs=self.n_envs,
            maxsteps=self.n_steps
        )
        # n_step_advantage = n_step_advantage_goal + n_step_advantage_fire
        n_step_advantage = general_nstep_adv_sequential(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0,
            nenvs=self.n_envs,
            maxsteps=self.n_steps
        )
        # disc_ret = calculate_n_step_reward(
        #     mb_rewards,
        #     self.discount,
        #     mb_values,
        # )
        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays CHECK THIS OUT!!!YOU HAVE ALREADY THE TIMESTEP DIM!!!
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),#, G.transpose()
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose(),
            FEATURE_KEYS.value_target_goal: (n_step_advantage_goal + mb_values_goal[:, :-1]).transpose(),# R.transpose()# (we left out the last element) if you add to the advantage the value you get the target for your value function training. Check onenote in cmu_gym/Next Steps
            FEATURE_KEYS.value_target_fire: (n_step_advantage_fire + mb_values_fire[:, :-1]).transpose()
        }
        # full_input = {
        #     # these are transposed because action/obs
        #     # processers return [time, env, ...] shaped arrays CHECK THIS OUT!!!YOU HAVE ALREADY THE TIMESTEP DIM!!!
        #     FEATURE_KEYS.advantage: n_step_advantage.transpose(),#, G.transpose()
        #     FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()# R.transpose()# (we left out the last element) if you add to the advantage the value you get the target for your value function training. Check onenote in cmu_gym/Next Steps
        # }
        #(MINE) Probably we combine all experiences from every worker below
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        # Below comment it out for LSTM
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()} # The function takes in nsteps x nenvs x [dims] and combines nsteps*nenvs x [dims].

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1 # It is used only for printing reasons as the outer while loop takes care to stop the number of batches
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_meta_batch(self):
        mb_actions = []
        mb_obs = []
        mb_rnns = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32) # n x d array (ndarray)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)

        # EVERYTHING IS HAPPENING ON PARALLEL!!!
        r_=np.zeros((self.envs.num_envs, 1), dtype=np.float32) # Instead of 1 you might use n_steps
        a_=np.zeros((self.envs.num_envs), dtype=np.int32)
        latest_obs = self.latest_obs # (MINE) =state(t)
        # rnn_state = self.agent.theta.state_init
        rnn_state = self.agent.theta.state_init
        for n in range(self.n_steps):
            action_ids, value_estimate, rnn_state_new = self.agent.step_recurrent(latest_obs, rnn_state, r_, a_) # Automatically returns [num_envs, outx] for each outx you want
            # print('|step:', n, '|actions:', action_ids)
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            obs_raw = self.envs.step(action_ids)
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)

            mb_rnns.append(rnn_state)
            rnn_state = rnn_state_new
            r_ = obs_raw[1] # (nenvs,) but you need (nenvs,1)
            r_ = np.reshape(r_,[self.envs.num_envs,1]) # gets into recurrency as [nenvs,1] # The 1 might be used as timestep
            a_ = action_ids

            mb_rewards[:, n] = [t for t in obs_raw[1]]
            mb_done[:, n] = [t for t in obs_raw[2]]

            # Shouldnt this part below be OUT of the nstep loop? NO: You check if done=True and you extract the additional info that Monitor outputs
            indx=0 # env count
            for t in obs_raw[2]: # Monitor returns additional stuff such as epis_reward and epis_length etc apart the obs, r, done, info
                # obs_raw[2] = done = [True, False, False, True,...] each element corresponds to an env (index gives the env)
                if t == True: # done=true
                    # Put reward in scores
                    epis_reward = obs_raw[3][indx]['episode']['r']
                    epis_length = obs_raw[3][indx]['episode']['l'] # EPISODE LENGTH HERE!!!
                    last_step_r = obs_raw[1][indx]
                    self._handle_episode_end(epis_reward, epis_length, last_step_r) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                    # Here you have to reset the rnn_state of that env (rnn_state has h and c EACH has dims [batch_size x hidden dims]
                    rnn_state[0][indx] = np.zeros(256)
                    rnn_state[1][indx] = np.zeros(256)
                    #reset the relevant r_ and a_
                    r_[indx] = 0
                    a_[indx] = 0

                indx = indx + 1 # finished envs count
        # Below: if all are zeros then you get nsteps
        mb_l = self.first_nonzero(mb_done,axis=1) # the last obs s-->sdone are not getting in the training!!!This is because sdone--> ? and R=0
        # Substitute 0s with 1 declaring 1 step
        # mb_l[mb_l==0] = 1
        for b in range(mb_l.shape[0]):
            if mb_l[b] < self.n_steps:
                mb_l[b] = mb_l[b] + 1 # You start stepping from step 0 to 1
        mask = ~(np.ones(mb_rewards.shape).cumsum(axis=1).T > mb_l).T
        mb_rewards = mb_rewards*mask
        # Below: r + gammaV(s')(1-done) - V(s)// V(s')=mb_values[:,-1] but if its done we dont care if we put the last nstep V or the actual V(s_done+1) as the whole term is gonna be zero
        # From V(nstep) estimate the expected reward - what if though we finished the sequence earlier???
        # This is for the last obs which you do not store. All other values that will be used as targets are available
        # mb_values[:, -1] = self.agent.get_recurrent_value(latest_obs, rnn_state, r_, a_) # Put at last slot the estimated future expected reward for bootstrap the V after the nsteps
        mask_v = ~(np.ones(mb_values.shape).cumsum(axis=1).T > mb_l).T
        mb_values = mb_values*mask_v
        # We take the vector of values after nsteps and we use ONLY the valid ones in the loop below
        vec = self.agent.get_recurrent_value(latest_obs, rnn_state, r_, a_)
        for b in range(mb_l.shape[0]):
            if mb_l[b] == self.n_steps:
                mb_values[b, -1] = vec[b] # if the sequence didnt end within nsteps then we add the value so the term gammaVt+1(1-done) is not 0
        # Mask below the values that enter the nstep advantage
        n_step_advantage = general_nstep_adv_sequential(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0,
            nenvs=self.n_envs,
            maxsteps=self.n_steps
        )
        # n_step_advantage = general_n_step_advantage(
        #     mb_rewards,
        #     mb_values,
        #     self.discount,
        #     mb_done,
        #     lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        # )
        # prev_rewards = [0] + mb_rewards[:, :-1]#.tolist() # from the rewards you take out the last element and replace it with 0
        prev_rewards = np.c_[np.zeros((self.envs.num_envs, 1), dtype=np.float32), mb_rewards[:, :-1]]
        # Below we add one zero action element and we take out the a_t so we get at=0:t-1
        prev_actions = [np.zeros((self.envs.num_envs), dtype=np.int32)] + mb_actions[:-1] # You have to pad this probably to have equal lengths with your data in terms of nsteps
        full_input = {
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose() # NETWORK TARGET (A+V=R)
        }

        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        # full_input.update(self.action_processer.combine_batch(prev_actions)) # THIS FEEDS AGAIN THE SELECTED_ID_ACTION PLACEHOLDER!!!!
        # full_input.update(self.action_processer.combine_batch(prev_rewards)) # THIS FEEDS AGAIN THE SELECTED_ID_ACTION PLACEHOLDER!!!!
        # Below: [maxsteps x batch] --> [batch x maxsteps]
        full_input = {k: np.swapaxes(v, 0, 1) for k, v in full_input.items()} # obs should be first trasnposed and then combine else you loose the time order
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()} # YOU CAN COMMENT THIS AND UNCOMMENT BELOW
        # full_input = {k: np.swapaxes(v,0,1) for k, v in full_input.items()}

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            if self.policy_type == MetaPolicy:
                self.agent.train_recurrent(full_input,mb_l,prev_rewards,prev_actions)
            else:
                self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_trained_batch(self):

        latest_obs = self.latest_obs # (MINE) =state(t)

        action_ids, value_estimate, fc, action_probs = self.agent.step_eval(latest_obs) # (MINE) AGENT STEP = INPUT TO NN THE CURRENT STATE AND OUTPUT ACTION
        print('|actions:', action_ids)
        obs_raw = self.envs.step(action_ids[0]) # It will also visualize the next observation if all the episodes have ended as after success it retunrs the obs from reset
        latest_obs = self.obs_processer.process(obs_raw[0:-3])  # Take only the first element which is the rgb image and ignore the reward, done etc
        print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))

        # if obs_raw[2]: # done is True
        #     # for r in obs_raw[1]: # You will double count here as t
        #     self._handle_episode_end(obs_raw[1])  # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently

        self.latest_obs = latest_obs # (MINE) state(t) = state(t+1), the usual s=s'
        self.batch_counter += 1
        #print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()
        return obs_raw[0:-3], action_ids[0], value_estimate[0], obs_raw[1], obs_raw[2], obs_raw[3], fc, action_probs

    def run_trained_factored_batch(self):

        latest_obs = self.latest_obs # (MINE) =state(t)

        action_ids, value_estimate, value_estimate_goal, value_estimate_fire, fc, action_probs = self.agent.step_eval_factored(latest_obs) # (MINE) AGENT STEP = INPUT TO NN THE CURRENT STATE AND OUTPUT ACTION
        print('|actions:', action_ids)
        obs_raw = self.envs.step(action_ids) # It will also visualize the next observation if all the episodes have ended as after success it retunrs the obs from reset
        latest_obs = self.obs_processer.process(obs_raw[0:-3])  # Take only the first element which is the rgb image and ignore the reward, done etc
        print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))

        self.latest_obs = latest_obs # (MINE) state(t) = state(t+1), the usual s=s'
        self.batch_counter += 1
        #print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()
        return obs_raw[0:-3], action_ids[0], value_estimate[0], value_estimate_goal[0], value_estimate_fire[0], obs_raw[1], obs_raw[2], obs_raw[3], fc, action_probs
