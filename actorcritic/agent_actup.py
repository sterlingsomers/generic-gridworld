import collections
import os
import numpy as np
import tensorflow as tf
# from pysc2.lib import actions
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES
from actorcritic.policy import FullyConvPolicy, MetaPolicy, RelationalPolicy,\
    FullyConvPolicyAlt, FullyConv3DPolicy, FactoredPolicy, FactoredPolicy_PhaseI, FactoredPolicy_PhaseII#, LSTM
from common.preprocess import ObsProcesser, FEATURE_KEYS, AgentInputTuple
from common.util import weighted_random_sample, select_from_each_row, ravel_index_pairs
from pyactup import *
import pickle
import math
import tensorboard.plugins.beholder as beholder_lib
# import saliency

#LOG_DIRECTORY = '/tmp/beholder-demo/SCII'
LOG_DIRECTORY = '_files/summaries/Test'
def _get_placeholders(spatial_dim, nsteps, nenvs, policy_type, obs_d):
    sd = spatial_dim
    if policy_type == 'MetaPolicy':
        feature_list = [
        (FEATURE_KEYS.alt0_grass, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt0_bush, tf.float32, [None, 20, 20]),
    # FEATURE_KEYS.available_action_ids: get_available_actions_flags(obs),
        (FEATURE_KEYS.alt0_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt0_hiker, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt1_pine, tf.float32, [None, 20, 20]),  # numpy.array is redundant
        (FEATURE_KEYS.alt1_pines, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt1_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt2_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt3_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.minimap_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_MINIMAP_CHANNELS]),
        (FEATURE_KEYS.screen_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_SCREEN_CHANNELS]),
        (FEATURE_KEYS.screen_unit_type, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.is_spatial_action_available, tf.float32, [None]),
        # (FEATURE_KEYS.available_action_ids, tf.float32, [None, len(actions.FUNCTIONS)]),
        (FEATURE_KEYS.selected_spatial_action, tf.int32, [None, 2]),
        (FEATURE_KEYS.selected_action_id, tf.int32, [None]),
        (FEATURE_KEYS.value_target, tf.float32, [None]),
        (FEATURE_KEYS.value_target_goal, tf.float32, [None]),
        (FEATURE_KEYS.value_target_fire, tf.float32, [None]),
        (FEATURE_KEYS.rgb_screen, tf.float32, [nenvs, None, obs_d[0], obs_d[1], 3]), #[None, 32, 100, 100, 3] for LSTM
        (FEATURE_KEYS.alt_view, tf.float32, [nenvs, None, obs_d[0], obs_d[1], 3]), #[None, 32, 100, 100, 3] for LSTM
        (FEATURE_KEYS.player_relative_screen, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.player_relative_minimap, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.advantage, tf.float32, [None]),
        (FEATURE_KEYS.prev_actions, tf.int32, [None, None]),
        (FEATURE_KEYS.prev_rewards, tf.float32, [None, None]),
        (FEATURE_KEYS.altitudes, tf.int32, [None]),
        (FEATURE_KEYS.image_vol, tf.float32, [None, 5, 100, 100, 3]),
        (FEATURE_KEYS.joined, tf.float32, [None, 100, 200, 3]),
    ]
    else:
        feature_list = [
            (FEATURE_KEYS.alt0_grass, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt0_bush, tf.float32, [None, 20, 20]),
            # FEATURE_KEYS.available_action_ids: get_available_actions_flags(obs),
            (FEATURE_KEYS.alt0_drone, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt0_hiker, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt1_pine, tf.float32, [None, 20, 20]),  # numpy.array is redundant
            (FEATURE_KEYS.alt1_pines, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt1_drone, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt2_drone, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.alt3_drone, tf.float32, [None, 20, 20]),
            (FEATURE_KEYS.minimap_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_MINIMAP_CHANNELS]),
            (FEATURE_KEYS.screen_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_SCREEN_CHANNELS]),
            (FEATURE_KEYS.screen_unit_type, tf.int32, [None, sd, sd]),
            (FEATURE_KEYS.is_spatial_action_available, tf.float32, [None]),
            # (FEATURE_KEYS.available_action_ids, tf.float32, [None, len(actions.FUNCTIONS)]),
            (FEATURE_KEYS.selected_spatial_action, tf.int32, [None, 2]),
            (FEATURE_KEYS.selected_action_id, tf.int32, [None]),
            (FEATURE_KEYS.value_target, tf.float32, [None]),
            (FEATURE_KEYS.value_target_goal, tf.float32, [None]),
            (FEATURE_KEYS.value_target_fire, tf.float32, [None]),
            (FEATURE_KEYS.rgb_screen, tf.float32, [None, obs_d[0], obs_d[1], 3]),
            (FEATURE_KEYS.alt_view, tf.float32, [None, obs_d[0], obs_d[1], 3]),
            (FEATURE_KEYS.player_relative_screen, tf.int32, [None, sd, sd]),
            (FEATURE_KEYS.player_relative_minimap, tf.int32, [None, sd, sd]),
            (FEATURE_KEYS.advantage, tf.float32, [None]),
            (FEATURE_KEYS.prev_actions, tf.int32, [None, None]),
            (FEATURE_KEYS.prev_rewards, tf.float32, [None, None]),
            (FEATURE_KEYS.altitudes, tf.int32, [None]),
            (FEATURE_KEYS.image_vol, tf.float32, [None, 5, 100, 100, 3]),
            (FEATURE_KEYS.joined, tf.float32, [None, 100, 200, 3]),
        ]
    return AgentInputTuple(
        **{name: tf.placeholder(dtype, shape, name) for name, dtype, shape in feature_list}
    )

class ACMode:
    A2C = "a2c"
    PPO = "ppo"


SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "total"])


class ActorCriticAgent:
    _scalar_summary_key = "scalar_summaries"

    def __init__(self,
            sess: tf.Session,
            summary_path: str,
            all_summary_freq: int,
            scalar_summary_freq: int,
            spatial_dim: int,
            mode: str,
            clip_epsilon=0.2,
            unit_type_emb_dim=4,
            loss_value_weight=1.0,
            entropy_weight_spatial=1e-6,
            entropy_weight_action_id=1e-5,
            max_gradient_norm=None,
            optimiser="adam",
            optimiser_pars: dict = None,
            policy=None,
            num_actions=4,
            num_envs=1,
            nsteps=1,
            obs_dim = None,
    ):
        """
        Actor-Critic Agent for learning pysc2-minigames
        https://arxiv.org/pdf/1708.04782.pdf
        https://github.com/deepmind/pysc2

        Can use
        - A2C https://blog.openai.com/baselines-acktr-a2c/ (synchronous version of A3C)
        or
        - PPO https://arxiv.org/pdf/1707.06347.pdf

        :param summary_path: tensorflow summaries will be created here
        :param all_summary_freq: how often save all summaries
        :param scalar_summary_freq: int, how often save scalar summaries
        :param spatial_dim: dimension for both minimap and screen
        :param mode: a2c or ppo
        :param clip_epsilon: epsilon for clipping the ratio in PPO (no effect in A2C)
        :param loss_value_weight: value weight for a2c update
        :param entropy_weight_spatial: spatial entropy weight for a2c update
        :param entropy_weight_action_id: action selection entropy weight for a2c update
        :param max_gradient_norm: global max norm for gradients, if None then not limited
        :param optimiser: see valid choices below
        :param optimiser_pars: optional parameters to pass in optimiser
        :param policy: Policy class
        """

        assert optimiser in ["adam", "rmsprop"]
        assert mode in [ACMode.A2C, ACMode.PPO]
        self.mode = mode
        self.sess = sess
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.unit_type_emb_dim = unit_type_emb_dim
        self.summary_path = summary_path
        os.makedirs(summary_path, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.all_summary_freq = all_summary_freq
        self.scalar_summary_freq = scalar_summary_freq
        self.train_step = 0
        self.max_gradient_norm = max_gradient_norm
        self.clip_epsilon = clip_epsilon
        self.num_actions= num_actions
        self.num_envs = num_envs
        self.nsteps = nsteps
        self.obs_dims = obs_dim
        self.policy_type = policy
        # self.policy = FullyConvPolicy if ( (policy == 'FullyConv') or (policy == 'Relational')) else MetaPolicy
        if policy == 'FullyConv':
            self.policy = FullyConvPolicy
        elif policy == 'Relational':
            self.policy = RelationalPolicy
        elif policy == 'MetaPolicy':
            self.policy = MetaPolicy
        elif policy == 'FullyConv3D':
            self.policy = FullyConv3DPolicy
        elif policy == 'AlloAndAlt':
            self.policy = FullyConvPolicyAlt
        elif (policy == 'FactoredPolicy'):
            self.policy = FactoredPolicy
        elif policy == 'FactoredPolicy_PhaseI':
            self.policy = FactoredPolicy_PhaseI
        elif policy == 'FactoredPolicy_PhaseII':
            self.policy = FactoredPolicy_PhaseII
        else: print('Unknown Policy')

        # assert (self.policy_type == 'MetaPolicy') and not (self.mode == ACMode.PPO) # For now the policy in PPO is not calculated taken into account recurrencies

        opt_class = tf.train.AdamOptimizer if optimiser == "adam" else tf.train.RMSPropOptimizer
        if optimiser_pars is None:
            pars = {
                "adam": {
                    "learning_rate": 1e-4, # orig:4
                    "epsilon": 5e-7
                },
                "rmsprop": {
                    "learning_rate": 2e-4
                }
            }[optimiser]
        else:
            pars = optimiser_pars
        self.optimiser = opt_class(**pars)

        #sterling's stuff
        self.memory = Memory(noise=0.0,decay=0.0,temperature=t,threshold=-100.0,mismatch=10,optimized_learning=False)
        set_similarity_function(self.angle_similarity, *['goal_rads', 'adisary_rads'])
        set_similarity_function(self.distance_similarity, *['goal_disdtance', 'advisary_distance'])

        self.data = pickle.load(open('...','rb'))
        #Before using the distances, they have to be normalized (0 to 1)
        #Normalize by dividing by the max in the data
        distances = []
        for x in self.data:
            distances.append(x['goal_distance'])
            distances.append(x['advisary_distance'])
        #distances = [x['goal_distance'],x['advisary_distance'] for x in self.data]
        self.max_distance = max(distances)
        for datum in self.data:
            datum['goal_distance'] = datum['goal_distance'] / self.max_distance
            datum['advisary_distance'] = datum['advisary_distance'] / self.max_distance

        for chunk in self.data:
            self.memory.learn(**chunk)

    def angle_similarity(self, x, y):
        PI = math.pi
        TAU = 2 * PI
        result = min((2 * PI) - abs(x - y), abs(x - y))
        normalized = result / TAU
        xdeg = math.degrees(x)
        ydeg = math.degrees(y)
        resultdeg = math.degrees(result)
        normalized2 = resultdeg / 180
        # print("sim anle", 1 - normalized2)
        return 1 - normalized2

    def distance_similarity(self, x, y):
        x = x / self.max_distance
        result = 1 - abs(x - y)
        # print("sim distance", result, x, y)
        return result

    def gridmap_to_symbols(self, gridmap, agent, value_to_objects):
        agent_location = np.where(gridmap == agent)
        agent_location = (int(agent_location[0]), int(agent_location[1]))
        goal_location = 0
        advisary_location = 0
        return_dict = {}
        for stuff in value_to_objects:
            if 'entity_type' in value_to_objects[stuff]:
                if value_to_objects[stuff]['entity_type'] == 'goal':
                    goal_location = np.where(gridmap == stuff)
                if value_to_objects[stuff]['entity_type'] == 'advisary':
                    advisary_location = np.where(gridmap == stuff)
        if goal_location:
            goal_location = (int(goal_location[0]), int(goal_location[1]))
            goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])
            path_agent_to_goal = self.env.getPathTo(agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['goal_rads'] = goal_rads
            return_dict['goal_distance'] = len(points_in_path) / self.max_distance
        if advisary_location:
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
                                       advisary_location[1] - agent_location[1])
            path_agent_to_advisary = self.env.getPathTo(agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['advisary_rads'] = advisary_rads
            return_dict['advisary_distance'] = len(points_in_path) / self.max_distance

        # the distances need to be normalized

        return return_dict


    def actup_step(self,grid_map, self_value, value_to_objects):
        self.memory.advance(0.1)
        possible_actions = ['noop','down','up','left','right']
        blends = []
        for action in possible_actions:
            probe_chunk = self.gridmap_to_symbols(grid_map, self_value,value_to_objects)
            blend_value = self.memory.blend(action, **probe_chunk)
            blends.append(blend_value)
        return blends

    def init(self):
        self.sess.run(self.init_op)
        if self.mode == ACMode.PPO:
            self.update_theta()

    def _get_select_action_probs(self, pi):
        action_id = select_from_each_row(
            pi.action_id_log_probs, self.placeholders.selected_action_id
        )

        total = action_id

        return SelectedLogProbs(action_id, total)

    def _scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor,
            collections=[tf.GraphKeys.SUMMARIES, self._scalar_summary_key])

    def build_model(self):
        self.placeholders = _get_placeholders(self.spatial_dim, self.nsteps, self.num_envs, self.policy_type, self.obs_dims)
        with tf.variable_scope("theta"):
            self.theta = self.policy(self, trainable=True).build() # (MINE) from policy.py you build the net. Theta is

        selected_log_probs = self._get_select_action_probs(self.theta)

        if self.mode == ACMode.PPO:
            # could also use stop_gradient and forget about the trainable
            with tf.variable_scope("theta_old"):
                theta_old = self.policy(self, trainable=False).build() # theta old is used as a constant here

            new_theta_var = tf.global_variables("theta/")
            old_theta_var = tf.global_variables("theta_old/")

            assert len(tf.trainable_variables("theta/")) == len(new_theta_var)
            assert not tf.trainable_variables("theta_old/") # Has to be empty
            assert len(old_theta_var) == len(new_theta_var)

            self.update_theta_op = [
                tf.assign(t_old, t_new) for t_new, t_old in zip(new_theta_var, old_theta_var)
            ]

            selected_log_probs_old = self._get_select_action_probs(theta_old)
            ratio = tf.exp(selected_log_probs.total - selected_log_probs_old.total)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
            l_clip = tf.minimum(
                ratio * self.placeholders.advantage,
                clipped_ratio * self.placeholders.advantage
            )
            self.sampled_action_id = weighted_random_sample(theta_old.action_id_probs)
            #self.sampled_spatial_action = weighted_random_sample(theta_old.spatial_action_probs)
            self.value_estimate = theta_old.value_estimate
            self._scalar_summary("action/ratio", tf.reduce_mean(clipped_ratio))
            self._scalar_summary("action/ratio_is_clipped",
                tf.reduce_mean(tf.to_float(tf.equal(ratio, clipped_ratio))))
            self.policy_loss = -tf.reduce_mean(l_clip)
        else:
            self.sampled_action_id = weighted_random_sample(self.theta.action_id_probs)
            if self.policy_type == 'FactoredPolicy' or self.policy_type == 'FactoredPolicy_PhaseI' or self.policy_type == 'FactoredPolicy_PhaseII':
                self.value_estimate_goal = self.theta.value_estimate_goal
                self.value_estimate_fire = self.theta.value_estimate_fire
                self.value_estimate = self.theta.value_estimate
            else:
                self.value_estimate = self.theta.value_estimate

        if self.policy_type == 'MetaPolicy':
            # RESHAPE ACTIONS, ADVANTAGES, VALUES. USE MASK TO COMPUTE CORRECT MEANS!!!
            batch_size = tf.shape(self.placeholders.rgb_screen)[0] # or maybe use -1
            max_steps = tf.shape(self.placeholders.rgb_screen)[1]
            mask = self.theta.mask # Check dims!!!
            self.mask = mask # for debug

            # Actions (already masked)
            self.action_id_probs = tf.reshape(self.theta.action_id_probs, [batch_size, max_steps, self.num_actions])
            self.action_id_log_probs = tf.reshape(self.theta.action_id_log_probs, [batch_size,max_steps,self.num_actions])

            # --------------
            # Cross Entropy
            # -------------
            entropy_i = tf.multiply(self.action_id_probs, self.action_id_log_probs) # [batch,max_steps,num_actions]
            # cross_entropy = -tf.reduce_sum(entropy_i, 2) # result: [batch,max_steps], axis=2 means sum wrt actions
            cross_entropy = tf.reduce_sum(entropy_i, 2)
            # mask = tf.sign(tf.reduce_max(tf.abs(entropy_i), 2)) # # [batch,max_steps] with zeros and ones
            cross_entropy *= mask
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, 1) # sum all policy values per timestep for each sequence. result: batch x 1
            cross_entropy /= tf.reduce_sum(mask, 1) # You sum the 1s of the [batch x maxsteps] over axis=1 (maxsteps) to get the actual length of each sequence in your batch
            self.neg_entropy_action_id = tf.reduce_mean(cross_entropy)
            # self.neg_entropy_action_id = tf.reduce_sum(cross_entropy_m) / tf.reduce_sum(tf.reduce_sum(mask, 1))
            # --------------
            #   Policy
            # --------------
            # Start with policy per timestep i per sequence. Result will be [batch * maxsteps]
            policy_i = selected_log_probs.total * self.placeholders.advantage # The selected log probs for masked actions should already be zero. The mask also (inside the policy.py) masks specific lengths so even if there are actions 0 (which are valid as a number) if not included in episode, they will be masked
            # Reshape now to calculate correct means
            policy = tf.reshape(policy_i, [batch_size, max_steps]) # result: [batch x maxsteps]
            policy = tf.reduce_sum(policy, 1) # sum all policy values per timestep for each sequence. result: batch x 1
            policy /= tf.reduce_sum(mask, 1)
            self.policy_loss = -tf.reduce_mean(policy)
            # self.policy_loss = tf.reduce_sum(policy_i) / tf.reduce_sum(tf.reduce_sum(mask, 1))

            # --------------
            #    Value
            # --------------
            vloss_i = tf.squared_difference(self.placeholders.value_target, self.theta.value_estimate)
            mse = tf.reshape(vloss_i, [batch_size, max_steps]) # result: [batch x maxsteps]
            mse = tf.reduce_sum(mse, 1) # sum all value losses per timestep for each sequence. result: batch x 1
            mse /= tf.reduce_sum(mask, 1) # Denominator is the number of timesteps per sequence [batch x 1] vector
            self.value_loss = tf.reduce_mean(mse) # the mean of the mean losses per sequence (so the denominator in mean will be the number of batches)
            # self.value_loss = tf.reduce_sum(vloss_i)/tf.reduce_sum(tf.reduce_sum(mask, 1))# alternative: instead of the mean of the mean per sequence we take the mean of all samples

        elif self.policy_type == 'FactoredPolicy' or self.policy_type == 'FactoredPolicy_PhaseI' or self.policy_type == 'FactoredPolicy_PhaseII':
            self.neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(self.theta.action_id_probs * self.theta.action_id_log_probs, axis=1))
            self.value_loss_goal = tf.losses.mean_squared_error(self.placeholders.value_target_goal, self.theta.value_estimate_goal) # value_target comes from runner/run_batch when you specify the full input
            self.value_loss_fire = tf.losses.mean_squared_error(self.placeholders.value_target_fire,
                                                                self.theta.value_estimate_fire)
            self.value_loss = tf.losses.mean_squared_error(self.placeholders.value_target, self.theta.value_estimate)
            self.policy_loss = -tf.reduce_mean(selected_log_probs.total * self.placeholders.advantage)
        else:
            self.neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(self.theta.action_id_probs * self.theta.action_id_log_probs, axis=1))
            self.value_loss = tf.losses.mean_squared_error(self.placeholders.value_target, self.theta.value_estimate) # value_target comes from runner/run_batch when you specify the full input
            self.policy_loss = -tf.reduce_mean(selected_log_probs.total * self.placeholders.advantage)

        """ Loss function choices """
        if self.policy_type == 'FactoredPolicy':
            loss = (
                    self.policy_loss
                    + (self.value_loss_goal + self.value_loss_fire + self.value_loss) * self.loss_value_weight
                    + self.neg_entropy_action_id * self.entropy_weight_action_id
            )
        elif self.policy_type == 'FactoredPolicy_PhaseI':
            loss = (
                self.policy_loss
                + self.value_loss* self.loss_value_weight
                + self.neg_entropy_action_id * self.entropy_weight_action_id)#\
                # + (self.value_loss_fire + self.value_loss_goal)*0.0 # when it was 0.0000 performance was good--so now might affect more? # You should try take them out of the loss equation completely as with symbolic differentiation might get values
        elif self.policy_type == 'FactoredPolicy_PhaseII':
            loss = (self.value_loss_fire + self.value_loss_goal)#* self.loss_value_weight # Not sure if this is needed
        else:
            loss = (
                self.policy_loss
                + self.value_loss * self.loss_value_weight
                + self.neg_entropy_action_id * self.entropy_weight_action_id
            )

        if self.policy_type == 'FactoredPolicy_PhaseI' or self.policy_type == 'FactoredPolicy_PhaseII':
            # list of the head variables
            head_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"theta/heads")
            tvars = tf.trainable_variables()
            for v in (head_train_vars):
                i = 0
                for tv in tvars:
                    if v.name == tv.name: del tvars[i]
                    i += 1
            # PHASE I
            if self.policy_type == 'FactoredPolicy_PhaseI':
                vars = tvars
            # PHASE II
            elif self.policy_type == 'FactoredPolicy_PhaseII':
                vars = head_train_vars
        else:
            vars = None # default
            tvars = None # added so FullyConv and other policies wont have problem with the extra saver

        self.train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm, # Caps the gradients at the value self.max_gradient_norm
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            variables=vars,
            name="train_op"
        )

        if self.policy_type == 'FactoredPolicy' or self.policy_type == 'FactoredPolicy_PhaseI' or self.policy_type == 'FactoredPolicy_PhaseII':
            self._scalar_summary("value_goal/estimate", tf.reduce_mean(self.value_estimate_goal)) # no correct!mean is for all samples but we use masks!!!
            self._scalar_summary("value_goal/target", tf.reduce_mean(self.placeholders.value_target_goal)) # no correct!mean is for all samples
            self._scalar_summary("value_fire/estimate", tf.reduce_mean(self.value_estimate_fire)) # no correct!mean is for all samples but we use masks!!!
            self._scalar_summary("value_fire/target", tf.reduce_mean(self.placeholders.value_target_fire))
            self._scalar_summary("loss/value_fire", self.value_loss_fire)
            self._scalar_summary("loss/value_goal", self.value_loss_goal)
            self._scalar_summary("value/estimate", tf.reduce_mean(self.value_estimate))
            self._scalar_summary("loss/value", self.value_loss)
        else:
            self._scalar_summary("value/estimate", tf.reduce_mean(self.value_estimate)) # no correct!mean is for all samples but we use masks!!!
            self._scalar_summary("value/target", tf.reduce_mean(self.placeholders.value_target))
            self._scalar_summary("loss/value", self.value_loss)
        # self._scalar_summary("action/is_spatial_action_available",
        #     tf.reduce_mean(self.placeholders.is_spatial_action_available))
        # self._scalar_summary("action/selected_id_log_prob",
        #     tf.reduce_mean(selected_log_probs.action_id)) # You need the corrected one
        self._scalar_summary("loss/policy", self.policy_loss)

        self._scalar_summary("loss/neg_entropy_action_id", self.neg_entropy_action_id)
        self._scalar_summary("loss/total", loss)
        # self._scalar_summary("value/advantage", tf.reduce_mean(self.placeholders.advantage)) # You need the corrected one (masked)
        # self._scalar_summary("action/selected_total_log_prob", # You need the corrected one (masked)
        #     tf.reduce_mean(selected_log_probs.total))

        #tf.summary.image('convs output', tf.reshape(self.theta.map_output,[-1,25,25,64]))

        self.init_op = tf.global_variables_initializer()
        #TODO: we need 2 savers. PhaseI: it will save only the headless network. PhaseII: it will save the whole network (previous params plus the heads params)
        self.saver_orig = tf.train.Saver(max_to_keep=2) # Save everything (tf.all_variables() which is different from tf.trainable_variables()) which includes Adam vars# keeps only the last 2 set of params and model checkpoints. If you want more increase the umber to keep
        # self.saver = tf.train.Saver(max_to_keep=2)
        # This saves and restores only the variables in the var_list
        # self.saver = tf.train.Saver(max_to_keep=2, var_list=tvars)  # 2 phase training
        self.saver = tf.train.Saver(max_to_keep=2, var_list=tvars) # 2 phase training. If tvars=None then saves everything
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(tf.get_collection(self._scalar_summary_key))
        #self.beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        #tf.summary.image('spatial policy', tf.reshape(self.theta.spatial_action_logits, [-1, 32, 32, 1]))

    ''' Necessary for Policy Saliencies'''
        # Below: Get the pi(at|st)
        # logits = self.graph.get_tensor_by_name('theta/action_id/Softmax:0')  # form of tensors <op name>:<out indx>
        # self.neuron_selector = tf.placeholder(tf.int64)
        # pi_at = logits[0][
        #     self.neuron_selector]  # logits is (?,5), logits[0 or 1] is (5,) dims and logits[0][smth] will return 1 of the 5
        # self.pi_at = tf.reshape(pi_at, [1])


    def _input_to_feed_dict(self, input_dict):
        return {k + ":0": v for k, v in input_dict.items()} # Add the :0 after the name of each feature

    def step(self, obs):
        # (MINE) Pass the observations through the net
        feed_dict = self._input_to_feed_dict(obs)

        action_id, value_estimate = self.sess.run(
            [self.sampled_action_id, self.value_estimate],
            feed_dict=feed_dict
        )

        # spatial_action_2d = np.array(
        #     np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        # ).transpose()

        ##### BEHOLDER
        #activations= [np.reshape(images[0], (32, 32, 32)), np.reshape(images[1], (32, 32, 32)), np.reshape(images[2], (32, 32))]
        # spatial_policy_im = np.reshape(images[1], (32, 32))
        #image = np.reshape(convs_im[0], (13,13,64)) # DRONE
        # # Create 3 channel-image
        # spatial_policy_im = np.stack((spatial_policy_im)*3, -1).transpose()
        #activations= [images[0], images[1]]#, spatial_policy_im]
        # self.beholder.update(
        #     session=self.sess,
        #     arrays=image,#activations,# + [first_of_batch] + gradient_arrays
        #     frame=image,
        # )
        ########
        #self.summary_writer.add_summary(images[2]) # seems not working cauz of merging all

        return action_id, value_estimate

    def step_recurrent(self, obs, rnn_state, prev_reward, prev_action):
        # (MINE) Pass the observations through the net
        feed_dict = self._input_to_feed_dict(obs)
        # Net receives a placeholder [batch x maxsteps x dims] so we stack 31 same images with the first. THIS MIGHT NEED FIXING: I think it can be fixed only if maxsteps
        # is None and you fix the batch_size (STILL though in training batch_size > step batch_size=1. In A3C batch_size is the length of the sequence. So no matter the number of steps it will work.
        #TODO: Batch_size is fixed from the flags. Maybe vary (using None) the timestep? NO cauz still you will need batch_size images for one step
        feed_dict['rgb_screen:0'] = np.expand_dims(feed_dict['rgb_screen:0'],axis=1)
        # feed_dict['rgb_screen:0'] = np.tile(feed_dict['rgb_screen:0'], (1, self.nsteps, 1, 1, 1)) # ones declare no change in dimension of the original array
        # # feed_dict['alt_view:0'] = np.expand_dims(feed_dict['alt_view:0'],axis=1)
        # # feed_dict['alt_view:0'] = np.tile(feed_dict['alt_view:0'], (1, 32, 1, 1, 1))
        # #Even if we mask in the policy we try to see if we put blank images
        # feed_dict['rgb_screen:0'][:, 1:,:,:,:]=0

        action_id, value_estimate, state_out = self.sess.run(
            [self.sampled_action_id, self.value_estimate, self.theta.state_out],
            feed_dict={
                self.placeholders.rgb_screen: feed_dict['rgb_screen:0'],
                # self.placeholders.alt_view: feed_dict['alt_view:0'],
                # self.theta.prev_rewards: prev_reward,#np.vstack(prev_rewards),
                # self.theta.prev_actions: prev_action,
                # self.theta.state : rnn_state
                self.theta.mb_dones: [1]*self.num_envs, # list of ones [1,1,...,1]
                self.theta.state_in[0]: rnn_state[0], # when you feed it has to be numpy and not a tensor
                self.theta.state_in[1]: rnn_state[1]
            }
        )

        # action_id = np.reshape(action_id, [self.num_envs, self.nsteps])[:,0]
        # value_estimate = np.reshape(value_estimate, [self.num_envs, self.nsteps])[:, 0]
        return action_id, value_estimate, state_out

    def step_factored(self, obs):
        # (MINE) Pass the observations through the net
        feed_dict = self._input_to_feed_dict(obs)

        action_id, value_estimate_goal, value_estimate_fire, value_estimate = self.sess.run(
            [self.sampled_action_id, self.value_estimate_goal, self.value_estimate_fire, self.theta.value_estimate],
            feed_dict=feed_dict
        )
        return action_id, value_estimate_goal, value_estimate_fire, value_estimate

    def step_eval(self, obs):
        # (MINE) Pass the observations through the net

        # feed_dict = {'rgb_screen:0' : obs['rgb_screen']},
        #              # 'alt_view:0': obs['alt_view']}
        feed_dict = self._input_to_feed_dict(obs) # FireGrid

        action_id, value_estimate, fc, action_probs = self.sess.run(
            [self.sampled_action_id, self.value_estimate, self.theta.fc1, self.theta.action_id_probs],
            feed_dict=feed_dict
        )

        return action_id, value_estimate, fc, action_probs

    def step_eval_factored(self, obs):
        # (MINE) Pass the observations through the net

        # feed_dict = {'rgb_screen:0' : obs['rgb_screen']},
        #              # 'alt_view:0': obs['alt_view']}
        feed_dict = self._input_to_feed_dict(obs)  # FireGrid

        action_id, value_estimate, value_estimate_goal, value_estimate_fire, fc, action_probs = self.sess.run(
            [self.sampled_action_id, self.value_estimate, self.value_estimate_goal, self.value_estimate_fire, self.theta.fc1, self.theta.action_id_probs],
            feed_dict=feed_dict#TODO: ERASE THIS FOR DRONE ENV!!!
        )
        return action_id, value_estimate, value_estimate_goal, value_estimate_fire, fc, action_probs

#TODO: Step Saliency for factored rewards
    # def step_eval_factored_saliency(self, obs):
    #     # (MINE) Pass the observations through the net
    #
    #     # feed_dict = {'rgb_screen:0' : obs['rgb_screen']},
    #     #              # 'alt_view:0': obs['alt_view']}
    #     feed_dict = self._input_to_feed_dict(obs)  # FireGrid
    #
    #     action_id, value_estimate, value_estimate_goal, value_estimate_fire, fc, action_probs = self.sess.run(
    #         [self.sampled_action_id, self.value_estimate, self.value_estimate_goal, self.value_estimate_fire, self.theta.fc1, self.theta.action_id_probs],
    #         feed_dict=feed_dict#TODO: ERASE THIS FOR DRONE ENV!!!
    #     )
    #     ##### UNCOMMENT BELOW
    #     obs_b = np.squeeze(ob.astype(float)) # remove the 1 batch dimension by squeezing
    #     # Baseline is a black image (for integrated gradients)
    #     baseline = np.zeros(obs_b.shape)
    #     baseline.fill(-1)
    #     images = self.placeholders.rgb_screen # Inputs placeholder to differentiate with respect to it.
    #     # ============ VALUE GRADIENT ======================
    #     values = self.graph.get_tensor_by_name('theta/Squeeze:0')
    #     V = tf.reshape(values, [1])
    #     # Vanilla
    #     # Allocentric #############
    #     gradient_saliency = saliency.GradientSaliency(self.graph, self.sess, V, images)
    #     # Below you have to put the other image as input in order to compute deriv w.r.t. the other one
    #     smoothgrad_V = gradient_saliency.GetSmoothedMask(obs_b, feed_dict={self.value_estimate: value_estimate, 'alt_view:0': obsb})
    #     # # Integrated
    #     # # gradient_saliency = saliency.IntegratedGradients(self.graph, self.sess, V, images)
    #     # # smoothgrad_V = gradient_saliency.GetSmoothedMask(obs_b, feed_dict={self.value_estimate: value_estimate}, x_steps=25, x_baseline=baseline)
    #     smoothgrad_V_gray_allo = saliency.VisualizeImageGrayscale(smoothgrad_V)
    #     # # Instead of smoothgrad_V_gray use RGB
    #     # smoothgrad_V_gray = (smoothgrad_V - smoothgrad_V.min()) / (
    #     #         smoothgrad_V.max() - smoothgrad_V.min())
    #     #
    #     mask_allo = copy.deepcopy(smoothgrad_V_gray_allo)
    #     mask_allo[mask_allo<0.7] = 0
    #     # Egocentric ############
    #     obs_b = np.squeeze(obsb.astype(float))  # remove the 1 batch dimension by squeezing
    #     # Baseline is a black image (for integrated gradients)
    #     baseline = np.zeros(obs_b.shape)
    #     baseline.fill(-1)
    #     images = self.placeholders.alt_view  # Inputs placeholder to differentiate with respect to it.
    #     # Value
    #     values = self.graph.get_tensor_by_name('theta/Squeeze:0')
    #     V = tf.reshape(values, [1])
    #     # Vanilla
    #     gradient_saliency = saliency.GradientSaliency(self.graph, self.sess, V, images)
    #     smoothgrad_V = gradient_saliency.GetSmoothedMask(obs_b,
    #                                                      feed_dict={self.value_estimate: value_estimate,
    #                                                                 self.placeholders.rgb_screen: ob})
    #     smoothgrad_V_gray_ego = saliency.VisualizeImageGrayscale(smoothgrad_V)
    #     mask_ego = copy.deepcopy(smoothgrad_V_gray_ego)
    #     mask_ego[mask_ego<0.7] = 0
    #
    #     ##### UNCOMMENT ABOVE
    #
    #     # ============ POLICY GRADIENT ======================
    #     # Vanilla
    #     # gradient_act_saliency = saliency.GradientSaliency(self.graph, self.sess, self.pi_at, images)
    #     # smoothgrad_pi = gradient_act_saliency.GetSmoothedMask(obs_b, feed_dict={self.neuron_selector: action_id[0]})
    #     # gradient_act_saliency = saliency.IntegratedGradients(self.graph, self.sess, self.pi_at, images)
    #     # # smoothgrad_pi = gradient_act_saliency.GetSmoothedMask(obs_b, feed_dict={self.neuron_selector: action_id[0]}, x_steps=25, x_baseline=baseline)
    #     # # Integrated
    #     # #smoothgrad_pi_gray = saliency.VisualizeImageGrayscale(smoothgrad_pi)
    #     # # Instead of smoothgrad_V_gray use RGB
    #     # smoothgrad_pi_gray = (smoothgrad_pi - smoothgrad_pi.min()) / (
    #     #         smoothgrad_pi.max() - smoothgrad_pi.min())
    #
    #
    #
    #
    #
    #     return action_id, value_estimate, value_estimate_goal, value_estimate_fire, fc, action_probs

    def train(self, input_dict):
        feed_dict = self._input_to_feed_dict(input_dict)
        ops = [self.train_op] # (MINE) From build model above the train_op contains all the operations for training

        write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )
        write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if write_all_summaries:
            ops.append(self.all_summary_op)
        elif write_scalar_summaries:
            ops.append(self.scalar_summary_op)

        # Debugging: checking vars before and after training
        # tvars = tf.trainable_variables()
        # params = self.sess.run(tvars)
        # For all vars including Adam vars
        # vars = tf.all_variables()
        # vars_all = self.sess.run(vars)

        r = self.sess.run(ops, feed_dict)  # (MINE) TRAIN!!!

        # tvars_ = tf.trainable_variables()
        # params_ = self.sess.run(tvars_)
        # # For all vars including Adam vars
        # vars_ = tf.all_variables()
        # vars_all_ = self.sess.run(vars_)

        # Compare params after training for change: 0-9, 10-13 heads
        # print('Value goal head weights are unchanged: ', np.array_equal(params[10], params_[10])) # or equivalent and maybe faster (params[0]==params_[0]).all()
        # print('Value fire head weights are unchanged: ', np.array_equal(params[11], params_[11]))
        # print('---------------------------------------------------------')
        # print('Total Value head weights are unchanged: ', np.array_equal(params[8], params_[8]))
        # print('Policy head weights are unchanged: ', np.array_equal(params[6], params_[6]))
        # print('conv1 weights are unchanged: ', np.array_equal(params[0], params_[0]))
        # print('fc1 weights are unchanged: ', np.array_equal(params[4], params_[4]))

        if write_all_summaries or write_scalar_summaries:
            self.summary_writer.add_summary(r[-1], global_step=self.train_step)

        self.train_step += 1 # Should be equal to the "batches" variable (name is used wrong by pekaalto, should be updates)

    def train_recurrent(self, input_dict, mb_l, prev_reward, prev_action): # The input dictionary is designed in the runner with advantage function and other stuff in order to be used in the training.
        feed_dict = self._input_to_feed_dict(input_dict)
        feed_dict['rgb_screen:0'] = np.expand_dims(feed_dict['rgb_screen:0'],axis=1)
        feed_dict['rgb_screen:0'] = np.reshape(feed_dict['rgb_screen:0'], [self.num_envs, self.nsteps, 100, 100, 3]) #TODO: BETTER TO BRING THEM IN READY AS WE DONT KNOW IF ORDER IS PRESERVED WHEN HE COMBINES DIMS in runner
        # feed_dict['alt_view:0'] = np.expand_dims(feed_dict['alt_view:0'], axis=1)
        # feed_dict['alt_view:0'] = np.reshape(feed_dict['alt_view:0'], [2, 32, 100, 100, 3])

        ops = [self.train_op] # (MINE) From build model above the train_op contains all the operations for training

        write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )
        write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if write_all_summaries:
            ops.append(self.all_summary_op)
        elif write_scalar_summaries:
            ops.append(self.scalar_summary_op)
        # You can either use the rnn_state from the previous batch (you need to save it somehow) or you reset at every batch the initial state of the rnn
        rnn_state = self.theta.state_init
        r = self.sess.run(ops, feed_dict={
            self.placeholders.advantage: feed_dict['advantage:0'],
            self.placeholders.value_target: feed_dict['value_target:0'],
            self.placeholders.selected_action_id: feed_dict['selected_action_id:0'],
            self.placeholders.rgb_screen: feed_dict['rgb_screen:0'],
            # self.placeholders.alt_view: feed_dict['alt_view:0'],
            # self.theta.prev_rewards: prev_reward,# feed_dict['prev_rewards:0'],  # np.vstack(prev_rewards),
            # self.theta.prev_actions: prev_action,#feed_dict['prev_actions:0'],
            # self.theta.step_size: [32],
            self.theta.mb_dones: mb_l,
            self.theta.state_in[0]: rnn_state[0],
            self.theta.state_in[1]: rnn_state[1]
        })  # (MINE) TRAIN!!!

        if write_all_summaries or write_scalar_summaries:
            self.summary_writer.add_summary(r[-1], global_step=self.train_step)

        self.train_step += 1

    def get_value(self, obs):
        feed_dict = self._input_to_feed_dict(obs)
        return self.sess.run(self.value_estimate, feed_dict=feed_dict)

    def get_recurrent_value(self, obs, rnn_state, prev_reward, prev_action):
        feed_dict = self._input_to_feed_dict(obs)
        feed_dict['rgb_screen:0'] = np.expand_dims(feed_dict['rgb_screen:0'],axis=1)
        # feed_dict['rgb_screen:0'] = np.tile(feed_dict['rgb_screen:0'], (1, self.nsteps, 1, 1, 1)) # ones declare no change in dimension of the original array
        # # feed_dict['alt_view:0'] = np.expand_dims(feed_dict['alt_view:0'],axis=1)
        # # feed_dict['alt_view:0'] = np.tile(feed_dict['alt_view:0'], (1, 32, 1, 1, 1))
        # feed_dict['rgb_screen:0'][:, 1:, :, :, :] = 0

        value_estimate =  self.sess.run(self.value_estimate, feed_dict={
                self.placeholders.rgb_screen: feed_dict['rgb_screen:0'],
                # self.placeholders.alt_view: feed_dict['alt_view:0'],
                # self.theta.prev_rewards: prev_reward,#np.vstack(prev_rewards),
                # self.theta.prev_actions: prev_action,
                self.theta.mb_dones: [1]*self.num_envs,
                self.theta.state_in[0]: rnn_state[0], # when you feed it has to be numpy and not a tensor
                self.theta.state_in[1]: rnn_state[1]
            }
             )
        # value_estimate = np.reshape(value_estimate, [self.num_envs, self.nsteps])[:, 0]
        return value_estimate

    def get_factored_value(self, obs):
        feed_dict = self._input_to_feed_dict(obs)
        return self.sess.run([self.value_estimate_goal, self.value_estimate_fire, self.theta.value_estimate],
                             feed_dict=feed_dict)

    def flush_summaries(self):# used in run_agent.py/ _save_if_training
        self.summary_writer.flush()

    def save(self, path, step=None): # used in run_agent.py/ _save_if_training
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("saving model to %s, step %d" % (path, step))
        if self.policy_type == 'FactoredPolicy_PhaseII': # save everything
            self.saver_orig.save(self.sess, path + '/model.ckpt', global_step=step) # Used in 2 phase training. Specifically in 2nd phase in which we save everything
        else:
            self.saver.save(self.sess, path + '/model.ckpt', global_step=step) # save specific variables

    def load(self, path, training):
        ckpt = tf.train.get_checkpoint_state(path)
        if training:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.saver_orig.restore(self.sess, ckpt.model_checkpoint_path) # CHANGE HERE: Training: saver, Testing: saver_orig
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded old model with train_step %d" % self.train_step)
        self.train_step += 1

    def update_theta(self):
        if self.mode == ACMode.PPO:
            self.sess.run(self.update_theta_op)
