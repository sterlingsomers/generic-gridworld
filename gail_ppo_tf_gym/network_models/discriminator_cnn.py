import tensorflow as tf
from tensorflow.contrib import layers
from gym.spaces import Dict

class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        if isinstance(env.observation_space,Dict):
            ob_space = env.observation_space['img']
        else: ob_space = env.observation_space

        self.num_actions = env.action_space.n
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name

            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape))
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            # add noise for stabilise training # TODO this is only for continuous actions!!!
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1) #TODO: dont concat (comment out)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape))
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n) #TODO: One hot is for hard labeling
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1) #TODO: dont concat (comment out)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(obs=self.expert_s, acts=expert_a_one_hot) #TODO: use input1 and input2 or
                # obs and acts
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(obs=self.agent_s, acts=agent_a_one_hot) #TODO: use input1 and input2 or
                # obs and acts

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=8,
            kernel_size=4, #4, orig:8
            stride=2,#2, orig:4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            # trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=16,
            kernel_size=3, #3, orig:4
            stride=1,#,2
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            # trainable=self.trainable
        )
        # conv3 = layers.conv2d(
        #     inputs=conv2,
        #     data_format="NHWC",
        #     num_outputs=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding='SAME',
        #     activation_fn=tf.nn.relu,
        #     scope="%s/conv3" % name,
        #     # trainable=self.trainable
        # )

        # if self.trainable:
        #     tf.layers.summarize_activation(conv1)
        #     tf.layers.summarize_activation(conv2)
        #     tf.layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def _build_fcs(self ,inputs):
        self.fc1 = layers.fully_connected(
            inputs,
            num_outputs=128,#,512 for pacman
            activation_fn=tf.nn.relu,
            scope="fc1",
            # trainable=self.trainable
        )
        # self.fc2 = layers.fully_connected(
        #     self.fc1,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc2",
        #     # trainable=self.trainable
        # )
        # Add layer normalization for better stability
        # self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable) # VERY BAD FOR THE 2D GRIDWORLD!!!

        # action_id_probs = layers.fully_connected(
        #     self.fc1,
        #     num_outputs=self.num_actions,#len(actions.FUNCTIONS),
        #     activation_fn=tf.nn.softmax,
        #     scope="action_id",
        #     trainable=self.trainable
        # )
        return self.fc1

    def construct_network(self, obs, acts):
        obs_in = tf.cast(obs, tf.float32) / 255.
        cnn_layer = self._build_convs(obs_in, 'cnn_layer')
        flat_layer = layers.flatten(cnn_layer)
        # acts = tf.expand_dims(acts,1)
        # acts = layers.fully_connected(
        #     acts,
        #     num_outputs=64,#NOTES: Output is reward (between 0-1),
        #     activation_fn=tf.nn.tanh,
        #     scope="action_enc",
        #     # trainable=self.trainable
        # )
        h = tf.concat([flat_layer, acts], axis=1) # (s,a) as input creates state transitions
        # layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        # layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        # layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        # prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        h = self._build_fcs(h)
        prob = layers.fully_connected(
            h,
            num_outputs=1,#NOTES: Output is reward (between 0-1),
            activation_fn=tf.sigmoid,
            scope="action_id",
            # trainable=self.trainable
        )

        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})
    # Notes: Outputs below
    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

