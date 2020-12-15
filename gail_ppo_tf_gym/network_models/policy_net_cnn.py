import tensorflow as tf
from tensorflow.contrib import layers
from gym.spaces import Dict


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        #Notes: This has separate nets for value and policy. Also it has option for stochastic or deterministic policy.
        # TODO: Check if the space is Dict (isinstance) and select 'img' or if its Box select 'features'
        if isinstance(env.observation_space,Dict):
            ob_space = env.observation_space['img']
        else:
            ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
            obs_in = tf.cast(self.obs, tf.float32) / 255. # We need this!!!
            # self.screen_output = self._build_convs(obs_in, "screen_network")
            # map_output_flat = layers.flatten(self.screen_output)
            #
            # # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
            # self.fc1 = layers.fully_connected(
            #     map_output_flat,
            #     num_outputs=512,  # ,512 for pacman
            #     activation_fn=tf.nn.relu,
            #     scope="fc1",
            #     trainable=True
            # )
            # self.fc1 = layers.fully_connected(
            #     self.fc1,
            #     num_outputs=512,
            #     activation_fn=tf.nn.relu,
            #     scope="fc2",
            #     trainable=True
            # )
            # # Add layer normalization for better stability
            # # self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable) # VERY BAD FOR THE 2D GRIDWORLD!!!
            #
            # self.act_probs = layers.fully_connected(
            #     self.fc1,
            #     num_outputs=act_space.n,  # len(actions.FUNCTIONS),
            #     activation_fn=tf.nn.softmax,
            #     scope="action_soft",
            #     trainable=True
            # )
            # self.v_preds = tf.squeeze(layers.fully_connected(
            #     # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            #     self.fc1,
            #     num_outputs=1,
            #     activation_fn=None,
            #     scope='value',
            #     trainable=True
            # ), axis=1)
            # cnn_layer = self._build_convs(obs_in, 'cnn_layer')
            # flat_layer = layers.flatten(cnn_layer)
            # fcs = self._build_fcs(flat_layer)

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.conv2d(inputs=obs_in, filters=16, kernel_size=(8,8), strides=(2,2), padding='same',
                                           activation=tf.nn.relu)
                layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=(4,4), strides=(1,1), padding='same',
                                           activation=tf.nn.relu)
                layer_flat = tf.layers.flatten(layer_2)
                layer_fc = tf.layers.dense(inputs=layer_flat, units=128, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_fc, units=act_space.n, activation=tf.nn.softmax)
                # layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                # layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                # layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                # self.act_probs = tf.layers.dense(inputs=fcs, units=act_space.n, activation=tf.nn.softmax)
                # cnn_layer = self._build_convs(obs_in, 'cnn_layer')
                # flat_layer = layers.flatten(cnn_layer)
                # fcs = self._build_fcs(flat_layer)
            # self.act_probs = layers.fully_connected(inputs=fcs, num_outputs=act_space.n, activation_fn=tf.nn.softmax)
            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.conv2d(inputs=obs_in, filters=16, kernel_size=(8,8), strides=(2,2), padding='same',
                                           activation=tf.nn.relu)
                layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=(4,4), strides=(1,1),padding='same',
                                           activation=tf.nn.relu)
                layer_flat = tf.layers.flatten(layer_2)
                layer_fc = tf.layers.dense(inputs=layer_flat, units=128, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_fc, units=1, activation=None)
                # layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                # layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                #self.v_preds = tf.layers.dense(inputs=fcs, units=1, activation=None)
                # cnn_layer = self._build_convs(obs_in, 'cnn_layer')
                # flat_layer = layers.flatten(cnn_layer)
                # fcs = self._build_fcs(flat_layer)
            # self.v_preds = layers.fully_connected(inputs=fcs, num_outputs=1, activation_fn=None)

            self.act_stochastic = tf.multinomial(self.act_probs, num_samples=1) # Notes: It was tf.log(self.act_probs)...
            #Notes: multinomial takes log softmax as input so log probs
            # self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.random.categorical(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=8,
            kernel_size=8, #4, orig:8
            stride=4,#2, orig:4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=True
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=16,
            kernel_size=4, #3, orig:4
            stride=2,#,2
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=True
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

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

