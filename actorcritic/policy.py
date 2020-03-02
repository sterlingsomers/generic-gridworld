import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from common.modules import multihead_attention, feedforward, build_Relation, normalize


class FullyConvPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_residual_block(self, inputs, name):
        # https://github.com/wenxinxu/resnet-in-tensorflow/blob/8ba8d8905e099cd7e1b1cf1f84b89f603f7613a0/resnet.py#L56

        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            # orig: 32 (you either leave it 64 or you need to pad the channel dimension when you do the skip/residual connection)
            kernel_size=8,  # 8, orig:4
            stride=4,  # orig:2
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        print('CONV1', conv1.get_shape().as_list())
        ''' Residual blcok'''
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=3,  # 4, orig:3
            stride=1,  # 2,orig:1
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        print('CONV2', conv2.get_shape().as_list())
        conv3 = layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=None,  # tf.nn.relu,
            scope="%s/conv3" % name,
            trainable=self.trainable
        )
        print('CONV3', conv3.get_shape().as_list())

        # We check if the channels between the first layer (that is going to be added to the third) and the third are the same. Else we pad
        # padding will put around the volume zeros
        input_channel = conv1.get_shape().as_list()[-1]  # get the input channels
        output_channel = conv3.get_shape().as_list()[-1]  # get the input channels
        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            # stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            # stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        if increase_dim is True:
            # NO POOLING FOR NOW
            # pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
            #                               strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(conv1, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                   input_channel // 2]])
        else:
            padded_input = conv1

        out = conv3 + padded_input
        out = tf.nn.relu(out)

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            layers.summarize_activation(out)

        # return conv2
        return out

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=4,  # 4, orig:8
            stride=2,  # 2, orig:4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=3,  # 3, orig:4
            stride=1,  # ,2
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        conv3 = layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv3" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            layers.summarize_activation(conv3)

        # return conv2
        return conv3

    def build(self):
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        # self.screen_output = self._build_residual_block(screen_px, "screen_network")
        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.map_output = self.screen_output
        map_output_flat = layers.flatten(self.map_output)

        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=512,  # ,512 for pacman
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        self.fc1 = layers.fully_connected(
            self.fc1,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            scope="fc2",
            trainable=self.trainable
        )
        # Add layer normalization for better stability
        # self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable) # VERY BAD FOR THE 2D GRIDWORLD!!!

        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used. SEEMS THAT YOU NEED THE LOG!!!
        # TODO: Check below for correctness!!!
        # action_id_log_probs = action_id_probs#logclip(action_id_probs) # This one might not be necessary!!!
        action_id_log_probs = logclip(action_id_probs)
        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs

        return self


# End-to-End training with heads
class FactoredPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=8,  # 8
            stride=4,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        # conv3 = layers.conv2d(
        #     inputs=conv2,
        #     data_format="NHWC",
        #     num_outputs=64,
        #     kernel_size=2,
        #     stride=1,
        #     padding='SAME',
        #     activation_fn=tf.nn.relu,
        #     scope="%s/conv3" % name,
        #     trainable=self.trainable
        # )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.map_output = self.screen_output
        map_output_flat = layers.flatten(self.map_output)

        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        # Add layer normalization for better stability
        # self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable) # VERY BAD FOR THE 2D GRIDWORLD!!!--> Not sure

        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        value_estimate_goal = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value_goal',
            trainable=self.trainable  # self.trainable
        ), axis=1
        )
        value_estimate_fire = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value_fire',
            trainable=self.trainable  # self.trainable#
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used. SEEMS THAT YOU NEED THE LOG!!!
        # TODO: Check below for correctness!!!
        # action_id_log_probs = action_id_probs#logclip(action_id_probs) # This one might not be necessary!!!
        action_id_log_probs = logclip(action_id_probs)
        self.value_estimate = value_estimate
        self.value_estimate_fire = value_estimate_fire
        self.value_estimate_goal = value_estimate_goal
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs

        return self


class FactoredPolicy_PhaseI:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=4,  # 8
            stride=2,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=2,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        conv3 = layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=2,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv3" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            layers.summarize_activation(conv3)

        # return conv2
        return conv3

    def build(self):
        # units_embedded = layers.embed_sequence(
        #     self.placeholders.screen_unit_type,
        #     vocab_size=SCREEN_FEATURES.unit_type.scale, # 1850
        #     embed_dim=self.unittype_emb_dim, # 5
        #     scope="unit_type_emb",
        #     trainable=self.trainable
        # )
        #
        # # Let's not one-hot zero which is background
        # player_relative_screen_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_screen,
        #     num_classes=SCREEN_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        # player_relative_minimap_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_minimap,
        #     num_classes=MINIMAP_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        #
        # channel_axis = 2
        # alt0_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker],
        #     axis=channel_axis
        # )
        # alt1_all = tf.concat(
        #     [self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone],
        #     axis=channel_axis
        # )
        # alt2_all = tf.concat(
        #     [self.placeholders.alt2_drone],
        #     axis=channel_axis
        # )
        # alt3_all = tf.concat(
        #     [self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )

        # VOLUMETRIC APPROACH
        # alt_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker,
        #      self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone, self.placeholders.alt2_drone,
        #      self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )
        # self.spatial_action_logits = layers.conv2d(
        #     alt_all,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.map_output = self.screen_output
        map_output_flat = layers.flatten(self.map_output)

        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        self.fc1 = layers.fully_connected(
            self.fc1,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            scope="fc2",
            trainable=self.trainable
        )
        # Add layer normalization for better stability
        # self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable) # VERY BAD FOR THE 2D GRIDWORLD!!!

        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)
        with tf.variable_scope('heads'):
            value_estimate_goal = tf.stop_gradient(tf.squeeze(layers.fully_connected(
                # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
                self.fc1,
                num_outputs=1,
                activation_fn=None,
                scope='value_goal',
                trainable=self.trainable  # self.trainable
            ), axis=1
            ))
            value_estimate_fire = tf.stop_gradient(tf.squeeze(layers.fully_connected(
                # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
                self.fc1,
                num_outputs=1,
                activation_fn=None,
                scope='value_fire',
                trainable=self.trainable  # self.trainable#
            ), axis=1))

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used. SEEMS THAT YOU NEED THE LOG!!!
        # TODO: Check below for correctness!!!
        # action_id_log_probs = action_id_probs#logclip(action_id_probs) # This one might not be necessary!!!
        action_id_log_probs = logclip(action_id_probs)
        self.value_estimate = value_estimate
        self.value_estimate_fire = value_estimate_fire
        self.value_estimate_goal = value_estimate_goal
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs

        return self


class FactoredPolicy_PhaseII:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = tf.stop_gradient(layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=4,  # 8
            stride=2,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        ))
        conv2 = tf.stop_gradient(layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=2,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        ))
        conv3 = tf.stop_gradient(layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=2,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv3" % name,
            trainable=self.trainable
        ))

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            layers.summarize_activation(conv3)

        # return conv2
        return conv3

    def build(self):
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.map_output = self.screen_output
        map_output_flat = layers.flatten(self.map_output)

        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = tf.stop_gradient(layers.fully_connected(
            # I add a stop_gradient here just in case (CHECK the weights if they change with and without the stop_gradient)
            map_output_flat,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        ))
        # Add layer normalization for better stability
        # self.fc1 = tf.stop_gradient(layers.layer_norm(self.fc1,trainable=self.trainable)) # VERY BAD FOR THE 2D GRIDWORLD!!!
        self.fc1 = tf.stop_gradient(layers.fully_connected(
            self.fc1,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            scope="fc2",
            trainable=self.trainable
        ))
        action_id_probs = tf.stop_gradient(layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        ))
        value_estimate = tf.stop_gradient(tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1))
        with tf.variable_scope('heads'):
            value_estimate_goal = tf.squeeze(layers.fully_connected(
                # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
                self.fc1,
                num_outputs=1,
                activation_fn=None,
                scope='value_goal',
                trainable=self.trainable  # self.trainable
            ), axis=1
            )
            value_estimate_fire = tf.squeeze(layers.fully_connected(
                # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
                self.fc1,
                num_outputs=1,
                activation_fn=None,
                scope='value_fire',
                trainable=self.trainable  # self.trainable#
            ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used. SEEMS THAT YOU NEED THE LOG!!!
        # TODO: Check below for correctness!!!
        # action_id_log_probs = action_id_probs#logclip(action_id_probs) # This one might not be necessary!!!
        action_id_log_probs = logclip(action_id_probs)
        self.value_estimate = value_estimate
        self.value_estimate_fire = value_estimate_fire
        self.value_estimate_goal = value_estimate_goal
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs

        return self


# class MetaPolicy:
#     """
#     Meta Policy with recurrency on observations, actions and rewards
#     """
#
#     def __init__(self,
#                  agent,
#                  trainable: bool = True
#                  ):
#         # type agent: ActorCriticAgent
#         self.placeholders = agent.placeholders
#         self.trainable = trainable
#         self.unittype_emb_dim = agent.unit_type_emb_dim
#         self.num_actions = agent.num_actions
#
#     def _build_convs(self, inputs, name):
#         conv1 = layers.conv2d(
#             inputs=inputs,
#             data_format="NHWC",
#             num_outputs=32,
#             kernel_size=8,  # 8
#             stride=4,  # 4
#             padding='SAME',
#             activation_fn=tf.nn.relu,
#             scope="%s/conv1" % name,
#             trainable=self.trainable
#         )
#         conv2 = layers.conv2d(
#             inputs=conv1,
#             data_format="NHWC",
#             num_outputs=64,
#             kernel_size=4,  # 4
#             stride=1,  # 2,#
#             padding='SAME',
#             activation_fn=tf.nn.relu,
#             scope="%s/conv2" % name,
#             trainable=self.trainable
#         )
#         # conv3 = layers.conv2d(
#         #     inputs=conv2,
#         #     data_format="NHWC",
#         #     num_outputs=64,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding='SAME',
#         #     activation_fn=tf.nn.relu,
#         #     scope="%s/conv3" % name,
#         #     trainable=self.trainable
#         # )
#
#         if self.trainable:
#             layers.summarize_activation(conv1)
#             layers.summarize_activation(conv2)
#             # layers.summarize_activation(conv3)
#
#         return conv2
#         # return conv3
#
#     def build(self):
#         screen_px = tf.cast(self.placeholders.rgb_screen,
#                             tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
#         alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
#         self.screen_output = self._build_convs(screen_px, "screen_network")
#         self.alt_output = self._build_convs(alt_px, "alt_network")
#         self.map_output = tf.concat([self.screen_output, self.alt_output], axis=2) # should be 3
#
#         self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32) # num_envs x num_steps (it was [None,1])
#         self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
#         self.prev_actions_onehot = tf.one_hot(self.prev_actions, self.num_actions, dtype=tf.float32)
#         # self.prev_actions_onehot = tf.squeeze(self.prev_actions_onehot,[1])
#         # self.prev_actions_onehot = layers.embed_sequence(
#         #                                 self.prev_actions,
#         #                                 vocab_size=a_size,  # 1850
#         #                                 embed_dim=5,  # 5
#         #                                 scope="unit_type_emb",
#         #                                 trainable=self.training
#         # )
#
#         hidden = tf.concat([layers.flatten(self.map_output), self.prev_actions_onehot, self.prev_rewards], 1)
#         # hidden = layers.flatten(self.map_output)
#         # Below, feed the batch_size!
#         # self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)#.shape(self.placeholders.rgb_screen)
#         lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
#         # lstm_cell.trainable = self.trainable
#
#         # Initialization: you create an initial state which will be fed as self.state in the unfolded net. So you have to define the self.state_init AND the self.state
#         # or maybe have only the self.state defined and assined?
#         # init_vars = lstm_cell.zero_state(2, tf.float32)
#         # init_c = tf.Variable(init_vars.c, trainable=self.trainable)
#         # init_h = tf.Variable(init_vars.h, trainable=self.trainable)
#         # self.state_init = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
#         #
#         # state_vars = lstm_cell.zero_state(2, tf.float32)
#         # state_c = tf.Variable(state_vars.c, trainable=self.trainable)
#         # state_h = tf.Variable(state_vars.h, trainable=self.trainable)
#         # self.state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
#         # self.state = (state_c, state_h)
#
#
#         c_init = np.zeros((2, lstm_cell.state_size.c), np.float32)
#         h_init = np.zeros((2, lstm_cell.state_size.h), np.float32)
#         # # (or bring the batch_size from out) The following should be defined in the runner and you need a self before the lstm_cell. Because you get a numpy array below you need the batch size
#         self.state_init = [c_init, h_init]#lstm_cell.zero_state(2, dtype=tf.float32)# Its already a tensor with a numpy array full of zeros
#         self.c_in = tf.placeholder(tf.float32, [2, lstm_cell.state_size.c])
#         self.h_in = tf.placeholder(tf.float32, [2, lstm_cell.state_size.h])
#         self.state_in = (self.c_in, self.h_in) # You need this so from outside you can feed the two placeholders
#         rnn_in = tf.reshape(hidden,[-1,1,80017])#tf.expand_dims(hidden, [0]) # 1 is the timestep, if you have more you might need -1 also there
#         # step_size = tf.shape(self.prev_rewards)[:1]
#         state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
#         lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
#             lstm_cell, rnn_in, initial_state=state_in, time_major=False) #sequence_length=step_size,
#         # # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
#         # #     lstm_cell, rnn_in, initial_state=self.state_init,time_major=False)
#         lstm_c, lstm_h = lstm_state
#         self.state_out = (lstm_c, lstm_h)#(lstm_c[:1, :], lstm_h[:1, :])
#         # self.state_out = lstm_state
#
#         # layer = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
#         # lstm_outputs, self.new_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=self.state, dtype=tf.float32)
#         #
#         # self.trained_state_c = tf.assign(self.state[0], self.new_state[0])
#         # trained_state_h = tf.assign(self.state[1], self.new_state[1])
#         # self.state_out = tf.contrib.rnn.LSTMStateTuple(self.trained_state_c, trained_state_h) # the new state will be get in the net as self.state
#
#         rnn_out = tf.reshape(lstm_outputs, [-1, 256])
#
#         # Add layer normalization
#         # fc1_ = layers.layer_norm(rnn_out,trainable=self.trainable)
#
#         # map_output_flat = layers.flatten(self.map_output)
#         ''' COMBOS '''
#         #TODO: Omit the fc layer and go straight to the action and value layer:
#         # Just use the rnn_out as input to these layers
#         #TODO: Use the layer normalization:
#         # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
#         fc1 = layers.fully_connected(
#             rnn_out,
#             num_outputs=256,
#             activation_fn=tf.nn.relu,
#             scope="fc1",
#             trainable=self.trainable
#         )
#
#         # Add layer normalization
#         # fc1_ = layers.layer_norm(fc1,trainable=self.trainable)
#
#         # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
#         # estimate
#         action_id_probs = layers.fully_connected(
#             fc1, #rnn_out
#             num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
#             activation_fn=tf.nn.softmax,
#             scope="action_id",
#             trainable=self.trainable
#         )
#         value_estimate = tf.squeeze(layers.fully_connected(
#             # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
#             fc1, #rnn_out
#             num_outputs=1,
#             activation_fn=None,
#             scope='value',
#             trainable=self.trainable
#         ), axis=1)
#
#         def logclip(x):
#             return tf.log(tf.clip_by_value(x, 1e-12, 1.0))
#
#         action_id_log_probs = logclip(action_id_probs)
#
#         self.value_estimate = value_estimate
#         self.action_id_probs = action_id_probs
#         self.action_id_log_probs = action_id_log_probs
#         return self

class MetaPolicy:
    """
    Meta Policy with recurrency on observations, actions and rewards
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions
        self.nenvs = agent.num_envs
        # self.maxsteps = agent.nsteps

    def get_rnn_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=8,  # 8
            stride=4,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2

    def build(self):
        self.mb_dones = tf.placeholder(shape=[None],
                                       dtype=tf.int32)  # size of this is batch_size and each element is the actual sequence length
        # Maybe you need to bring in the inputs as batch x max_steps and not as batch * max_steps. The reshape should have only ONE None dimension so timestep should vary BUT when you test then the env vary
        # or maybe you need even the 1-step prediction to create a batch x self.nsteps x 100 x 100 tensor with 0s
        batch_size = tf.shape(self.placeholders.rgb_screen)[0]
        maxsteps = tf.shape(self.placeholders.rgb_screen)[1]
        rgb_screen = tf.reshape(self.placeholders.rgb_screen, [-1, 100, 100, 3])
        # alt_view = tf.reshape(self.placeholders.alt_view, [-1, 100, 100, 3])
        screen_px = tf.cast(rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(alt_view, tf.float32) / 255.

        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.screen_output = tf.reshape(self.screen_output,
                                        [-1, maxsteps, 25, 25, batch_size * maxsteps])  # This can be a tensor!!!
        # self.alt_output = tf.reshape(self.alt_output, [-1, tf.shape(self.placeholders.alt_view)[1], 25, 25, 64])

        # self.map_output = tf.concat([self.screen_output, self.alt_output], axis=3) # should be 4. with 3 you get both allo and ego in a 25x50image
        self.map_output = self.screen_output
        # hidden = layers.flatten(self.map_output)
        # Reshape hidden as batch x timesteps x dim
        dim = 25 * 25 * 64  # tf.reduce_prod(tf.shape(self.map_output)[2:])# 64 is the num of outputs of the 2nd CNN # tf cannot infer the channel dim (its ? ) but you need the channel dim known in order for tf to work
        hidden = tf.reshape(self.map_output, [batch_size, tf.shape(self.map_output)[1], dim])
        # Below, feed the batch_size!
        # self.batch_size = tf.shape(self.placeholders.rgb_screen[0])
        lstm_cell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)
        # lstm_cell.trainable = self.trainable

        # Initialization: you create an initial state which will be fed as self.state in the unfolded net. So you have to define the self.state_init AND the self.state
        # or maybe have only the self.state defined and assined?
        # init_vars = lstm_cell.zero_state(2, tf.float32)
        # init_c = tf.Variable(init_vars.c, trainable=self.trainable)
        # init_h = tf.Variable(init_vars.h, trainable=self.trainable)
        # self.state_init = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
        #
        # state_vars = lstm_cell.zero_state(2, tf.float32)
        # state_c = tf.Variable(state_vars.c, trainable=self.trainable)
        # state_h = tf.Variable(state_vars.h, trainable=self.trainable)
        # self.state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
        # self.state = (state_c, state_h)

        c_init = np.zeros((self.nenvs, lstm_cell.state_size.c),
                          np.float32)  # [BATCH SIZE x cell_size]. You need to bring the n_envs here. The batch_size above is a TENSOR and not computed till the time that you throw in the data. But this is numpy and u need a value
        h_init = np.zeros((self.nenvs, lstm_cell.state_size.h), np.float32)  # 1 cell for each env
        # # (or bring the batch_size from out) The following should be defined in the runner and you need a self before the lstm_cell. Because you get a numpy array below you need the batch size
        self.state_init = [c_init,
                           h_init]  # lstm_cell.zero_state(2, dtype=tf.float32)# Its already a tensor with a numpy array full of zeros
        self.c_in = tf.placeholder(tf.float32,
                                   [self.nenvs, lstm_cell.state_size.c])  # PLACEHOLDER SHAPE has to be a number!!!
        self.h_in = tf.placeholder(tf.float32, [self.nenvs, lstm_cell.state_size.h])
        self.state_in = (self.c_in, self.h_in)  # You need this so from outside you can feed the two placeholders
        rnn_in = hidden  # tf.reshape(hidden,[-1,1,80017])#tf.expand_dims(hidden, [0]) # 1 is the timestep, if you have more you might need -1 also there
        self.mask = tf.sequence_mask(self.mb_dones, tf.shape(self.placeholders.rgb_screen)[1],
                                     # mask has [batch x maxsteps]#tf.shape(rnn_in[1]), # The tf.shape is for the max_timesteps
                                     dtype=tf.float32)  # by default it will be bool which doesnt work with multiplicaiton
        rnn_in = rnn_in * tf.expand_dims(self.mask,
                                         axis=2)  # expand mask to the 3rd dim of the tensor [batch x masteps x dims] so you cover actual values

        self.seq_length = self.mb_dones  # self.get_rnn_length(rnn_in) # mb_dones = seq_length. Better use the mb_dones rather than the function which finds non zero elements
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=self.seq_length,
            time_major=False)  # sequence_length=step_size,

        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c, lstm_h)  # (lstm_c[:1, :], lstm_h[:1, :])
        # self.state_out = lstm_state

        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # Add layer normalization
        # fc1 = layers.layer_norm(rnn_out,trainable=self.trainable)

        # map_output_flat = layers.flatten(self.map_output)
        ''' COMBOS '''
        # TODO: Omit the fc layer and go straight to the action and value layer:
        # Just use the rnn_out as input to these layers
        # TODO: Use the layer normalization:
        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        fc1 = layers.fully_connected(
            rnn_out,  # fc1
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )

        # Add layer normalization
        # fc1 = layers.layer_norm(fc1,trainable=self.trainable)

        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            fc1,  # rnn_out
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            fc1,  # rnn_out
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        action_id_log_probs = logclip(action_id_probs)  # Apply the mask again below here!!!
        action_id_log_probs = tf.reshape(action_id_log_probs, [batch_size, maxsteps, self.num_actions])
        action_id_log_probs *= tf.expand_dims(self.mask, axis=2)
        action_id_log_probs = tf.reshape(action_id_log_probs, [tf.shape(rgb_screen)[0], self.num_actions])

        action_id_probs = tf.reshape(action_id_probs, [batch_size, maxsteps, self.num_actions])
        action_id_probs *= tf.expand_dims(self.mask, axis=2)
        action_id_probs = tf.reshape(action_id_probs, [tf.shape(rgb_screen)[0],
                                                       self.num_actions])  # back to batch*maxteps(=tf.shape(rgb_screen)[0]) x 16 so we can sample

        value_estimate = tf.reshape(value_estimate, [batch_size, maxsteps])
        value_estimate *= self.mask  # tf.expand_dims(self.mask, axis=0)
        value_estimate = tf.reshape(value_estimate, [tf.shape(rgb_screen)[0]])

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs
        return self


class RelationalPolicy:
    """
    Relational RL from Attention is All you need and Relational RL for SCII and BoxWorld
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions
        self.MHDPA_blocks = 2

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,  # 32,#12,
            kernel_size=8,  # 2
            stride=4,  # 1
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,  # 64, #24
            kernel_size=4,  # 2
            stride=1,  # 1
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        self.alt_output = self._build_convs(alt_px, "alt_network")

        self.cnn_outputs = tf.concat([self.screen_output, self.alt_output],
                                     axis=3)  # if you use 2 then you calculate relations between the ego and the allo

        # self.cnn_outputs = tf.layers.max_pooling2d(
        #     self.cnn_outputs,
        #     3,
        #     2,
        #     padding='valid',
        #     data_format='channels_last',
        #     name='max_pool_for_inputs'
        # ) #for 3,2 then out is 12,12,128
        # with tf.device("/cpu:0"):
        #     self.relation = build_Relation(self.cnn_outputs)

        shape = self.cnn_outputs.get_shape().as_list()
        channels = shape[3]
        dim = shape[1]
        self.relation = tf.reshape(self.cnn_outputs, [-1, shape[1] * shape[2], shape[3]])
        # Stacked MHDPA Blocks with shared weights
        for i in range(self.MHDPA_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                self.relation, self.attention_w = multihead_attention(queries=self.relation,
                                                                      keys=self.relation,
                                                                      num_units=64,
                                                                      # how many dims you want the keys to be, None gives you the dims of ur entity. Should be 528/8=64+2
                                                                      num_heads=2,
                                                                      trainable=self.trainable,
                                                                      channels=channels
                                                                      # dropout_rate=hp.dropout_rate,
                                                                      # is_training=is_training,
                                                                      # causality=False
                                                                      )  # NUM UNITS YOU DONT NEED πολλαπλασια του conv output but πολλαπλασια του num_heads!!! Deepmind use 64* (2-4 heads)

        # self.cnn1d = feedforward(self.MHDPA, num_units=[4 * 66, 66])  # You can use MLP instead of conv1d
        # The max pooling which converts a nxnxk to a k vector
        self.relation = tf.reshape(self.relation, [-1, dim, dim, channels])  # [-1, 13, 13, 66] [-1, 25, 25, 130]
        self.max_pool = tf.layers.max_pooling2d(self.relation, dim, dim)
        map_output_flat = layers.flatten(self.max_pool)
        # map_output_flat = layers.flatten(self.spatial_softmax)

        fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        fc1 = layers.layer_norm(fc1, trainable=self.trainable)
        # fc2 = layers.fully_connected(
        #     fc1,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc2",
        #     trainable=self.trainable
        # )
        # fc3 = layers.fully_connected(
        #     fc2,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc3",
        #     trainable=self.trainable
        # )
        # fc4 = layers.fully_connected(
        #     fc3,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc4",
        #     trainable=self.trainable
        # )

        # Policy
        action_id_probs = layers.fully_connected(
            fc1,
            num_outputs=self.num_actions,  # actions are from 0 to num_actions-1 || len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs
        return self


class FullyConvPolicyAlt:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=8,  # 8
            stride=4,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
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
        #     trainable=self.trainable
        # )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        # units_embedded = layers.embed_sequence(
        #     self.placeholders.screen_unit_type,
        #     vocab_size=SCREEN_FEATURES.unit_type.scale, # 1850
        #     embed_dim=self.unittype_emb_dim, # 5
        #     scope="unit_type_emb",
        #     trainable=self.trainable
        # )
        #
        # # Let's not one-hot zero which is background
        # player_relative_screen_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_screen,
        #     num_classes=SCREEN_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        # player_relative_minimap_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_minimap,
        #     num_classes=MINIMAP_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        #
        # channel_axis = 2
        # alt0_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker],
        #     axis=channel_axis
        # )
        # alt1_all = tf.concat(
        #     [self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone],
        #     axis=channel_axis
        # )
        # alt2_all = tf.concat(
        #     [self.placeholders.alt2_drone],
        #     axis=channel_axis
        # )
        # alt3_all = tf.concat(
        #     [self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )

        # VOLUMETRIC APPROACH
        # alt_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker,
        #      self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone, self.placeholders.alt2_drone,
        #      self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )
        # self.spatial_action_logits = layers.conv2d(
        #     alt_all,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        # alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        # self.alt_output = self._build_convs(alt_px, "alt_network")

        self.alts_ph = self.placeholders.altitudes  # tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.alt = tf.one_hot(self.alts_ph, 4,
                              dtype=tf.float32)  # seems that having input tf.int32 to the mlp in the next line does not work
        # self.altitudes = layers.fully_connected(
        #     self.alt,
        #     num_outputs=28, # = num of units, each unit has one output
        #     activation_fn=tf.nn.relu,
        #     scope="numeric_feats",
        #     trainable=self.trainable
        # )
        # self.altitudes = layers.layer_norm(self.altitudes, trainable=self.trainable)

        # minimap_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255.
        # self.alt0_output = self._build_convs(alt0_all, "alt0_network")
        # self.alt1_output = self._build_convs(alt1_all, "alt1_network")
        # self.alt2_output = self._build_convs(alt2_all, "alt2_network")
        # self.alt3_output = self._build_convs(alt3_all, "alt3_network")

        # VOLUMETRIC APPROACH
        # self.alt0_output = self._build_convs(self.spatial_action_logits, "alt0_network")

        """(MINE) As described in the paper, the state representation is then formed by the concatenation
        of the screen and minimap outputs as well as the broadcast vector stats, along the channer dimension"""
        # State representation (last layer before separation as described in the paper)
        # self.map_output = tf.concat([self.alt0_output, self.alt1_output, self.alt2_output, self.alt3_output], axis=2)
        # self.map_output = tf.concat([self.alt0_output, self.alt1_output], axis=2)

        self.map_output = self.screen_output  # tf.concat([self.screen_output, self.alt_output], axis=2)
        map_output_flat = tf.concat([layers.flatten(self.map_output), self.alt],
                                    1)  # self.altitudes if you use the feedforward

        # self.map_output = self.screen_output
        # The output layer (conv) of the spatial action policy with one ouput. So this means that there is a 1-1 mapping
        # (no filter that convolvues here) between layer and output. So eventually for every position of the layer you get
        # one value. Then you flatten it and you pass it into a softmax to get probs.
        # self.spatial_action_logits = layers.conv2d(
        #     self.map_output,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        #
        # spatial_action_probs = tf.nn.softmax(layers.flatten(self.spatial_action_logits))

        # map_output_flat = layers.flatten(self.map_output)
        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        # Add layer normalization for better stability
        self.fc1 = layers.layer_norm(self.fc1, trainable=self.trainable)

        # fc1 = normalize(fc1, train=False) # wont work cauz PPO compares global variables with trainable variables so no matter True/False assertion will give error
        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # spatial_action_log_probs = (
        #     logclip(spatial_action_probs)
        #     * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        # )

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        # self.spatial_action_probs = spatial_action_probs
        self.action_id_log_probs = action_id_log_probs
        # self.spatial_action_log_probs = spatial_action_log_probs
        return self


class FullyConv3DPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NDHWC",
            num_outputs=32,
            kernel_size=8,  # (5,8,8),  # 8
            stride=4,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NDHWC",
            num_outputs=64,
            kernel_size=4,  # (5,4,4),  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
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
        #     trainable=self.trainable
        # )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        # units_embedded = layers.embed_sequence(
        #     self.placeholders.screen_unit_type,
        #     vocab_size=SCREEN_FEATURES.unit_type.scale, # 1850
        #     embed_dim=self.unittype_emb_dim, # 5
        #     scope="unit_type_emb",
        #     trainable=self.trainable
        # )
        #
        # # Let's not one-hot zero which is background
        # player_relative_screen_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_screen,
        #     num_classes=SCREEN_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        # player_relative_minimap_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_minimap,
        #     num_classes=MINIMAP_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        #
        # channel_axis = 2
        # alt0_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker],
        #     axis=channel_axis
        # )
        # alt1_all = tf.concat(
        #     [self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone],
        #     axis=channel_axis
        # )
        # alt2_all = tf.concat(
        #     [self.placeholders.alt2_drone],
        #     axis=channel_axis
        # )
        # alt3_all = tf.concat(
        #     [self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )

        # VOLUMETRIC APPROACH
        # alt_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker,
        #      self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone, self.placeholders.alt2_drone,
        #      self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )
        # self.spatial_action_logits = layers.conv2d(
        #     alt_all,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        # screen_px = tf.cast(self.placeholders.image_vol,
        #                     tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        screen_px = tf.cast(self.placeholders.joined,
                            tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")

        map_output_flat = layers.flatten(self.screen_output)

        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        # Add layer normalization for better stability
        self.fc1 = layers.layer_norm(self.fc1, trainable=self.trainable)

        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(
            action_id_probs)  # This one might not be necessary!!! RECHECK THIS ONE cauz now the neg entropy in TB seems positive

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs

        return self