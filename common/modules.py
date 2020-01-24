# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


def normalize(inputs,
              epsilon=1e-8,
              scope="ln", # layer normalization
              reuse=None,
              train = False):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), trainable=train)
        gamma = tf.Variable(tf.ones(params_shape), trainable=train)
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


# def embedding(inputs,
#               vocab_size,
#               num_units,
#               zero_pad=True,
#               scale=True,
#               scope="embedding",
#               reuse=None):
#     '''Embeds a given tensor.
#     Args:
#       inputs: A `Tensor` with type `int32` or `int64` containing the ids
#          to be looked up in `lookup table`.
#       vocab_size: An int. Vocabulary size.
#       num_units: An int. Number of embedding hidden units.
#       zero_pad: A boolean. If True, all the values of the fist row (id 0)
#         should be constant zeros.
#       scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
#       scope: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#     Returns:
#       A `Tensor` with one more rank than inputs's. The last dimensionality
#         should be `num_units`.
#
#     For example,
#
#     ```
#     import tensorflow as tf
#
#     inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
#     outputs = embedding(inputs, 6, 2, zero_pad=True)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print sess.run(outputs)
#     >>
#     [[[ 0.          0.        ]
#       [ 0.09754146  0.67385566]
#       [ 0.37864095 -0.35689294]]
#      [[-1.01329422 -1.09939694]
#       [ 0.7521342   0.38203377]
#       [-0.04973143 -0.06210355]]]
#     ```
#
#     ```
#     import tensorflow as tf
#
#     inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
#     outputs = embedding(inputs, 6, 2, zero_pad=False)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print sess.run(outputs)
#     >>
#     [[[-0.19172323 -0.39159766]
#       [-0.43212751 -0.66207761]
#       [ 1.03452027 -0.26704335]]
#      [[-0.11634696 -0.35983452]
#       [ 0.50208133  0.53509563]
#       [ 1.22204471 -0.96587461]]]
#     ```
#     '''
#     with tf.variable_scope(scope, reuse=reuse):
#         lookup_table = tf.get_variable('lookup_table',
#                                        dtype=tf.float32,
#                                        shape=[vocab_size, num_units],
#                                        initializer=tf.contrib.layers.xavier_initializer())
#         if zero_pad:
#             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
#                                       lookup_table[1:, :]), 0)
#         outputs = tf.nn.embedding_lookup(lookup_table, inputs)
#
#         if scale:
#             outputs = outputs * (num_units ** 0.5)
#
#     return outputs


# def positional_encoding(inputs,
#                         num_units,
#                         zero_pad=True,
#                         scale=True,
#                         scope="positional_encoding",
#                         reuse=None):
#     '''Sinusoidal Positional_Encoding.
#     Args:
#       inputs: A 2d Tensor with shape of (N, T).
#       num_units: Output dimensionality
#       zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
#       scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
#       scope: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#     Returns:
#         A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
#     '''
#
#     N, T = inputs.get_shape().as_list()
#     with tf.variable_scope(scope, reuse=reuse):
#         position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
#
#         # First part of the PE function: sin and cos argument
#         position_enc = np.array([
#             [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
#             for pos in range(T)])
#
#         # Second part, apply the cosine to even columns and sin to odds.
#         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
#         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
#
#         # Convert to a tensor
#         lookup_table = tf.convert_to_tensor(position_enc)
#
#         if zero_pad:
#             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
#                                       lookup_table[1:, :]), 0)
#         outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
#
#         if scale:
#             outputs = outputs * num_units ** 0.5
#
#         return outputs

def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        channels=None,
                        # dropout_rate=0,
                        # is_training=True,
                        # causality=False,
                        trainable=None,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]. From hyperparameter file in github N: batches, T=max num of words in a sentence, C= hidden_units
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, trainable=trainable)  # (N, T_q, C) # trainable wasnt existing anywhere we add it now
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, trainable=trainable)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, trainable=trainable)  # (N, T_k, C)

        Q = layers.layer_norm(Q, trainable=trainable)
        K = layers.layer_norm(K, trainable=trainable)
        V = layers.layer_norm(V, trainable=trainable)


        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h) heads * batches
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Should we normalize all heads separately in a loop?
        # Q = layers.layer_norm(Q, trainable=trainable)
        # K = layers.layer_norm(K, trainable=trainable)
        # V = layers.layer_norm(V, trainable=trainable)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k) # multiply the Q which is num_entities x dq (e.g. 66) if last conv has 64 outputs

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Till here is the same almost as the neuralmonkey code

        # # Key Masking
        # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        # key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        # #
        # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        #
        # # Causality = Future blinding
        # if causality:
        #     diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        #     tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
        #     masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
        #
        #     paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        #     outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        attention_w = outputs
        # # Query Masking
        # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        # query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # outputs *= query_masks  # broadcasting. (N, T_q, C)
        #
        # # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape. (1) the original sequence of actions. RESIDUAL LINKS MIGHT BE DIFFERENT THOUGH
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C) # C=528=66*8

        # outputs = feedforward(outputs, num_units=[4 * num_units, num_units]) # (MINE)
        outputs = feedforward(outputs, num_units=[256, channels], trainable=trainable) # contains residual + normalization, 4* is as a rules of thumbs for first layer to be 4 times inputs size
        # Residual connection (2) # Here or after the feeforward?
        outputs += queries

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)
        outputs = layers.layer_norm(outputs, trainable=trainable)

    return outputs, attention_w


def feedforward(inputs,
                num_units=[2048, 528],
                scope="multihead_attention",
                trainable=None,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True, "trainable": trainable} # and here trainable is added (and below)
        outputs = tf.layers.conv1d(**params) # 1-D convolution as written in Reddit, it can be substituted with MLP

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True, "trainable": trainable} # put relu here, it was None
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        # outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels


    return ((1 - epsilon) * inputs) + (epsilon / K)


def build_Relation(conv):
    """
    Build the relation block for each pair of objects
    Here we use it for constructing matrix E (so only one object)
    """
    # E = []
    # with tf.device("/cpu:0"):
    relations = []
    w = conv.get_shape().as_list()[1]
    h = conv.get_shape().as_list()[2]
    # for each combination of pixel in the final feature map, create a relation pair.
    for i in range(w * h): # πχ για grid 12x12: for object i in range 144. This loop seems to be the memory problem when backpropagating
        o_i = conv[:, int(i / w), int(i % w), :]  # Take the object from the feature map (first dim is batch and last one is channels) dim =channelsize
        o_i = tag_obj(o_i, i, w)
        # o_i = tf.reshape(o_i,[-1,1,o_i.get_shape().as_list()[1]]) # [None,1,64+2]
        # for j in range(w * h):
        #     o_j = self.conv4[:, int(j / w), int(j % w), :]
        #     # tag the object pair with coordinate
        #     o_j = tag_obj(o_j, j, w)
        #     if i == 0 and j == 0:
        #         relation = g_theta(o_i, o_j, self.question_embed, reuse=False)
        #     else:
        #         relation = g_theta(o_i, o_j, self.question_embed, reuse=True)
        relations.append(o_i)
    E = tf.stack(relations, axis=0)  # axis=0 means κάθετα # [1024,?,66]
    E = tf.transpose(E, [1, 0, 2])
        # sum over the output from g_theta
        # self.relations = tf.reduce_sum(relations, axis=0, name='relation')
    return E


def tag_obj(o, i, d):
    """
    tag the object with normalized coordinate
    """
    # with tf.device("/cpu:0"):
    coor = tf.tile(tf.expand_dims(
        [float(int(i / d)) / d * 2 - 1, (i % d) / d * 2 - 1], axis=0
    ),
        # [o.get_shape().as_list()[0], 1]
        [tf.shape(o)[0], 1]
    )  # this is from -1 to 1
    o = tf.concat([o, tf.to_float(coor)], axis=1)  # out is a vector dim = [?,66] which is 64 + 2 from coordinates

    return o
