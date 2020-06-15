# Minimal code in order to run a RL network with a frozen graph with weights.

import tensorflow as tf
import numpy as np
import os
os.getcwd()

class A2C(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Load trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(np.float32, shape=[None, 10, 10, 3], name='rgb_screen')
            # self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
            tf.import_graph_def(graph_def, {'rgb_screen': self.input})

        self.graph.finalize()  # Graph is read-only after this statement

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)

        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph=self.graph)

    def test(self, data):
        # you can add a generic get_tensor_by_name("import/%s:0"%layer) and add layer as an input after
        # data. You can check the way Ludis feature visualization does it.

        # Know your output node name
        value_tensor = self.graph.get_tensor_by_name("import/theta/value/BiasAdd:0")
        policy_tensor = self.graph.get_tensor_by_name("import/theta/action_id/Softmax:0")
        fc2 = self.graph.get_tensor_by_name("import/theta/fc2/Relu:0")
        conv1 = self.graph.get_tensor_by_name("import/theta/screen_network/conv1/Relu:0")
        conv2 = self.graph.get_tensor_by_name("import/theta/screen_network/conv2/Relu:0")
        conv3 = self.graph.get_tensor_by_name("import/theta/screen_network/conv3/Relu:0")
        fc2_logit_W = self.graph.get_tensor_by_name("import/theta/action_id/weights:0")
        logits_pre_bias = self.graph.get_tensor_by_name("import/theta/action_id/MatMul:0")
        conv1W = self.graph.get_tensor_by_name("import/theta/screen_network/conv1/weights:0")
        output = self.sess.run(
            [value_tensor, policy_tensor, fc2, conv1, conv2, conv3, fc2_logit_W, logits_pre_bias, conv1W],
            feed_dict={self.input: data})

        return output
# Specify the path and load the model
m = A2C(model_filepath = 'networkb.pb')
# incoming observation
img = np.random.random([1,10,10,3]) # Should be 0-255 float. You can feed a batch and the results will also going to be batched.
# By getting the policy you can select actions according to np.argmax(policy, axis=1) or with random choice with
# p=policy
value, policy, fc2, conv1, conv2, conv3, fc2_logit_W, logits_pre_bias, conv1W = m.test(data=img)