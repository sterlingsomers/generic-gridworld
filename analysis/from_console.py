import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn import manifold
import os
os.chdir('/Users/constantinos/Documents/Projects/genreal_grid')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


path = '/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/'
filename = '2020_Mar08_time12-46_net_vs_pred_best_noop'
df = pd.read_pickle(path + filename + '.df')
im = df['map'][1089].copy()
im[np.where(im == 3)] = 0  # or take out the np.where it works without it
points = np.where(im == 0)  # Empty points
maps = []
for i in range(points[0].size):
    imagio = im.copy()
    imagio[points[0][i], points[1][i]] = 3
    maps.append(imagio)
pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/value_to_objects', 'rb')
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/value_to_objects_dark','rb')
value_to_objects = pickle.load(pickle_in)

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

    def test(self,
             data):  # you can add a generic get_tensor_by_name("import/%s:0"%layer) and add layer as an input after data
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

def _gridmap_to_image(current_grid_map):
    colors = {
        'red': (252, 3, 3),
        'green': (3, 252, 78),
        'blue': (3, 15, 252),
        'yellow': (252, 252, 3),
        'pink': (227, 77, 180),
        'purple': (175, 4, 181),
        'orange': (255, 162, 0),
        'white': (255, 255, 255),
        'aqua': (0, 255, 255),
        'black': (0, 0, 0),
        'gray': (96, 96, 96),
        'dark_aqua': (55, 90, 185),
        'dark_orange': (199, 142, 27),
        'dark_green': (67, 105, 18)
    }
    dims = current_grid_map.shape
    image = np.zeros((dims[0], dims[1], 3), dtype=np.uint8)
    image[:] = [96, 96, 96]
    # put border walls in
    walls = np.where(current_grid_map == 1.0)
    for x, y in list(zip(walls[0], walls[1])):
        # print((x,y))
        image[x, y, :] = colors[value_to_objects[1]['color']]
    object_values = [1, 2, 3, 4]
    for obj_val in object_values:
        obj = np.where(current_grid_map == obj_val)
        image[obj[0], obj[1], :] = colors[value_to_objects[obj_val]['color']]
    return image


map_batch_ = [_gridmap_to_image(grid_map) for grid_map in maps]
map_batch_feed_ = np.stack(map_batch_)


m = A2C(model_filepath='./analysis/networkb.pb')
value, policy, fc2, conv1, conv2, conv3, fc2_logit_W, logits_pre_bias, conv1W = m.test(data=map_batch_feed_)
actions = np.argmax(policy,axis=1)
probs = np.round(policy,2)
labels = ['NOOP', 'DOWN', 'UP', 'LEFT','RIGHT']

#Tight Solution Perfect
plt.figure(figsize = (8,8))
# plt.title('Action: NOOP | Value=0.85')
gs1 = gridspec.GridSpec(8, 8)
gs1.update(wspace=0.005, hspace=0.7) # set the spacing between axes.
# gs1.tight_layout
for i in range(62):
    ax1 = plt.subplot(gs1[i], facecolor=(1, 1, 1))
    plt.axis('on')
    ax1.set_title(r"$\bf{" + str(i) + "}$"+ '\n' + labels[actions[i]][0]+ ':' + str(probs[i,actions[i]]) + ', v:' + str(np.round(value[i],2)[0])
                  ,fontsize= 6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(map_batch_feed_[i,...])

plt.show() # COMMENT THIS IF YOU WANT TO SAVE!!!
# plt.savefig('agent_all_positions.pdf',bbox_inches='tight')

# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=8)
# X_tsne = tsne.fit_transform(fc2)
# lbls = ['{0}'.format(i) for i in range(fc2.shape[0])]
# plt.subplots_adjust(bottom = 0.1)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='')
# plt.axis('off')
# for label, x, y in zip(lbls, X_tsne[:, 0], X_tsne[:, 1]):
#     # print(label, x, y)
#     plt.annotate(
#         label,
#         xy=(x, y)
#         #textcoords='offset points') # REMOVE the textcoords as it sends the annotations far away!!!
# plt.show()

# IBL-MFEC predator domain
# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np
# data = pd.read_csv('/Users/constantinos/Documents/Projects/MFEC_02orig/gridworld_05-13-15-08_0p99/results.csv')
# r = data[' episode_nums'].values
# epoch = data['epoch'].values
#
# def smooth(scalars, weight):  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value
#     return smoothed
#
# y = smooth(r, 0.8)
# plt.plot(epoch,r, alpha=0.2, label='orig')
# plt.plot(epoch,y, color= '#000080',label='smoothed')
# plt.legend()
# plt.xlabel('episodes')
# plt.ylabel('reward')
# plt.show()
# data.columns