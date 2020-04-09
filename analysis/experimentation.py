from lucid.misc.io import show
# import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.io.showing import _image_url, _display_html
from lucid.modelzoo.vision_base import Model
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import scipy.ndimage as nd
import os
print(os.getcwd())
import pickle

pickle_in = open('./maps.lst','rb') # The case is from the 0 index of the df
maps = pickle.load(pickle_in)
pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/value_to_objects','rb')
value_to_objects = pickle.load(pickle_in)

def _gridmap_to_image(current_grid_map):
    colors = {
        'red': (252, 3, 3),
        'green': (3, 252, 78),
        'blue': (3, 15, 252),
        'yellow': (252, 252, 3),
        'pink': (227, 77, 180),
        'purple': (175, 4, 181),
        'orange': (255,162,0),
        'white': (255,255,255),
        'aqua': (0,255,255),
        'black': (0, 0, 0),
        'gray' : (96,96,96)
    }
    dims = current_grid_map.shape
    image = np.zeros((dims[0],dims[1],3), dtype=np.uint8)
    image[:] = [96,96,96]
    #put border walls in
    walls = np.where(current_grid_map == 1.0)
    for x,y in list(zip(walls[0],walls[1])):
        #print((x,y))
        image[x,y,:] = colors[value_to_objects[1]['color']]

    object_values = [1,2,3,4]
    for obj_val in object_values:
        obj = np.where(current_grid_map == obj_val)
        image[obj[0],obj[1],:] = colors[value_to_objects[obj_val]['color']]

    return image

class A2Cnet(Model):
    model_path = 'networkb.pb'
    # image_shape = [1, 10, 10, 3]
    image_value_range = (0., 255.)
    input_name = 'rgb_screen'

map_batch = [_gridmap_to_image(grid_map) for grid_map in maps]

imag = map_batch[22].copy()
# imag = imag.reshape([1,10,10,3])
print(imag.shape)
# imag = imag/255 # DONT DO THIS!!! CAUZ YOU DO IT ALREADY IN THE NETWORK!!!
imag = imag.astype('float32')

# model = models.InceptionV1()
# model.load_graphdef()

model = A2Cnet()
model.load_graphdef()

with tf.Graph().as_default(), tf.Session() as sess:
    t_input = tf.placeholder_with_default(imag, [None, None, 3])#, name='rgb_screen') # A placeholder op that passes through `input` when its output is not fed
    # t_input = tf.placeholder(dtype='float32',shape=[None, None, 3])
    T = render.import_model(model, t_input, t_input)

    # Compute activations
    acts = T('theta/action_id/Softmax').eval()
    vals = T('theta/value/BiasAdd').eval()
    # output = sess.run([vals, acts], feed_dict={imag})

print('values:', vals[0])
print('probs:',np.round(acts[0],3))