import pickle
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
# from sklearn import manifold
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
print(os.getcwd())
# os.chdir('..')
# print(os.getcwd())

#%% Create Dataframes
def act_label(action):
    if action == 4:
        action_label = 'RIGHT'
    elif action == 3:
        action_label = 'LEFT'
    elif action == 2:
        action_label = 'UP'
    elif action == 1:
        action_label = 'DOWN'
    elif action == 0:
        action_label = 'NOOP'
    return action_label

def encode_features(map, dist2goal, dist2pred):
    def agent_neighbors(map, agent_xy):
        # params: map is a 2D numpy array
        # params: agent_xy is a numpy array of agent's coords
        left_n = int(map[agent_xy[0], agent_xy[1] - 1] != 0)
        right_n = int(map[agent_xy[0], agent_xy[1] + 1] != 0)
        up_n = int(map[agent_xy[0] - 1, agent_xy[1]] != 0)
        down_n = int(map[agent_xy[0] + 1, agent_xy[1]] != 0)
        out = np.array([left_n, right_n, up_n, down_n])

        return out

    # Which direction the agent should look in order to see the predator/goal?
    def pointing(Origin, Obj):
        # params: Origin is where the agent is
        direction = np.array([0, 0])
        xdist = Origin[0] - Obj[0]
        # print('dx=', xdist)
        ydist = Origin[1] - Obj[1]
        # print('dy=', ydist)

        # we reverse x,y (numpy world) --> y,x (human world)
        if xdist == 0:
            direction[1] = 0
        elif xdist < 0:
            direction[1] = -1  # down
        else:
            direction[1] = 1  # up

        if ydist == 0:
            direction[0] = 0
        elif ydist < 0:
            direction[0] = 1  # right
        else:
            direction[0] = -1  # left

        return direction

    agent_xy = np.array(np.where(map == 3)).ravel()
    # print('Agent:',agent_xy)
    goal_xy = np.array(np.where(map == 2)).ravel()
    # print('Goal:',goal_xy)
    predator_xy = np.array(np.where(map == 4)).ravel()
    # print('Predator:',predator_xy)

    ag_neighs = agent_neighbors(map, agent_xy)
    direction_goal = pointing(agent_xy, goal_xy)
    direction_predator = pointing(agent_xy, predator_xy)

    dists = np.append(dist2goal, dist2pred)
    return np.concatenate([ag_neighs, direction_goal, direction_predator, dists])

h = pickle.load(open('./data/net_vs_pred/human_data_pandas_ready.pdr','rb'))
print('ok')
# h participants (list) --> missions (list) --> dict per mission with steps, actions, features, etc
participant = 1
missions_list = h[participant]
df1 = pd.DataFrame.from_dict(missions_list[0])
symb1 = pd.DataFrame.from_dict(missions_list[0]['symbolic_1'])
df1.drop(['symbolic_1','symbolic_2'], axis=1, inplace=True)
df2=pd.concat([df1,symb1],axis=1)

# get column names and move manually obs to the end in a new list
# b.columns
df =df2[['participant', 'ep_id', 'step', 'action', 'goal_angle',
        'goal_distance', 'advisary_angle', 'advisary_distance',
        'up_wall_distance', 'left_wall_distance', 'down_wall_distance',
        'right_wall_distance', 'obs']]

for mission in range(1, len(missions_list)):  # missions_list for particp 0 = h[0]
    print(mission)
    instance = h[participant][mission]
    d_temp = pd.DataFrame.from_dict(instance)
    symb1_temp = pd.DataFrame.from_dict(instance['symbolic_1'])
    d_temp.drop(['symbolic_1', 'symbolic_2'], axis=1, inplace=True)
    b = pd.concat([d_temp, symb1_temp], axis=1)
    df = pd.concat([df, b], axis=0, sort=True)

# fix column order
df =df[['participant', 'ep_id', 'step', 'action', 'goal_angle',
        'goal_distance', 'advisary_angle', 'advisary_distance',
        'up_wall_distance', 'left_wall_distance', 'down_wall_distance',
        'right_wall_distance', 'obs']]

# reset index
df = df.reset_index(drop=True)
df['features'] = df.apply(lambda row: encode_features(row['obs'], row['goal_distance'], row['advisary_distance']),
                          axis=1)
df['action_label'] = df.apply(lambda row: act_label(row['action']),
                          axis=1)
filename = './data/net_vs_pred/' + str(participant) + '_participant.pdr'
df.to_pickle(filename)
print('ok')

#%% Read File
participant = 1
filename = './data/net_vs_pred/' + str(participant) + '_participant.pdr'
df = pd.read_pickle(filename)
#%% Data Selection
data = df['features'].values
data = np.vstack(data)
print(data.shape)
#%% Data Selection (Sterling)
df1 = df[df['ep_id']>=33]
data = df1[['goal_angle',
        'goal_distance', 'advisary_angle', 'advisary_distance',
        'up_wall_distance', 'left_wall_distance', 'down_wall_distance',
        'right_wall_distance']].values
#%% Data Selection
x_fc = data
y_fc = df['action_label'].values
#%% Analysis
x_train_fc, x_test_fc, y_train_fc, y_test_fc = train_test_split(x_fc, y_fc, test_size=0.20, random_state=42)
#%%
min_max_scaler = preprocessing.MinMaxScaler()
x_train_fc = min_max_scaler.fit_transform(x_train_fc)
x_test_fc = min_max_scaler.transform(x_test_fc)
#%%
def softmax(weights): # Should be applied on a vector and NOT a matrix!
    """Compute softmax values for each sets of matching scores in x."""
    # weights = 5*weights
    e_x = np.exp(-weights)
    s = e_x / e_x.sum(1).reshape(weights.shape[0],1)
    return s
#%%
if   participant==0: k=30  # 0.65
elif participant==1: k=100 # 0.76

clf_neigh = KNeighborsClassifier(n_neighbors=k, weights=softmax)
clf_neigh.fit(x_train_fc, y_train_fc)
y_pred_fc = clf_neigh.predict(x_test_fc)
print('kNN Accuracy: %.4f' % clf_neigh.score(x_test_fc, y_test_fc))
#%% Compare to random
from sklearn.metrics import accuracy_score
y_random = np.random.randint(0,5,y_test_fc.shape[0]) # int [low,high)
accuracy_score(y_test_fc, y_random, normalize=True)

# %% Confusion Matrix

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
classifier = clf_neigh#clf_fc
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, x_test_fc, y_test_fc,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
#%%
# clf_neigh.classes_.tolist()
# out: ['DOWN', 'LEFT', 'NOOP', 'RIGHT', 'UP'] # Notes: This is just alphabetical order

##%% Policy from classifier (input can be a batch of features)

x_heat = min_max_scaler.transform(np.vstack(dd['features'].values))
probs = clf_neigh.predict_proba(x_heat)
print(probs.shape)