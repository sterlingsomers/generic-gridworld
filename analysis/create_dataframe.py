import pandas as pd
from common.misc_tools import *




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

def create_dataframe(obs, filename):
    data = []
    for epis in range(len(obs['player_episode_data'])): # -1 because we include the map
        print('EPISODE:',epis)

        epis_length = obs['player_episode_data'][epis]['actions'].__len__()
        # flag2 = 0 # Indicate that drop agent is in charge (the flag=1 happens only once and then everything else is 0)
        for timestep in range(epis_length):
            print(' ---> timestep:', timestep)
            ''' EPISODES '''
            episode = epis
            ''' TIMESTEPS '''
            tstep = timestep
            # ''' AGENT TYPE (STRING) '''
            # agent_type = 'one_policy'
            ''' ACTIONS (NUMERIC) '''
            actions = obs['player_episode_data'][epis]['actions'][timestep]
            ''' ACTIONS NAME (STRING) '''
            action_label = act_label(actions)
            # ''' ACTIONS PROB DISTR (NUMPY VEC) '''
            # action_dstr = obs['player_episode_data'][epis]['action_probs'][timestep]
            ''' REWARD '''
            reward = round(obs['environment_episode_data'][epis]['reward'][timestep],2)
            ''' DISTANCE AGENT-PREDATOR IN MOVES'''
            pathArray = obs['environment_episode_data'][epis]['observations'][timestep]
            agent = np.where(pathArray == 3)
            predator = np.where(pathArray == 4)
            path_to_pred = getPathTo(pathArray, agent, predator, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            dist = len(points_in_path)
            ''' DISTANCE GOAL-PREDATOR IN MOVES'''
            pathArray = obs['environment_episode_data'][epis]['observations'][timestep]
            goal = np.where(pathArray == 2)
            path_to_pred = getPathTo(pathArray, agent, goal, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            distg = len(points_in_path)
            ''' SALIENCIES'''
            saliencies = np.fromiter(obs['player_episode_data'][epis]['saliences'][timestep].values(), dtype=np.float)

            # dist = min([abs(agent[0][0] - predator[0][0]), abs(agent[1][0] - predator[1][0])])
            # ''' VALUES '''
            # values = round(obs[epis]['values'][timestep],2)
            # ''' DRONE X,Y POSITION (NUMPY) '''
            # drone_pos = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][-2:]
            # ''' DRONE CRASH (BOOLEAN) '''
            # crash = obs[epis]['crash'][timestep]
            # ''' DRONE CRASH EPIS FLAG (BOOLEAN) ''' NO CAUZ WE CARE ABOUT LOCATIONS OF CRASH -- MAYBE DOESNT MATTER WE CARE ONLY IF THE DRONE CRASHED IN A LOCATION, HOWEVER YOU WONT KNOW WHERE IF YOU DONT KMOW THE TIMESTEP HTAT IT CRASHED
            # crash = obs[epis]['crash']
            # ''' FC (VECTOR) '''
            # fc_rep = obs[epis]['fc'][timestep]

            data.append([episode, tstep, actions, action_label, round(reward,3), dist, distg, saliencies])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['episode', 'timestep', 'actions', 'action_label', 'rewards', 'distTopred', 'distTopGoal', 'saliencies']

    #TODO: Optional, load Tensorboard TSNE data and stack them to the dataframe!!!
    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    # Take out unsuccessful trials (get the reward array of an episode, detect if there is a -1 there, convert all rewards of this episode to -1)
    for epis in range(df['episode'].max()):
        r = df['rewards'].loc[(df['episode'] == epis)].values
        if -1 in r:
            df['rewards'].loc[(df['episode'] == epis)] = -1

    df = df.loc[(df['rewards'] != -1)] # remove all entries with -1
    df.to_pickle(path + filename + '.df')
    print('...dataframe saved')

def create_my_dataframe(obs, filename):
    data = []
    for epis in range(obs.__len__()): # -1 because we include the map
        print('EPISODE:',epis)

        epis_length = obs[epis]['actions'].__len__()
        # flag2 = 0 # Indicate that drop agent is in charge (the flag=1 happens only once and then everything else is 0)
        for timestep in range(epis_length):
            print(' ---> timestep:', timestep)
            ''' EPISODES '''
            episode = epis
            ''' TIMESTEPS '''
            tstep = timestep
            # ''' AGENT TYPE (STRING) '''
            # agent_type = 'one_policy'
            ''' ACTIONS (NUMERIC) '''
            actions = obs[epis]['actions'][timestep]
            ''' ACTIONS NAME (STRING) '''
            action_label = act_label(actions)
            # ''' ACTIONS PROB DISTR (NUMPY VEC) '''
            action_dstr = obs[epis]['action_probs'][timestep]
            ''' REWARD '''
            reward = round(obs[epis]['rewards'][timestep],2)
            ''' DISTANCE AGENT-PREDATOR IN MOVES'''
            pathArray = obs[epis]['obs'][timestep]
            agent = np.where(pathArray == 3)
            predator = np.where(pathArray == 4)
            path_to_pred = getPathTo(pathArray, agent, predator, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            dist = len(points_in_path)
            ''' DISTANCE GOAL-PREDATOR IN MOVES'''
            pathArray = obs[epis]['obs'][timestep]
            goal = np.where(pathArray == 2)
            path_to_pred = getPathTo(pathArray, agent, goal, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            distg = len(points_in_path)
            # ''' SALIENCIES'''
            # saliencies = np.fromiter(obs[epis]['saliences'][timestep].values(), dtype=np.float)

            # dist = min([abs(agent[0][0] - predator[0][0]), abs(agent[1][0] - predator[1][0])])
            ''' VALUES '''
            values = round(obs[epis]['values'][timestep],2)
            ''' MAP IMG (NUMPY) (10,10) ''' # for  ego representation graphics
            map = obs[epis]['obs'][timestep]
            # ''' DRONE X,Y POSITION (NUMPY) '''
            # drone_pos = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][-2:]
            # ''' DRONE CRASH (BOOLEAN) '''
            # crash = obs[epis]['crash'][timestep]
            # ''' DRONE CRASH EPIS FLAG (BOOLEAN) ''' NO CAUZ WE CARE ABOUT LOCATIONS OF CRASH -- MAYBE DOESNT MATTER WE CARE ONLY IF THE DRONE CRASHED IN A LOCATION, HOWEVER YOU WONT KNOW WHERE IF YOU DONT KMOW THE TIMESTEP HTAT IT CRASHED
            # crash = obs[epis]['crash']
            ''' FC (VECTOR) '''
            fc_rep = obs[epis]['fc'][timestep]

            data.append([episode, tstep, actions, action_label, round(reward,3), values, dist, distg, action_dstr, map, fc_rep])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['episode', 'timestep', 'actions', 'action_label', 'rewards', 'value_state', 'distTopred', 'distTopGoal', 'policy', 'map', 'fc']

    #TODO: Optional, load Tensorboard TSNE data and stack them to the dataframe!!!
    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    #TODO: You might need unsuccessful trials for tsne!!!
    # Take out unsuccessful trials (get the reward array of an episode, detect if there is a -1 there, convert all rewards of this episode to -1)
    for epis in range(df['episode'].max()):
        r = df['rewards'].loc[(df['episode'] == epis)].values
        if -1 in r:
            df['rewards'].loc[(df['episode'] == epis)] = -1

    df = df.loc[(df['rewards'] != -1)] # remove all entries with -1
    df.to_pickle(path + filename + '.df')
    print('...dataframe saved')

# path = '/Users/constantinos/Documents/Projects/genreal_grid/data/foranalysis/'
# type = '.dict'
# filename = ['sterlng_V2_20200207-142848',                       # Human data
#             'ACTR_sterling_V220200207-164712',                  # ACTR modeling Human
#             'trainedAgent20200206-130551',                      # network data
#             'ACTR_free_play_NETdata_MP20_n0.520200206-180324']  # ACTR modeling network
#
# lista = os.listdir(path)
# for name in filename:
#     pickle_in = open(path + name + type, 'rb')
#     obs = pickle.load(pickle_in)
#     if name + '.df' not in lista:
#         create_dataframe(obs, name)
#
# # df = pd.read_pickle(path + filename + '.df')
# # print('done')
#
# # Load one dataframe and plot
# # import seaborn as sns
# from matplotlib import pyplot as plt
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/'
#                  'foranalysis/sterlng_V2_20200207-142848.df', 'rb')
# obs_human = pickle.load(pickle_in)
# z_human = obs_human.loc[obs_human.action_label=='NOOP', 'distTopred']
# # sns.distplot(z_human, hist=False, kde=True,
# #              kde_kws={'linewidth': 3},
# #              label='Human')
#
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/'
#                  'foranalysis/ACTR_sterling_V220200207-164712.df', 'rb')
# obs_actr_hum = pickle.load(pickle_in)
# z_actr_human = obs_actr_hum.loc[obs_actr_hum.action_label=='NOOP', 'distTopred']
# # sns.distplot(z_actr_human, hist=False, kde=True,
# #              kde_kws={'linewidth': 3},
# #              label='IBL')
# S_actr_human=z_actr_human.value_counts()
# S_human=z_human.value_counts()
# probs_actr_human=S_actr_human/S_actr_human.sum()
# probs_human = S_human/S_human.sum()
# plt.bar([2,3],probs_human, alpha=0.5, label='Human')
# plt.bar([2,5,11,6,4,12,13],probs_actr_human, alpha=0.5, label='IBL')
# plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13])
# plt.legend(loc='upper right')
# ##############################################
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/'
#                  'foranalysis/trainedAgent20200206-130551.df', 'rb')
# obs_net = pickle.load(pickle_in)
# z_net = obs_net.loc[obs_net.action_label=='NOOP', 'distTopred']
# # sns.distplot(z_net, hist=False, kde=True,
# #              kde_kws={'linewidth': 3},
# #              label='Human')
#
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/data/'
#                  'foranalysis/ACTR_free_play_NETdata_MP20_n0.520200206-180324.df', 'rb')
# obs_actr_net = pickle.load(pickle_in)
# z_actr_net = obs_actr_net.loc[obs_actr_net.action_label=='NOOP', 'distTopred']
# # sns.distplot(z_actr_net, hist=False, kde=True,
# #              kde_kws={'linewidth': 3},
# #              label='IBL')
# S_actr_net=z_actr_net.value_counts()
# S_net=z_net.value_counts()
# probs_actr_net=S_actr_net/S_actr_net.sum()
# probs_net = S_net/S_net.sum()
# plt.bar([2],probs_net, alpha=0.5, label='RL agent')
# plt.bar([2,12,11,4,13,9,8,6,5],probs_actr_net, alpha=0.5, label='IBL')
# plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13])
# plt.legend(loc='upper right')

# df = pd.read_pickle(path + filename + '.df')
# data = df['saliencies']
# data = np.array(data.to_list())
# color = df['distTopGoal'].values
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=30)
# X_tsne = tsne.fit_transform(data)
# sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color)
# plt.colorbar(sc)
# plt.show()

# 3D TSNE
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_tsne[:,0],X_tsne[:,1], X_tsne[:,2],c=color,alpha=0.2)
# plt.show()
# Change the name of the file and run this till the end
path = '/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/'
filename = '2020_Apr15_time17-48_net_vs_pred_best_noop'#'2020_Mar24_time22-38_net_vs_pred_best_noop_dark'#'2020_Mar08_time12-46_net_vs_pred_best_noop'
pickle_in = open(path + filename + '.dct', 'rb')
obs = pickle.load(pickle_in)
create_my_dataframe(obs, filename)

# # Below we create an image for every position on the map that the agent can take (USE ONLY COPIES!!!)
# #%%
# # Load Dataframe created above
# df = pd.read_pickle(path + filename + '.df')
# im=df['map'][36].copy()
# im[np.where(im==3)] = 0 # or take out the np.where it works without it
# points = np.where(im==0) # Empty points
#
# maps = []
# for i in range(points[0].size):
#     imagio = im.copy()
#     imagio[points[0][i],points[1][i]] = 3
#     maps.append(imagio)
#
# pickle_in = open('/Users/constantinos/Documents/Projects/genreal_grid/analysis/maps_dark.lst', 'wb')
# pickle.dump(maps, pickle_in)