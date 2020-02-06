import pandas as pd
import pickle
import numpy as np


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


def create_dataframe(obs):
    data = []
    for epis in range(len(obs['player_episode_data'])): # -1 because we include the map
        print('EPISODE:',epis)

        epis_length = obs['player_episode_data'][epis]['steps'].__len__()
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
            actions = obs['player_episode_data'][epis]['steps'][timestep]
            ''' ACTIONS NAME (STRING) '''
            action_label = act_label(actions)
            # ''' ACTIONS PROB DISTR (NUMPY VEC) '''
            # action_dstr = obs['player_episode_data'][epis]['action_probs'][timestep]
            ''' REWARD '''
            reward = round(obs['environment_episode_data'][epis]['reward'][timestep],2)
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

            data.append([episode, tstep, actions, action_label, round(reward,3)])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['episode', 'timestep', 'actions', 'action_label', 'rewards']

    #TODO: Optional, load Tensorboard TSNE data and stack them to the dataframe!!!
    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(path + filename + '.df')
    print('...dataframe saved')

path = '/Users/constantinos/Documents/Projects/genreal_grid/data/net_vs_pred/'
filename = 'test20200205-152242.dict'
pickle_in = open(path + filename, 'rb')
obs = pickle.load(pickle_in)
# if os.path.isfile(path + '.df')==False:
create_dataframe(obs)
df = pd.read_pickle(path + filename + '.df')
print('done')