import gym
from gym import spaces
import copy
from common.misc_tools import *

#Notes: IMPORTANT! ob = env.unwrapped.state returns the full dictionary no matter the wrapper!
class DiscreteToBoxWrapper(gym.ObservationWrapper): # This one returns a BATCH of OBS but takes in 1!!! This one
    # does one hot encoding
    # Notes: It receives an observation (or a set of obs if you are in multienv mode) that are discrete and literally
    #  its just one number. E.g. for 1 env: obs= 3. Then the env.n gives the number of discrete values of the obs and
    #  by using it we create a vector of 0. Then we put 1 to the index 3 making it a 1-hot encoding. It won't work
    #  for Multidiscrete spaces (e.g. [5,3,4] means that we have 3 features and the 1st one has 5 discrete values,
    #  the 2nd feat 3 and the last one 4)
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, gym.spaces.Discrete), \
        #     "Should only be used to wrap Discrete envs."
        self.n = 6#self.observation_space.n
        dims = self.dims
        self.observation_space = spaces.Dict(
            {"grid": spaces.Box(low=0, high=10, shape=(dims[0], dims[0]), dtype=np.int),
             "img": spaces.Box(low=0, high=255, shape=self.img_obs_shape, dtype=np.uint8),
             "features": spaces.Box(low=1, high=8, shape=[6, ], dtype=np.int)
             })

    def observation(self, obs):
        obs_ = copy.deepcopy(obs['grid'])
        # print('map:',obs,'\n')
        agent_xy = np.array(np.where(obs_ == 3)).ravel()
        goal_xy = np.array(np.where(obs_ == 2)).ravel()
        predator_xy = np.array(np.where(obs_ == 4)).ravel()
        # print('agent:', agent_xy, 'predator:', predator_xy, 'goal:',goal_xy,'\n')
        # Some entities disappear when one steps on the other (last states usually) but these states are not stored
        if agent_xy.size == 0:
            agent_xy = np.array([0,0,])

        if predator_xy.size == 0:
            predator_xy = np.array([0,0,])

        if goal_xy.size == 0:
            goal_xy = np.array([0,0,])
        # NOTES: Get to the goal task
        # out = np.concatenate(np.array([agent_xy, goal_xy]).tolist())#, predator_xy, goal_xy]).tolist())
        # NOTES: Get to the goal and avoid predator task
        out = np.concatenate(np.array([agent_xy, predator_xy, goal_xy]).tolist())
        obs['features'] = out
        # print('out=', out, '\n')
        return obs

class FeaturesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = 10  # self.observation_space.n
        self.observation_space = spaces.Box(low=-1, high=14, shape=[10, ], dtype=np.float32) # NNs need floats and
        # not integers!!


    def observation(self, obs):
        def encode_features(map):

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

            if (agent_xy.size != 0) and (predator_xy.size != 0):
                ag_neighs = agent_neighbors(map, agent_xy)
                direction_goal = pointing(agent_xy, goal_xy)
                direction_predator = pointing(agent_xy, predator_xy)
                dist2goal = dist_agent_goal(map)
                dist2pred = dist_agent_pred(map)
                dists = np.append(dist2goal, dist2pred)
                observation = np.concatenate([ag_neighs, direction_goal, direction_predator, dists])
            else: # when done we need to return smth and also the finders should be avoided as wont work (entities
                # are not on the map anymore)
                observation = np.zeros(10)

            return observation

        def dist_agent_pred(map):
            ''' DISTANCE AGENT-PREDATOR IN MOVES'''
            pathArray = map
            agent = np.where(pathArray == 3)
            predator = np.where(pathArray == 4)
            path_to_pred = getPathTo(pathArray, agent, predator, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            dist = len(points_in_path)
            return dist

        def dist_agent_goal(map):
            ''' DISTANCE GOAL-PREDATOR IN MOVES'''
            pathArray = map
            agent = np.where(pathArray == 3)
            goal = np.where(pathArray == 2)
            path_to_pred = getPathTo(pathArray, agent, goal, free_spaces=[0])
            points_in_path = np.where(path_to_pred == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            dist = len(points_in_path)
            return dist

        map = copy.deepcopy(obs['grid'])
        new_obs = encode_features(map)
        return new_obs

class ImageOnlyWrapper(gym.ObservationWrapper):
    # Returns obs as images. It makes env functioning like its obs are images only!
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, gym.spaces.Discrete), \
        #     "Should only be used to wrap Discrete envs."
        self.n = 6#self.observation_space.n
        dims = self.dims
        self.observation_space = spaces.Box(low=0, high=255, shape=self.img_obs_shape, dtype=np.uint8)


    def observation(self, obs):
        obs_ = copy.deepcopy(obs['img'])
        return obs_

class CoordsOnlyWrapper(gym.ObservationWrapper): # This one returns a BATCH of OBS but takes in 1!!! This one
    # does one hot encoding
    # Notes: It receives an observation (or a set of obs if you are in multienv mode) that are discrete and literally
    #  its just one number. E.g. for 1 env: obs= 3. Then the env.n gives the number of discrete values of the obs and
    #  by using it we create a vector of 0. Then we put 1 to the index 3 making it a 1-hot encoding. It won't work
    #  for Multidiscrete spaces (e.g. [5,3,4] means that we have 3 features and the 1st one has 5 discrete values,
    #  the 2nd feat 3 and the last one 4)
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, gym.spaces.Discrete), \
        #     "Should only be used to wrap Discrete envs."
        self.n = 6#self.observation_space.n
        dims = self.dims
        self.observation_space = spaces.Box(low=1, high=8, shape=[6, ], dtype=np.float32)

    def observation(self, obs):
        obs_ = copy.deepcopy(obs['grid'])
        # print('map:',obs,'\n')
        agent_xy = np.array(np.where(obs_ == 3)).ravel()
        goal_xy = np.array(np.where(obs_ == 2)).ravel()
        predator_xy = np.array(np.where(obs_ == 4)).ravel()
        # print('agent:', agent_xy, 'predator:', predator_xy, 'goal:',goal_xy,'\n')
        # Some entities disappear when one steps on the other (last states usually) but these states are not stored
        if agent_xy.size == 0:
            agent_xy = np.array([0,0,])

        if predator_xy.size == 0:
            predator_xy = np.array([0,0,])

        if goal_xy.size == 0:
            goal_xy = np.array([0,0,])
        # NOTES: Get to the goal task
        # out = np.concatenate(np.array([agent_xy, goal_xy]).tolist())#, predator_xy, goal_xy]).tolist())
        # NOTES: Get to the goal and avoid predator task
        out = np.concatenate(np.array([agent_xy, predator_xy, goal_xy]).tolist())
        # print('out=', out, '\n')
        return out