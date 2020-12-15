import gym
import sys
import os
import copy
from gym import spaces
from gym.utils import seeding
import random
import itertools
import functools
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from common.maps import *
import numpy as np
from collections import OrderedDict
import PIL
from envs.core_v2 import *
import time

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

def step_wrapper(f):
    def _record_step(*args, **kwargs):
        obs_before_step = args[0].current_grid_map.copy()
        grid, reward, done, info = f(*args, **kwargs)
        if args[0].record_history:
            args[0].history['observations'].append(obs_before_step)
            args[0].history['reward'].append(reward)
            args[0].history['done'].append(done)

        return grid, reward, done, info
    return _record_step


# def record_action(f):
#     def _record_action(*args, **kwargs):
#         action = f(*args, **kwargs)
#         if args[0].record_history:
#             args[0].history['steps'].append(action)
#         return f(*args, **kwargs)
#     return _record_action


class GenericEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    value_to_objects = {1: {'class': 'wall', 'color': 'black', 'moveTo': 0}}
    object_values = [1]
    entities = {} #indexed by object value
    active_entities = {}
    backup_values = {}
    actions = {}
    reward = 0
    done = 0
    to_clean = []
    permanents = []
    free_spaces = [0]
    info = {}
    info['fire'] = 0
    info['goal'] = 0

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
        'gray' : (96,96,96),
        'dark_aqua': (55, 90, 185),
        'dark_orange': (199, 142, 27),
        'dark_green': (67, 105, 18)
    }

    def __init__(self,dims=(10,10),map='',agents=[],features=[],border=True, entities = [], wallrule=False):

        self.map = map
        self.features = features
        self.dims = dims
        self.record_history = False
        self.history = {}
        self.wallrule=wallrule
        self.border = border

        #before anything happens, setup the map
        self.setupMap(map,dims)

        #add (dynamic, random) features to the map
        self.addMapFeatures(features)

        # #add other entities to the environment
        for entity in entities: # WHY THIS? ENTITIES ARE EMPTY HERE
            class_to_use = getattr(sys.modules[__name__], entity['class'])
            class_to_use(self,entity_type=entity['entity_type'],color=entity['color'],position=entity['position'])


        self.base_grid_map = np.copy(self.current_grid_map)

        self.action_map = {1:lambda x: ((x[0]+1)%self.dims[0],(x[1])%self.dims[1]),
                           2:lambda x: ((x[0]-1)%self.dims[0],(x[1])%self.dims[1]),
                           3:lambda x: (x[0]%self.dims[0],(x[1]-1)%self.dims[1]),
                           4:lambda x: (x[0]%self.dims[0],(x[1]+1)%self.dims[1]),
                           0:lambda x: (x[0],x[1])}

        self.img_obs_shape = [dims[0], dims[0], 3]
        # self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.observation_space = spaces.Dict({"grid": spaces.Box(low=0, high=10, shape=(dims[0], dims[0]), dtype=np.int),
                                              "img": spaces.Box(low=0, high=255, shape=self.img_obs_shape, dtype=np.uint8),
                                              })
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-float('inf'), float('inf')) # or self.reward_range = (0,1)

        #edges
        edges = []
        left = 0
        right = self.current_grid_map.shape[0]
        top = 0
        bottom = self.current_grid_map.shape[1]
        left_edges = [(0, x) for x in range(bottom)]
        right_edges = [(right, x) for x in range(bottom)]
        top_edges = [(x, 0) for x in range(right)]
        bottom_edges = [(bottom, x) for x in range(right)]
        self.edges = left_edges + right_edges + top_edges + bottom_edges

        #Run the dynamic environment
        #self.run()

    def seed(self, seed=None):
        np.random.seed(seed=seed) # It works the seed now!!!

    def setRecordHistory(self,on=True,history_dict={'observations':[],'value_to_objects':0,'reward':[],'done':[]}):
        self.record_history = True
        self.history = history_dict
        self.history['value_to_objects'] = self.value_to_objects

    def getPathTo(self,start_location,end_location, free_spaces=[]):
        '''An A* algorithm to get from one point to another.
        free_spaces is a list of values that can be traversed.
        start_location and end_location are tuple point values.
        Returns a map with path denoted by -1 values. Inteded to use np.where(path == -1).'''
        pathArray = np.full(self.dims,0)
        if start_location[0] == end_location[0] and start_location[1] == end_location[1]:
            pathArray[start_location] = -1
            return pathArray

        for free_space in free_spaces:
            zeros = np.where(self.current_grid_map == free_space)
            zeros = list(zip(zeros[0],zeros[1]))
            for point in zeros:
                pathArray[point] = 1

        #Because we started with true (1), we start with a current value of 1 (which will increase to two)
        current_value = 1
        target_value = 0
        pathArray[start_location] = 2
        directions = [UP, DOWN, LEFT, RIGHT]
        random.shuffle(directions) #TODO
        stop = False
        while True:
            current_value += 1
            target_value = current_value + 1
            test_points = np.where(pathArray == current_value)
            test_points = list(zip(test_points[0],test_points[1]))
            random.shuffle(test_points) #TODO
            still_looking = False
            for test_point in test_points:
                for direction in directions:
                    if pathArray[self.action_map[direction](test_point)] and pathArray[self.action_map[direction](test_point)] + current_value <= target_value:
                        pathArray[self.action_map[direction](test_point)] = target_value
                        still_looking = True
                    # if not end_location[0].tolist():
                    #     print('emtpiness')
                    # print(self.action_map[direction](test_point), end_location)
                    # print(test_point, end_location, direction)
                    # try:
                    # print('direction=', direction)
                    # print('test_point=', test_point)
                    # print('endloc0=', end_location[0])
                    # print('endloc1=', end_location[1])
                    if self.action_map[direction](test_point) == (int(end_location[0]),int(end_location[1])):
                        pathArray[end_location] = - 1
                        still_looking = True
                        stop = True
                        break
                    # except Exception:
                    #     print('ERROR')

            if not still_looking:
                return pathArray
            if stop:
                break
        current_point = end_location
        while True:
            for direction in directions:
                if pathArray[self.action_map[direction](current_point)] == target_value - 1:
                    pathArray[current_point] = -1
                    current_point = self.action_map[direction](current_point)
                    target_value -= 1
                if current_point == start_location:
                    # pathArray[current_point] = -1
                    return pathArray

    def allNeighbors(self,x,y):
        X = self.current_grid_map.shape[0]
        Y = self.current_grid_map.shape[1]
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                                  for y2 in range(y - 1, y + 2)
                                  if (-1 < x <= X and
                                      -1 < y <= Y and
                                      (x != x2 or y != y2) and
                                      (0 <= x2 <= X) and
                                      (0 <= y2 <= Y))]
        return neighbors

    def getGoalValue(self):
        for value in self.value_to_objects:
            if 'entity_type' in self.value_to_objects[value]:
                if self.value_to_objects[value]['entity_type'] == 'goal':
                    return value

    def getNeighbors(self,location):
        up = self.action_map[UP](location)
        if not int(self.current_grid_map[up[0],up[1]]) in self.free_spaces:
            up = False
        down = self.action_map[DOWN](location)
        if not int(self.current_grid_map[down[0],down[1]]) in self.free_spaces:
            down = False
        left = self.action_map[LEFT](location)
        if not int(self.current_grid_map[left[0],left[1]]) in self.free_spaces:
            left = False
        right = self.action_map[RIGHT](location)
        if not int(self.current_grid_map[right[0],right[1]]) in self.free_spaces:
            right = False
        return {'up':up, 'down':down, 'left':left, 'right':right}

    def schedule_cleanup(self,value):
        self.to_clean.append(value)

    def cleanup(self):
        for entity in self.to_clean:
            del self.entities[entity]
        self.to_clean = []

    def addMapFeatures(self,features=[]):
        for feature in features:
            n_features = feature['start_number']

            # free_spaces = np.where(self.current_grid_map == 0)
            # random_free_space_i = random.choice(free_spaces[0])
            # random_free_space_j = random.choice(free_spaces[1])
            # random_free_space = (random_free_space_i,random_free_space_j)
            # self.current_grid_map[random_free_space] = object_value
            #pertubation = np.random.randint(-1, 1, (1,2))
            for i in range(n_features):
                object_value = self.object_values[-1] + 1
                self.object_values.append(object_value)
                self.value_to_objects[object_value] = {'entity_type': feature['entity_type'], 'color': feature['color'],
                                                       'moveTo': feature['moveTo']}
                free_spaces = []
                for free_space in self.free_spaces:
                    found_spaces = np.where(self.current_grid_map == free_space)
                    free_spaces.extend(list(zip(found_spaces[0], found_spaces[1])))
                the_space = random.choice(free_spaces)
                self.current_grid_map[the_space] = object_value
                # free_spaces = np.where(self.current_grid_map == 0)
                # free_spaces = [(x,y) for x,y in zip(free_spaces[0],free_spaces[1]) if x >= random_free_space_i - 1 and x <= random_free_space_i + 1 and
                #                y >= random_free_space_j - 1 and y <= random_free_space_j + 1]
                # free_space = random.choice(free_spaces)
                # self.current_grid_map[free_space] = object_value

    def setupMap(self,map,dims):
        if map == '':
            self.dims = dims
            self.current_grid_map = np.zeros(dims)
            #make border the walls (walls = 1)
            self.current_grid_map[0,:] = [1] * dims[0]
            self.current_grid_map[:,0] = [1] * dims[1]
            self.current_grid_map[:,-1] = [1] * dims[1]
            self.current_grid_map[-1,:] = [1] * dims[0]
        else:
            self.dims = maps[map]['map'].shape
            self.current_grid_map = maps[map]['map']
            self.object_values = np.unique(self.current_grid_map).tolist()
            if 'colors' in maps[map]:
                for value in maps[map]['colors']:
                    if value  in self.value_to_objects:
                        continue
                    self.value_to_objects[value] = {'color':maps[map]['colors'][value]}
            if 'properties' in maps[map]:
                for value in maps[map]['properties']:
                    self.value_to_objects[value].update(maps[map]['properties'][value])
            if 'permanent' in maps[map]:
                for value in maps[map]['permanent']:
                    self.permanents.append(value)
            if 'free-spaces' in maps[map]:
                [self.free_spaces.append(i) for i in maps[map]['free-spaces'] if not i in self.free_spaces]

    def moveToAgent(self,current_position,intended_position):
        return 0

    def moveToDefault(self,current_position,intended_position):
        '''What to do in the event you move to a goal'''

        current_position_value = self.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
        self.backup_values[(int(intended_position[0]), int(intended_position[1]))] = int(intended_position_value)
        self.current_grid_map[current_position] = 0.0
        self.current_grid_map[intended_position] = current_position_value

        return 0

    def moveToGoal(self,current_position,intended_position):
        '''What to do in the event you move to a goal'''
        current_position_value = self.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
        self.current_grid_map[current_position] = 0.0
        self.current_grid_map[intended_position] = current_position_value
        self.done = True
        self.reward += 1
        # print('GOAL REACHED')
        return 0

    def moveToObstacle(self,current_position,intended_position):
        '''What to do in the event you move to an obstacle'''
        return 0

    def obs_process_(self):
        # Find all entities in the grid
        obs = copy.deepcopy(self.current_grid_map)
        # print('map:',obs,'\n')
        agent_xy = np.array(np.where(obs == 3)).ravel()
        goal_xy = np.array(np.where(obs == 2)).ravel()
        predator_xy = np.array(np.where(obs == 4)).ravel()
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

    def obs_process(self):
        # positions + angles + distances
        gridmap = copy.deepcopy(self.current_grid_map)
        # print('map:',obs,'\n')
        # agent_xy = np.array(np.where(obs == 3)).ravel()
        # agent_location = np.where(gridmap == 3)
        # agent_location = (int(agent_location[0]), int(agent_location[1]))
        agent_location = 0
        goal_location = 0 # only if it doesnt exist it becomes 0
        advisary_location = 0
        return_dict = {}
        obs = -1*np.zeros(4, dtype=np.float) # zeros are plausible observations!

        for stuff in self.value_to_objects:
            if 'entity_type' in self.value_to_objects[stuff]:
                if self.value_to_objects[stuff]['entity_type'] == 'goal':
                    goal_location = np.where(gridmap == stuff)
                    goal_xy = np.array(goal_location).ravel()
                    # print('\n','goal',goal_location, '\n')
                if self.value_to_objects[stuff]['entity_type'] == 'advisary':
                    advisary_location = np.where(gridmap == stuff)
                    predator_xy = np.array(advisary_location).ravel()
                    # print('\n','predator=', advisary_location, 'pred_stuff=', stuff, '\n')
                if self.value_to_objects[stuff]['entity_type'] == 'agent':
                    agent_location = np.where(gridmap == stuff)
                    agent_xy = np.array(agent_location).ravel()
                    # print('\n','agent', agent_location, '\n')


        if (goal_xy.size != 0) and (agent_xy.size != 0):
            # print('\n','goal=',goal_xy.size,'agent=',agent_xy.size,'\n')
            goal_location = (int(goal_location[0]), int(goal_location[1]))
            xdist_goal_agent = goal_location[0] - agent_location[0]
            ydist_goal_agent = goal_location[1] - agent_location[1]
            # if goal_location[0] < agent_location[0]:
            #     xdist_goal_agent = - xdist_goal_agent
            # if goal_location[1] < agent_location[1]:
            #     ydist_goal_agent = - ydist_goal_agent
            # atan2(y,x)
            # print(xdist_goal_agent, ydist_goal_agent,'\n')
            goal_rads = math.atan2(ydist_goal_agent, xdist_goal_agent)

            path_agent_to_goal = self.getPathTo(agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))

            if goal_rads < 0:
                goal_rads += 2*math.pi
            return_dict['goal_rads'] = goal_rads/(2*math.pi) # normalize
            return_dict['goal_distance'] = len(points_in_path) / self.max_distance
            obs[0] = return_dict['goal_rads']
            obs[1] = return_dict['goal_distance']

        if (predator_xy.size != 0) and (agent_xy.size != 0):
            # print('\n', 'predator=', predator_xy.size, 'agent=',agent_xy.size, '\n')
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            xdist_advers_agent = advisary_location[0] - agent_location[0]
            ydist_advers_agent = advisary_location[1] - agent_location[1]
            # if advisary_location[0] < agent_location[0]:
            #     xdist_advers_agent = - xdist_advers_agent
            # if advisary_location[1] < agent_location[1]:
            #     ydist_advers_agent = - ydist_advers_agent
            # print(xdist_advers_agent, ydist_advers_agent, '\n')
            advisary_rads = math.atan2(ydist_advers_agent, xdist_advers_agent)
            path_agent_to_advisary = self.getPathTo(agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))

            if advisary_rads < 0:
                advisary_rads += 2*math.pi
            return_dict['advisary_rads'] = advisary_rads/(2*math.pi) # normalize
            return_dict['advisary_distance'] = len(points_in_path) / self.max_distance
            obs[2] = return_dict['advisary_rads']
            obs[3] = return_dict['advisary_distance']

        # the distances need to be normalized


        return obs#return_dict

    def reset(self):
        # for entity in self.entities:
        #     entity_loc = np.where(self.current_grid_map == entity)
        #     for x,y in list(zip(entity_loc[0],entity_loc[1])):
        #         self.current_grid_map[(x,y)] = 0

        #First, erase them
        for object_value in self.object_values:
            if object_value <= 1:
                continue
            obj_loc = np.where(self.current_grid_map == object_value)
            for x,y in list(zip(obj_loc[0],obj_loc[1])):
                self.current_grid_map[(x,y)] = 0

        self.done = False
        self.reward = 0
        self.info['goal'] = 0
        self.info['fire'] = 0


        self.base_grid_map = np.copy(self.current_grid_map)
        # for entity in self.entities:
        #     # free_spaces = np.where(self.current_grid_map == 0)
        #     # free_spaces = list(zip(free_spaces[0], free_spaces[1]))
        #     self.entities[entity].place(position='random-free')

        #then put them back in and deal with history
        self.history['observations'] = []
        self.history['reward'] = []
        self.history['done'] = []

        for entity in self.entities:
            entity_object = self.entities[entity]
            if entity_object.record_history:
                entity_object.history['actions'] = []
            entity_object.place(position=entity_object.position,position_coords=entity_object.position_coords)
        # for object_value in self.object_values:
        #     if object_value <= 1:
        #         continue
        #     free_spaces = []
        #     for free_space in self.free_spaces:
        #         found_spaces = np.where(self.current_grid_map == free_space)
        #         free_spaces.extend(list(zip(found_spaces[0], found_spaces[1])))
        #     the_space = random.choice(free_spaces)
        #     self.current_grid_map[the_space] = object_value
        # obs_ = {}
        # obs_['img'] = self._gridmap_to_image()
        # obs_['map'] = self.current_grid_map.copy()
        # # obs_['agent_id'] = 0 #
        # obs_['objects_id'] = self.value_to_objects
        self.state = {'grid': self.current_grid_map.copy(),
                      'img':  self._gridmap_to_image()
                        }
        return self.state #obs_#self._gridmap_to_image()#obs_

    def _gridmap_to_image(self):
        image = np.zeros((self.dims[0],self.dims[1],3), dtype=np.uint8)
        image[:] = [96,96,96]
        #put border walls in
        walls = np.where(self.current_grid_map == 1.0)
        for x,y in list(zip(walls[0],walls[1])):
            #print((x,y))
            image[x,y,:] = self.colors[self.value_to_objects[1]['color']]


        for obj_val in self.object_values:
            obj = np.where(self.current_grid_map == obj_val)
            # print('obj', obj, '\n')
            image[obj[0],obj[1],:] = self.colors[self.value_to_objects[obj_val]['color']]

        return image

    def render(self, mode='rgb_array'):
        """
        Returns an RGB image of your environment.
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        """
        if mode == 'rgb_array':
            size_factor = 100  # The higher the better
            image = self._gridmap_to_image()
            initial_img = PIL.Image.fromarray(image)
            size = tuple((np.array(initial_img.size) * size_factor).astype(int))
            initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))

            initial_img = np.flip(np.rot90(initial_img), 0)

            return initial_img  # self._gridmap_to_image()#np.array(...)  # return RGB frame suitable for video
        elif mode == 'human':
            ...  # pop up a window and render (for a human to play the game)
        else:
            super(GenericEnv, self).render(mode=mode)  # just raise an exception
        # return self._gridmap_to_image()

    @step_wrapper
    def step(self, action):
        # info = {}
        obs_ = {}
        obs = self._gridmap_to_image()
        obs = obs.reshape((1,obs.shape[0],obs.shape[1],obs.shape[2]))
        grid_map = self.current_grid_map
        entity_actions = []

        for entity in self.entities:
            ent_obj = self.entities[entity]
            ent_obj.stepCheck()
        if self.done:
            print('done')
            self.state = {'grid': self.current_grid_map.copy(),
                          'img': self._gridmap_to_image()
                          }
            return self.state, self.reward, self.done, self.info

        for entity in self.active_entities:
            if not type(self.entities[entity]) == NetworkAgent:
                action_chosen = self.active_entities[entity].getAction(obs)
                entity_actions.append(action_chosen)
                # if self.active_entities[entity].record_history:
                #     self.active_entities[entity].history['steps'].append(action_chosen)
            else:
                entity_actions.append(action)
        # print('ent_actions:',entity_actions)

        #this loop will carry out what COULD happen
        for entity, an_action in zip(self.active_entities, entity_actions):
            if an_action == 0:
                self.entities[entity].intended_position = self.entities[entity].current_position
                continue

            current_position = self.active_entities[entity].current_position
            position_function = self.action_map[an_action]
            #print('pos_func', position_function)
            intended_position = position_function(current_position)
            self.active_entities[entity].intended_position = intended_position
            if self.current_grid_map[intended_position] == 1.0: #hit a wall
                if self.wallrule:
                    self.active_entities[entity].hitWall()
                self.active_entities[entity].intended_position = self.active_entities[entity].current_position

            self.current_grid_map[current_position] = 0 #erase the person from their old spot

        #this loop carries out any action towards non-active entities (e.g. goals, reactive obstacles)
        for entity in self.active_entities:
            other_entities = [x for x in self.entities if x not in self.active_entities]
            for other in other_entities:
                if self.active_entities[entity].intended_position == self.entities[other].current_position:
                    self.entities[other].moveToMe(self.active_entities[entity])
        if self.done:
            #print('done')
            self.state = {'grid': self.current_grid_map.copy(),
                          'img': self._gridmap_to_image()
                          }
            return self.state, self.reward, self.done, self.info # self._gridmap_to_image()


        #this loop checks for collisions, carries out the consequence
        for entity in self.active_entities:
            entity_object = self.active_entities[entity]
            if entity_object.current_position == entity_object.intended_position:
                continue
            other_entities = [self.active_entities[x] for x in self.active_entities if not x == entity]
            for other_entity_object in other_entities:
                # if type(entity_object) == NetworkAgent:
                # print('type',type(entity_object),'entity obj pos=', entity_object.current_position, 'entity obj intended pos=',
                #       entity_object.intended_position)
                # print('type',type(other_entity_object),'other entity obj pos=', other_entity_object.current_position, 'other entity obj intended pos=',
                #       other_entity_object.intended_position)
                if entity_object.intended_position == other_entity_object.intended_position:
                    # print("intended postions", entity_object, other_entity_object)
                    other_entity_object.moveToMe(entity_object)
                    # print('self=',self)
                    if self.done:
                        # print("reward", self.reward, self.done)
                        # obs_['img'] = self._gridmap_to_image()
                        # obs_['map'] = self.current_grid_map.copy()
                        # obs_['objects_id'] = self.value_to_objects
                        self.state = {'grid': self.current_grid_map.copy(),
                                      'img': self._gridmap_to_image()
                                      }
                        return self.state, self.reward, self.done, self.info # obs_ # self._gridmap_to_image()
                if entity_object.current_position == other_entity_object.intended_position and other_entity_object.current_position == entity_object.intended_position:
                    other_entity_object.moveToMe(entity_object)
                    if self.done:
                        # print("reward", self.reward, self.done)
                        # obs_['img'] = self._gridmap_to_image()
                        # obs_['map'] = self.current_grid_map.copy()
                        # obs_['objects_id'] = self.value_to_objects
                        self.state = {'grid': self.current_grid_map.copy(),
                                      'img': self._gridmap_to_image()
                                      }
                        return self.state, self.reward, self.done, self.info # obs_ self._gridmap_to_image()


        if not self.done:
            for entity in self.active_entities:
                entity_object = self.entities[entity]
                entity_object.current_position = entity_object.intended_position
                # print('entities', entity_object.current_position, 'ent_type', type(entity_object))
                self.current_grid_map[entity_object.current_position] = entity_object.value
            # print("reward", self.reward, self.done)
            # obs_['img'] = self._gridmap_to_image()
            # obs_['map'] = self.current_grid_map.copy()
            # obs_['objects_id'] = self.value_to_objects
            self.state = {'grid': self.current_grid_map.copy(),
                          'img': self._gridmap_to_image()
                          }
            return self.state, self.reward, self.done, self.info # self._gridmap_to_image()
        else:
            # print("reward", self.reward, self.done)
            # obs_['img'] = self._gridmap_to_image()
            # obs_['map'] = self.current_grid_map.copy()
            # obs_['objects_id'] = self.value_to_objects
            self.state = {'grid': self.current_grid_map.copy(),
                          'img': self._gridmap_to_image()
                          }
            return self.state, self.reward, self.done, self.info # self._gridmap_to_image()