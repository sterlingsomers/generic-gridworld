import gym
import sys
import os
import copy
from gym import spaces
from gym.utils import seeding
import random
import itertools
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from common.maps import *
import numpy as np



import time

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GenericEnv(gym.Env):
    value_to_objects = {1: {'class': 'wall', 'color': 'black', 'moveTo': 0}}
    object_values = [1]
    entities = {} #indexed by object value
    backup_values = {}
    actions = {}
    reward = 0
    done = 0
    to_clean = []
    permanents = []
    free_spaces = [0]

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





    def __init__(self,dims=(12,12),map='',agents=[],features=[],entities=[]):

        #before anything happens, setup the map
        self.setupMap(map,dims)

        #add (dynamic, random) features to the map
        self.addMapFeatures(features)


        #Add agents to the environment (1 required).  Will be placed on reset()
        for agent in agents:
            class_to_use = getattr(sys.modules[__name__], agent['class'])
            class_to_use(self,entity_type=agent['entity_type'],color=agent['color'],position=agent['position'])


        #add other entities to the environment
        for entity in entities:
            class_to_use = getattr(sys.modules[__name__], entity['class'])
            class_to_use(self,entity_type=entity['entity_type'],color=entity['color'],position=entity['position'])



        self.base_grid_map = np.copy(self.current_grid_map)

        self.action_map = {1:lambda x: ((x[0]+1)%self.dims[0],(x[1])%self.dims[1]),
                           2:lambda x: ((x[0]-1)%self.dims[0],(x[1])%self.dims[1]),
                           3:lambda x: (x[0]%self.dims[0],(x[1]-1)%self.dims[1]),
                           4:lambda x: (x[0]%self.dims[0],(x[1]+1)%self.dims[1]),
                           0:lambda x: (x[0],x[1])}

        #Run the dynamic environment
        #self.run()

    def getPathTo(self,start_location,end_location, free_spaces=[]):
        '''An A* algorithm to get from one point to another.
        free_spaces is a list of values that can be traversed.
        start_location and end_location are tuple point values.

        Returns a map with path denoted by -1 values. Inteded to use np.where(path == -1).'''
        pathArray = np.full(self.dims,0)

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
        stop = False
        while True:
            current_value += 1
            target_value = current_value + 1
            test_points = np.where(pathArray == current_value)
            test_points = list(zip(test_points[0],test_points[1]))
            still_looking = False
            for test_point in test_points:
                for direction in directions:
                    if pathArray[self.action_map[direction](test_point)] and pathArray[self.action_map[direction](test_point)] + current_value <= target_value:
                        pathArray[self.action_map[direction](test_point)] = target_value
                        still_looking = True
                    # print(self.action_map[direction](test_point), end_location)
                    if self.action_map[direction](test_point) == end_location:
                        pathArray[end_location] = - 1
                        stop = True
                        break
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
                    pathArray[current_point] = -1
                    return pathArray


    def getGoalValue(self):
        for value in self.value_to_objects:
            if 'class' in self.value_to_objects[value]:
                if self.value_to_objects[value]['class'] == 'goal':
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
            object_value = self.object_values[-1] + 1
            self.object_values.append(object_value)
            self.value_to_objects[object_value] = {'class':feature['type'], 'color':feature['color'], 'moveTo':feature['moveTo']}
            free_spaces = np.where(self.current_grid_map == 0)
            random_free_space_i = random.choice(free_spaces[0])
            random_free_space_j = random.choice(free_spaces[1])
            random_free_space = (random_free_space_i,random_free_space_j)
            self.current_grid_map[random_free_space] = object_value
            #pertubation = np.random.randint(-1, 1, (1,2))
            for i in range(n_features - 1):
                free_spaces = np.where(self.current_grid_map == 0)
                free_spaces = [(x,y) for x,y in zip(free_spaces[0],free_spaces[1]) if x >= random_free_space_i - 1 and x <= random_free_space_i + 1 and
                               y >= random_free_space_j - 1 and y <= random_free_space_j + 1]
                free_space = random.choice(free_spaces)
                self.current_grid_map[free_space] = object_value




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
        return 0

    def moveToObstacle(self,current_position,intended_position):
        '''What to do in the event you move to an obstacle'''
        return 0


    def reset(self):
        self.done = False
        self.reward = 0
        for entity in self.entities:
            # free_spaces = np.where(self.current_grid_map == 0)
            # free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            self.entities[entity].place(position='random-free')


        return self._gridmap_to_image()



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
            image[obj[0],obj[1],:] = self.colors[self.value_to_objects[obj_val]['color']]



        return image




    def step(self):
        info = 0
        obs = self._gridmap_to_image()
        actions = []
        count = 0
        for entity in self.entities:
            actions.append(self.entities[entity].getAction(obs))
        for entity, action in zip(self.entities, actions):
            count += 1
            if action == 0:
                continue
            print('action',action, 'entity', entity, 'count', count)
            current_position = np.where(self.current_grid_map == self.entities[entity].value)
            position_function = self.action_map[action]
            intended_position = position_function(current_position)
            intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
            current_position_value = self.current_grid_map[current_position[0], current_position[1]]
            if not len(current_position[0]):
                continue
            if intended_position_value == 0.0:
                self.current_grid_map[current_position[0],current_position[1]] = 0.0
                self.current_grid_map[intended_position[0],intended_position[1]] = self.entities[entity].value
            elif intended_position_value == 1.0:
                pass #do nothing if it's a wall
            else:
                if int(intended_position_value) in self.entities:
                    self.entities[int(intended_position_value)].moveTo(current_position,intended_position)
                else:
                    moveTo = getattr(self,self.value_to_objects[int(intended_position_value)]['moveTo'])
                    moveTo(current_position,intended_position)
                    continue
        bv = self.backup_values.copy()
        for position in bv:
            grid_value = self.current_grid_map[(position[0],position[1])]
            if grid_value in self.permanents:
                self.current_grid_map[(position[0],position[1])] = self.backup_values[(position[0],position[1])]
                del self.backup_values[(position[0],position[1])]

                # if (int(current_position[0]), int(current_position[1])) in self.backup_values:
                #     self.current_grid_map[(int(current_position[0]), int(current_position[1]))] = self.backup_values[
                #         (int(current_position[0]), int(current_position[1]))]
                #     del self.backup_values[(int(current_position[0]), int(current_position[1]))]


        self.cleanup()
        return self._gridmap_to_image(), self.reward, self.done, info












