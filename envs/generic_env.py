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

import numpy as np


import threading
from time import sleep


import time

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GenericEnv(gym.Env):
    value_to_objects = {1: {'class': 'wall', 'color': 'black', 'moveTo': 0}}
    object_values = [1]
    entities = []

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





    def __init__(self,dims=(12,12),map='',agents=[{'class':'agent','color':'green','position':'random-free'}],features=[],entities=[]):

        #before anything happens, setup the map
        self.setupMap(map,dims)

        #add (dynamic, random) features to the map
        self.addMapFeatures(features)

        #Add agents to the environment (1 required).  Will be placed on reset()
        for agent in agents:
            self.entities.append(self.entity(self,entity_type=agent['class'],color=agent['color'],position=agent['position']))



        self.base_grid_map = np.copy(self.current_grid_map)

        self.action_map = {1:lambda x: ((x[0]+1)%self.dims[0],(x[1])%self.dims[1]),
                           2:lambda x: ((x[0]-1)%self.dims[0],(x[1])%self.dims[1]),
                           3:lambda x: (x[0]%self.dims[0],(x[1]-1)%self.dims[1]),
                           4:lambda x: (x[0]%self.dims[0],(x[1]+1)%self.dims[1])}

        # self.run()

    class entity:
        def __init__(self, outer, entity_type='', color='', position='random-free'):
            self.outer = outer
            self.value = outer.object_values[-1] + 1
            self.outer.object_values.append(self.value)
            self.outer.value_to_objects[self.value] = {'color': color}
            self.color = color
            self.moveTo = 'moveToDefault'

        def moveTo(self,current_position,intended_position):
            return 0

        def place(self, position='random-free'):
            if position == 'random-free':
                free_spaces = np.where(self.outer.current_grid_map == 0)
                free_spaces = list(zip(free_spaces[0], free_spaces[1]))
                free_space = random.choice(free_spaces)
                self.outer.current_grid_map[free_space] = self.value

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



    def moveToAgent(self,current_position,intended_position):
        return 0

    def moveToFire(self,current_position,intended_position):
        return 0

    def moveToWater(self,current_position,intended_position):
        return 0

    def run(self):
        time.sleep(1)
        tr = threading.Thread(target = self.update)
        tr.start()
        return True

    def update(self):
        while True:
            for goal in self.goals:
                self.value_to_objects[goal]['color'] = random.choice(list(self.colors.keys()))
            sleep(1)

    def moveToGoal(self,current_position,intended_position):
        '''What to do in the event you move to a goal'''

        current_position_value = self.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
        print('asdfadsfad',int(intended_position_value))
        if self.value_to_objects[int(intended_position_value)]['color'] == self.value_to_objects[int(current_position_value)]['color']:
            self.current_grid_map[current_position] = 0.0
            self.current_grid_map[intended_position] = current_position_value

        return 0

    def moveToObstacle(self,current_position,intended_position):
        '''What to do in the event you move to an obstacle'''
        return 0


    def reset(self):
        for entity in self.entities:
            free_spaces = np.where(self.current_grid_map == 0)
            free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            entity.place(position='random-free')


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

        # for agent_val in self.agents:
        #     agent = np.where(self.current_grid_map == agent_val)
        #     image[agent[0],agent[1],:] = self.colors[self.value_to_objects[agent_val]['color']]
        #
        # for goal_val in self.goals:
        #     goal = np.where(self.current_grid_map == goal_val)
        #     image[goal[0],goal[1],:] = self.colors[self.value_to_objects[goal_val]['color']]
        #
        # for obstacle_val in self.obstacles:
        #     obstacle = np.where(self.current_grid_map == obstacle_val)
        #     image[obstacle[0],obstacle[1],:] = self.colors[self.value_to_objects[obstacle_val]['color']]



        return image




    def step(self, action, value):
        action = int(action)
        reward, done, info = 0,0,0
        print("action", action)
        current_position = np.where(self.current_grid_map == value)
        position_function = self.action_map[action]
        intended_position = position_function(current_position)
        intended_position_value = self.current_grid_map[intended_position[0],intended_position[1]]
        print("cur",current_position)
        print("intended",intended_position,intended_position_value)



        #movement to empty space
        if intended_position_value == 0.0:
            self.current_grid_map[current_position[0],current_position[1]] = 0.0
            self.current_grid_map[intended_position[0],intended_position[1]] = value
        elif intended_position_value == 1.0:
            pass #do nothing if it's a wall
        else:
            if self.value_to_objects[int(intended_position_value)]['moveTo']:
                if type(self.value_to_objects[int(intended_position_value)]['moveTo']) == str:
                    moveFunc = getattr(self,self.value_to_objects[int(intended_position_value)]['moveTo'])
                    moveFunc(current_position,intended_position)
                else:
                    moveTo = self.value_to_objects[int(intended_position_value)]['moveTo'](current_position,intended_position)




        return self._gridmap_to_image(), reward, done, info




