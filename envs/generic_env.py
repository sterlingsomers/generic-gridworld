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

# from common.graph import *


import time

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

class gridObject():
    colors = {
        'red':(252,3,3),
        'green':(3,252,78),
        'blue':(3,15,252),
        'yellow':(252,252,3),
        'pink':(227,77,180),
        'purple':(175,4,181),
        'black':(0,0,0)
    }

    def __init__(self, color='random', category='obstacle'):
        if color=='random':
            self.color = self.colors[random.choice(self.colors.keys())]

        else:
            self.color = self.colors[color]

class wall(gridObject):

    def __init__(self, color='black'):
        self.color = self.colors['black']

    def moveTo(self,env):
        '''Do nothing because there's a wall there.'''
        return 0



class GenericEnv(gym.Env):
    value_to_objects = {1: {'class': 'wall', 'color': (0, 0, 0), 'moveTo': 0}}
    object_values = [1]
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
        'black': (0, 0, 0)
    }

    def __init__(self,dims=(12,12),agents={'number':1,'position':['random'],'color':['blue']},
                 goals={'number':1,'position':['random'],'color':['red']},
                 obstacles={'number':1, 'position':['random'], 'color':['green']}):



        self.agents = []
        for color,position in zip(agents['color'],agents['position']):
            value = self.object_values[-1] + 1
            self.agents.append(value)
            self.value_to_objects[value] = {'class':'agent', 'color':color,'moveTo':0,'position':position}
            self.object_values.append(value)


        self.goals = []
        if goals:
            goal_defaults = {'number':1,'position':['random'],'color':['red']}
            for key in [x for x in goal_defaults.keys() if not x in goals.keys()]:
                goals[key] = goal_defaults[key]
            for color,position in zip(goals['color'],goals['position']):
                value = self.object_values[-1] + 1
                self.goals.append(value)
                self.value_to_objects[value] = {'class':'agent', 'color':color, 'moveTo':self.moveToGoal, 'position':position}
                self.object_values.append(value)

        self.obstacles = []
        if obstacles:
            obstacle_defaults = {'number': 1, 'position': ['random'], 'color': ['green']}
            for key in [x for x in obstacle_defaults.keys() if not x in obstacles.keys()]:
                obstacles[key] = obstacle_defaults[key]
            for color, position in zip(obstacles['color'], obstacles['position']):
                value = self.object_values[-1] + 1
                self.goals.append(value)
                self.value_to_objects[value] = {'class': 'obstacle', 'color': color, 'moveTo': self.moveToObstacle,
                                                'position': position}
                self.object_values.append(value)


        self.dims = dims
        self.position_node_map = {}
        self.position_key_color_map = {}
        self.current_grid_map = np.zeros(dims)
        #make border the walls (walls = 1)
        self.current_grid_map[0,:] = [1] * dims[0]
        self.current_grid_map[:,0] = [1] * dims[1]
        self.current_grid_map[:,-1] = [1] * dims[1]
        self.current_grid_map[-1,:] = [1] * dims[0]



        self.base_grid_map = np.copy(self.current_grid_map)

        self.action_map = {1:lambda x: (x[0]+1,x[1]),
                           2:lambda x: (x[0]-1,x[1]),
                           3:lambda x: (x[0],x[1]-1),
                           4:lambda x: (x[0],x[1]+1)}



    def moveToGoal(self,current_position,intended_position):
        '''What to do in the event you move to a goal'''
        current_position_value = self.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
        self.current_grid_map[current_position] = 0.0
        self.current_grid_map[intended_position] = current_position_value
        return 0

    def moveToObstacle(self,current_position,intended_position):
        '''What to do in the event you move to an obstacle'''
        current_position_value = self.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.current_grid_map[intended_position[0], intended_position[1]]
        self.current_grid_map[current_position] = 0.0
        self.current_grid_map[intended_position] = current_position_value
        free_spaces = np.where(self.current_grid_map == 0)
        free_spaces = list(zip(free_spaces[0], free_spaces[1]))

        free_space = random.choice(free_spaces)
        self.current_grid_map[free_space[0], free_space[1]] = intended_position_value



    def reset(self):


        # insert the agents
        for agent in self.agents:
            free_spaces = np.where(self.current_grid_map == 0)
            free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            if self.value_to_objects[agent]['position'] == 'random':
                free_space = random.choice(free_spaces)
                # self.agent_position = free_space
                self.current_grid_map[free_space[0], free_space[1]] = agent

        #insert the goals
        for goal in self.goals:
            free_spaces = np.where(self.current_grid_map == 0)
            free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            if self.value_to_objects[agent]['position'] == 'random':
                free_space = random.choice(free_spaces)
                # self.agent_position = free_space
                self.current_grid_map[free_space[0], free_space[1]] = goal

        for obstacle in self.obstacles:
            free_spaces = np.where(self.current_grid_map == 0)
            free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            if self.value_to_objects[agent]['position'] == 'random':
                free_space = random.choice(free_spaces)
                # self.agent_position = free_space
                self.current_grid_map[free_space[0], free_space[1]] = obstacle
        return self._gridmap_to_image()

        #

    def _gridmap_to_image(self):
        image = np.zeros((self.dims[0],self.dims[1],3), dtype=np.uint8)
        image[:] = [96,96,96]
        #put border walls in
        walls = np.where(self.current_grid_map == 1.0)
        for x,y in list(zip(walls[0],walls[1])):
            #print((x,y))
            image[x,y,:] = self.value_to_objects[1]['color']




        for agent_val in self.agents:
            agent = np.where(self.current_grid_map == agent_val)
            image[agent[0],agent[1],:] = self.colors[self.value_to_objects[agent_val]['color']]

        for goal_val in self.goals:
            goal = np.where(self.current_grid_map == goal_val)
            image[goal[0],goal[1],:] = self.colors[self.value_to_objects[goal_val]['color']]

        for obstacle_val in self.obstacles:
            obstacle = np.where(self.current_grid_map == goal_val)
            image[goal[0],goal[1],:] = self.colors[self.value_to_objects[obstacle_val]['color']]



        return image




    def step(self, action):
        action = int(action)
        reward, done, info = 0,0,0
        print("action", action)
        current_position = np.where(self.current_grid_map == 2)
        position_function = self.action_map[action]
        intended_position = position_function(current_position)
        intended_position_value = self.current_grid_map[intended_position[0],intended_position[1]]
        print("cur",current_position)
        print("intended",intended_position,intended_position_value)



        #movement to empty space
        if intended_position_value == 0.0:
            self.current_grid_map[current_position[0],current_position[1]] = 0.0
            self.current_grid_map[intended_position[0],intended_position[1]] = 2.0
        elif intended_position_value == 1.0:
            pass #do nothing if it's a wall
        else:
            moveTo = self.value_to_objects[int(intended_position_value)]['moveTo'](current_position,intended_position)




        return self._gridmap_to_image(), reward, done, info




