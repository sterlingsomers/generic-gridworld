import numpy as np
import random
import math
from threading import Lock
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
import PIL
import time

class Entity:

    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free'):
        self.env = env
        self.value = env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.active = True

    def getAction(self,obs):
        if self.active:
            return random.choice([UP])
        else:
            return 0


    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        self.active = False
        return 1

    def place(self, position='random-free'):
        if position == 'random-free':
            free_spaces = np.where(self.env.current_grid_map == 0)
            free_spaces = list(zip(free_spaces[0], free_spaces[1]))
            free_space = random.choice(free_spaces)
            self.env.current_grid_map[free_space] = self.value

    def update(self):
        pass
        # print("regular update")

class Agent(Entity):

    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        self.env.schedule_cleanup(self.value)
        print('agent says done in moveto',self.value)
        self.env.done = 1
        return 1

class HumanAgent(Agent):
    obs = None
    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free'):
        self.size_factor = 10
        self.env = env
        self.value = env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.action = 0

    def getAction(self,obs):
        #this updates the picture
        timeout = 0.300
        action = self.action
        self.obs = obs
        self.action = 0
        start = time.time()
        while time.time()  - start < timeout:
            if not action == 0:
                self.action = 0
                return action
            else:
                action = self.action
        return action


class Advisary(Entity):

    def moveTo(self,current_position,intended_position):
        self.env.done = 1
        self.env.reward = -1

    def update(self):
        # print("advisary update")
        #find the agent
        agent_value=2.0
        for entity in self.env.entities:
            if self.env.entities[entity].entity_type == 'agent':
                agent_value = entity
        agent_location = np.where(self.env.current_grid_map == agent_value)
        obstacle_location = np.where(self.env.current_grid_map == self.value)

        rad_to_agent = math.atan2(agent_location[0] - obstacle_location[0], agent_location[1] - obstacle_location[1])
        deg_to_agent = math.degrees(rad_to_agent)
        # print(deg_to_agent)
        if deg_to_agent >= 45 and deg_to_agent <= 135.0:
            self.env.step(1, self.value)
        elif deg_to_agent >= -45 and deg_to_agent <= 45.0:
            self.env.step(4, self.value)
        elif deg_to_agent >= -135 and deg_to_agent <= 45.0:
            self.env.step(2, self.value)
        else:
            self.env.step(3, self.value)
        #below is incase it gets stuck - it will take a random move
        post_location = np.where(self.env.current_grid_map == self.value)
        if obstacle_location == post_location:
            self.env.step(random.randint(1, 4), self.value)