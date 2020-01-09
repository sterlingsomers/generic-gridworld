import numpy as np
import random
import math
from threading import Lock
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from scipy.spatial.distance import cityblock
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



        return {'up':up,'down':down,'left':left,'right':right}

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
    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free',pygame='None'):
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
        self.pygame = pygame

    def getAction(self,obs):
        #this updates the picture
        key_pressed = None
        while key_pressed == None:
            event = self.pygame.event.wait()
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_LEFT: key_pressed = LEFT
                if event.key == self.pygame.K_RIGHT: key_pressed = RIGHT
                if event.key == self.pygame.K_DOWN: key_pressed = DOWN
                if event.key == self.pygame.K_UP: key_pressed = UP

        return key_pressed
        # print("event", event)
        # timeout = 5.500
        # action = self.action
        # self.obs = obs
        # self.action = 0
        # start = time.time()
        # while time.time()  - start < timeout:
        #     if not action == 0:
        #         self.action = 0
        #         return action
        #     else:
        #         action = self.action
        # return action


class Advisary(Entity):
    def getAgents(self):
        agents = []
        for entity in self.env.entities:
            if isinstance(self.env.entities[entity],Agent):
                agents.append(entity)
        return agents

    def getClosestTarget(self,targets):
        '''Returns the value of the closest target. Targts is a list of values.  Random target is chosen for ties'''
        distance_by_id = {}
        my_location = np.where(self.env.current_grid_map == self.value)
        for agent_value in targets:
            target_location = np.where(self.env.current_grid_map == agent_value)
            distance_by_id[agent_value] = cityblock(my_location,target_location)
        minval = min(distance_by_id.values())
        keys = [k for k, v in distance_by_id.items() if v==minval]
        return random.choice(keys)


    def moveTo(self,current_position,intended_position):
        self.env.done = 1
        self.env.reward = -1

    def getAction(self,obs):
        agents = self.getAgents()
        target = self.getClosestTarget(agents)

        target_location = np.where(self.env.current_grid_map == target)
        my_location = np.where(self.env.current_grid_map == self.value)

        rad_to_agent = math.atan2(target_location[0] - my_location[0], target_location[1] - my_location[1])
        deg_to_agent = math.degrees(rad_to_agent)
        # print(deg_to_agent)
        if deg_to_agent >= 45 and deg_to_agent <= 135.0:
            return 1
        elif deg_to_agent >= -45 and deg_to_agent <= 45.0:
            return 4
        elif deg_to_agent >= -135 and deg_to_agent <= 45.0:
            return 2
        else:
            return 3
