#This project neds a readme. 

##Intention
The intention of this project is to make a generic gridworld that can be (easily) setup for whatever
gridworld task you're interested in modeling. 

##Default Example
The default example is a 10x10 grid with border walls.  

run keyboard_agent.py in the root directory to run the example.



#File Description
###/envs/core.py
core.py has all the agent classes.
#####class Entity
This is the base class of all non-features. 
...

#Version Notes


#Sample Code
--version update (inheritance_update)

##Generating an environment
####Creating an empty environment with border walls, with specific dimensions
env = envs.generic_env.GenericEnv(dims=(10,10))

####Creating an evironment using a map
env = envs.generic_env.GenericEnv(map='small-empty')

##Adding entities
####Adding a goal
goal = Goal(env, entity_type='goal',color='green')

*Note: you may want to write your own moveToMe(entity_object) function to assign rewards, etc.*

####Adding an AI agent
ai = AIAgent(env, entity_type='agent',color='orange')

*Note: default AI is not very smart. Edit the getAction() to suit your needs*

####Adding a human player
*Note: Currently only 1 human player is possible. This example is extended, demonstrating a number of things including introductin your own reactive entity, human agents (and displaying the env with pygame). The code presented is not clean by any means but should provide some insight.*
```python
#!/usr/bin/env python
from __future__ import print_function
from threading import Thread
import sys, gym, time
from pyglet.window import key

import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *
from scipy.spatial.distance import cityblock
from itertools import permutations
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pygame
import numpy as np

import PIL

#A new class to extend old classes
class Obstacle(Entity):
    def __init__(self, env, obs_type='image', entity_type='goal', color='', position='random-free'):
        super().__init__(env, obs_type, entity_type, color, position)

    def moveToMe(self, entity_object):
        entity_object.intended_position = entity_object.current_position
        return 0


env = envs.generic_env.GenericEnv(dims=(10,10))
goal = Goal(env,entity_type='goal',color='green')
obstacles = []
for i in range(3):
    obstacles.append(Obstacle(env, color='yellow'))

player3 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame)
player4 = AIAgent(env,entity_type='agent',color='orange')
player4 = AIAgent(env,entity_type='agent',color='pink')
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data',position='near-goal')






human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

size_factor = 10

initial_image_data = env.reset()
initial_img = PIL.Image.fromarray(initial_image_data)
size = tuple((np.array(initial_img.size) * size_factor).astype(int))
initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))

initial_img = np.flip(np.rot90(initial_img),0)
#one noop
pygame.init()
display = pygame.display.set_mode(initial_img.shape[:2],0,32)
background = pygame.surfarray.make_surface(initial_img)
background = background.convert()
display.blit(background,(0,0))
pygame.display.update()
game_done = False
running = True

# t = Thread(target=run_player2)
# t.start()
clock = pygame.time.Clock()
while running:
    pygame.display.update()
    key_pressed = 0
    pygame.time.delay(50)
    # free_spaces = env.free_spaces + list(env.entities.keys()) + [3]
    # free_spaces.remove(advisary.value)
    # env.getPathTo((1,1),(18,6),free_spaces=free_spaces)
    obs, r, done, info = env.step([])

    obs = PIL.Image.fromarray(obs)
    size = tuple((np.array(obs.size) * size_factor).astype(int))
    obs = np.array(obs.resize(size, PIL.Image.NEAREST))
    surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
    display.blit(surf, (0, 0))
    pygame.display.update()
    for event in pygame.event.get():


        if event.type == pygame.QUIT:
            running = False


    if key_pressed and not game_done:
        if done:
            obs = env.reset()
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.blit(surf, (0,0))
        else:
            #pygame.surfarray.blit_array(background,obs)
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs),0))
            #pygame.transform.rotate(surf,180)
            display.blit(surf, (0,0))


    # pygame.time.delay(100)
    pygame.display.update()
    # clock.tick(100)
    if done:
        obs = env.reset()
        obs = PIL.Image.fromarray(obs)
        size = tuple((np.array(obs.size) * size_factor).astype(int))
        obs = np.array(obs.resize(size, PIL.Image.NEAREST))
        surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
        display.blit(surf, (0, 0))

pygame.quit()
```


