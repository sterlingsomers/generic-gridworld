#!/usr/bin/env python


#note to self: may want to normalize the angles BEFORE similarity function
#That way you're not compressing their differences.



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

import pickle
import PIL
import time

#A new class to extend old classes


# env = envs.generic_env.GenericEnv(map='small-empty',features=[{'entity_type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
env = envs.generic_env.GenericEnv(dims=(10,10))#,features=[{'entity_type':'obstacle','start_number':5,'color':'pink','moveTo':'moveToObstacle'}])
goal = Goal(env,entity_type='goal',color='green')
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
player3 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame)
#player4 = AIAgent(env,entity_type='agent',color='pink',pygame=pygame)
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data',position='near-goal')
#advisary2 = ChasingBlockingAdvisary(env,entity_type='advisary',color='pink',obs_type='data')

env.setRecordHistory()
player3.setRecordHistory()



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

import math

# t = Thread(target=run_player2)
# t.start()

data = {'environment_episode_data':[],'player_episode_data':[]}
episodes = 100
human = 'sterling'

clock = pygame.time.Clock()
while not player3.quit:
    pygame.display.update()
    key_pressed = 0
    pygame.time.delay(0)

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
        player3.action = key_pressed
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
        data['environment_episode_data'].append(env.history.copy())
        data['player_episode_data'].append(player3.history.copy())

        obs = env.reset()
        obs = PIL.Image.fromarray(obs)
        size = tuple((np.array(obs.size) * size_factor).astype(int))
        obs = np.array(obs.resize(size, PIL.Image.NEAREST))
        surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
        display.blit(surf, (0, 0))

print("quitting?", player3.quit)
timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(data,open(human + timestr + '.dict','wb'))
pygame.quit()
