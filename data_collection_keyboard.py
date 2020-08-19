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

human_data = pickle.load(open('symbolic_data_sterlingV220200207-152603.lst','rb'))
data = {'environment_episode_data':[],'player_episode_data':[],'stuck':[]}
episodes = 100
outputFileName = 'sterling_model_with_adivsary_data_'
write_data = False

# env = envs.generic_env.GenericEnv(map='small-empty',features=[{'entity_type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
env = envs.generic_env.GenericEnv(map='small-portals',wallrule=False)#,features=[{'entity_type':'obstacle','start_number':5,'color':'pink','moveTo':'moveToObstacle'}])
goal = RunAwayGoal(env, obs_type='data',entity_type='goal',color='green',pygame=pygame,displayPlan=True)
#player1 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame)
player1 = HumanAgent(env,entity_type='agent',color='pink',pygame=pygame)
#player2 = AIAgent(env,entity_type='agent',color='pink')#,pygame=pygame,mapping={'i':UP,'j':LEFT,'k':DOWN,'l':RIGHT,'m':NOOP})

player3 = TrainedAgent(env,model_filepath='/Users/paulsomers/gridworlds/generic/_files/models/lioness/lioness.pb', color='aqua')
#player3 = ChasingBlockingAdvisary(env,entity_type='advisary',color='blue',position='near-goal')
#player4 = ChasingBlockingAdvisary(env,entity_type='advisary',color='blue',position='near-goal')



# player3 = ACTR(env, data=human_data, mismatch_penalty=20,noise=0.25,multiprocess=True,processes=5)
#player3 = TrainedAgent(env,color='aqua',model_name='net_vs_pred_best_noop')
#player4 = AIAgent(env,entity_type='agent',color='pink',pygame=pygame)
# advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data',position='near-goal')
#advisary2 = ChasingBlockingAdvisary(env,entity_type='advisary',color='pink',obs_type='data')

# env.setRecordHistory()
# player1.setRecordHistory(history_dict={'actions':[],'saliences':[],'stuck':[]})
# advisary.setRecordHistory(history_dict={'actions':[]})



human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

size_factor = 20

initial_image_data = env.reset()
initial_img = PIL.Image.fromarray(initial_image_data)
size = tuple((np.array(initial_img.size) * size_factor).astype(int))
initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))

initial_img = np.flip(np.rot90(initial_img),0)
#one noop
pygame.init()
display = pygame.display.set_mode((initial_img.shape[0]+500, initial_img.shape[1] + 150))#initial_img.shape[:2],0,32)

#Set the display so that the graphics can be modified by the agent if needed
player1.setDisplay(display)
#player2.setDisplay(display)
goal.setDisplay(display)
#player2.setDisplay(display)
# player3.setDisplay(display)
background = pygame.surfarray.make_surface(initial_img)
background = background.convert()
display.blit(background,(0,0))
pygame.display.update()
game_done = False
running = True

import math




done = False


clock = pygame.time.Clock()
#while not player3.quit:
for i in range(episodes):
    print("Episode", i)
    episode_done = False
    steps = 0
    while not episode_done:
        print('step', steps)
        steps += 1
        if steps == 50:
            player1.stuck = 1
        if steps > 50:
            # data['environment_episode_data'].append(env.history.copy())
            # data['player_episode_data'].append(player3.history.copy())
            #
            # env.setRecordHistory()
            # player1.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})


            episode_done = True

            obs = env.reset()
            obs = PIL.Image.fromarray(obs)
            size = tuple((np.array(obs.size) * size_factor).astype(int))
            obs = np.array(obs.resize(size, PIL.Image.NEAREST))
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.blit(surf, (0, 0))

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
            player1.action = key_pressed
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
            # data['environment_episode_data'].append(env.history.copy())
            # data['player_episode_data'].append(player1.history.copy())
            # # data['advisary_episode_data'].append(advisary.history.copy())
            # env.setRecordHistory()
            # player1.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
            # advisary.setRecordHistory(history_dict={'actions':[]})
            episode_done = True

            obs = env.reset()
            obs = PIL.Image.fromarray(obs)
            size = tuple((np.array(obs.size) * size_factor).astype(int))
            obs = np.array(obs.resize(size, PIL.Image.NEAREST))
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.blit(surf, (0, 0))

#print("quitting?", player3.quit)
timestr = time.strftime("%Y%m%d-%H%M%S")
if write_data:
    pickle.dump(data,open(outputFileName + timestr + '.dict','wb'))
pygame.quit()
