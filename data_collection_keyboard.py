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

human_data = pickle.load(open('sterling_model_advisary.lst','rb'))

episodes = 200
number_of_runs = 20
human = 'learning_model_v1_encode_everything_EXCEPTNOOP_20percentcutoff'
write_data = True
run_data = []


data = {'environment_episode_data': [], 'player_episode_data': [], 'stuck': [], 'advisary_episode_data': [],
        'mismatch': 5, 'decay': 0.5, 'noise': 0.0, 'temperature':0.75}

# env = envs.generic_env.GenericEnv(map='small-empty',features=[{'entity_type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
env = envs.generic_env.GenericEnv(dims=(10,10))#,features=[{'entity_type':'obstacle','start_number':5,'color':'pink','moveTo':'moveToObstacle'}])
goal = Goal(env,entity_type='goal',color='green')
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
# player3 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame)
player3 = ACTR(env, data=human_data, mismatch_penalty=data['mismatch'],decay=data['decay'],noise=data['noise'],temperature=data['temperature'],multiprocess=False,processes=5)
#player3 = TrainedAgent(env,color='aqua',model_name='net_vs_pred_best_noop')
#player4 = AIAgent(env,entity_type='agent',color='pink',pygame=pygame)
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data',position='near-goal')
#advisary2 = ChasingBlockingAdvisary(env,entity_type='advisary',color='pink',obs_type='data')

env.setRecordHistory()
player3.setRecordHistory(history_dict={'actions':[],'saliences':[],'stuck':[]})
advisary.setRecordHistory(history_dict={'actions':[]})



human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

size_factor = 40

initial_image_data = env.reset()
player3.last_observation = np.zeros(player3.env.current_grid_map.shape)
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




done = False

advisary_action_map = {'advisary_up': UP, 'advisary_down': DOWN, 'advisary_noop': NOOP, 'advisary_left': LEFT,
                               'advisary_right': RIGHT}

clock = pygame.time.Clock()
    #while not player3.quit:
for i in range(number_of_runs):
    player3.memory.clear()
    data = {'environment_episode_data': [], 'player_episode_data': [], 'stuck': [], 'advisary_episode_data': [],
            'mismatch': data['mismatch'], 'decay': data['decay'], 'noise': data['noise'], 'temperature': data['temperature']}
    for i in range(episodes):
        print("Episode", i)
        episode_done = False
        steps = 0
        while not episode_done:
            print('step', steps)
            steps += 1
            if steps == 50:
                player3.stuck = 1
            if steps > 50:
                data['environment_episode_data'].append(env.history.copy())
                data['player_episode_data'].append(player3.history.copy())
                data['advisary_episode_data'].append(advisary.history.copy())
                env.setRecordHistory()
                player3.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
                advisary.setRecordHistory(history_dict={'actions': []})
                episode_done = True

                obs = env.reset()
                chunk = {}
                encode_chunk = True
                if player3.last_observation.any():
                    last_predator = np.where(player3.last_observation == 4)
                    last_predator = (int(last_predator[0]), int(last_predator[1]))
                    last_action = data['advisary_episode_data'][-1]['actions'][-1]
                    chunk = player3.gridmap_to_symbols(player3.last_observation.copy(), player3.value,
                                                       player3.env.value_to_objects)
                    for advisary_action in advisary_action_map:
                        chunk[advisary_action] = int((last_action == advisary_action_map[advisary_action]))
                    for key in chunk:
                        if 'distance' in key:
                            chunk[key] = chunk[key] / player3.max_distance
                if chunk:
                    player3.memory.learn(**chunk)
                player3.last_observation = np.zeros(player3.env.current_grid_map.shape)
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
                player3.action = key_pressed
                if done:
                    obs = env.reset()
                    player3.last_observation = np.zeros(player3.env.current_grid_map.shape)
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
                data['advisary_episode_data'].append(advisary.history.copy())
                # if data['environment_episode_data']['reward'][-1] == -1:
                #     print('why?')
                env.setRecordHistory()
                player3.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
                advisary.setRecordHistory(history_dict={'actions':[]})
                episode_done = True
                chunk = {}
                if player3.last_observation.any():
                    last_predator = np.where(player3.last_observation == 4)
                    last_predator = (int(last_predator[0]), int(last_predator[1]))
                    last_action = data['advisary_episode_data'][-1]['actions'][-1]
                    chunk = player3.gridmap_to_symbols(player3.last_observation.copy(), player3.value, player3.env.value_to_objects)
                    for advisary_action in advisary_action_map:
                        chunk[advisary_action] = int((last_action == advisary_action_map[advisary_action]))
                    for key in chunk:
                        if 'distance' in key:
                            chunk[key] = chunk[key] / player3.max_distance
                if chunk:
                    player3.memory.learn(**chunk)

                obs = env.reset()
                player3.last_observation = np.zeros(player3.env.current_grid_map.shape)
                obs = PIL.Image.fromarray(obs)
                size = tuple((np.array(obs.size) * size_factor).astype(int))
                obs = np.array(obs.resize(size, PIL.Image.NEAREST))
                surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
                display.blit(surf, (0, 0))
    run_data.append(data.copy())

#print("quitting?", player3.quit)
timestr = time.strftime("%Y%m%d-%H%M%S")
if write_data:
    pickle.dump(run_data,open(human + '_mismatch_' + repr(data['mismatch']).replace('.','') + '_decay_' + repr(data['decay']).replace('.','') + '_noise_' + repr(data['noise']).replace('.','') + '_temperature_' + repr(data['temperature']).replace('.','') + '_runs_' + repr(number_of_runs) + '_' + timestr + '.dict','wb'))
pygame.quit()
