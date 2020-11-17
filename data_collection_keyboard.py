#!/usr/bin/env python


#note to self: may want to normalize the angles BEFORE similarity function
#That way you're not compressing their differences.



from __future__ import print_function
from threading import Thread
import sys, gym, time
# from pyglet.window import key

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

#some imports for the participant stuff
import glob, os

pygame.init()
mission_configuration = pickle.load(open('100position.netw','rb'))
# mission_configuration = [None] * 101
# human_data = pickle.load(open('symbolic_data_sterlingV220200207-152603.lst','rb'))
os.chdir('./participant_data')
#determine the participant number
participant_files = []
for file in glob.glob('*.ptd'):
    participant_files.append(file)
participantNumber = len(participant_files)






data = {'environment_episode_data':[],'player_episode_data':[],'stuck':[],'advisary_episode_data':[]}
episodes = 100
outputFileName = 'participant-'
write_data = True

# env = envs.generic_env.GenericEnv(map='small-empty',features=[{'entity_type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
env = envs.generic_env.GenericEnv(dims=(10,10))#,features=[{'entity_type':'obstacle','start_number':5,'color':'pink','moveTo':'moveToObstacle'}])
goal = Goal(env,entity_type='goal',color='green')
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
player3 = HumanAgent(env,entity_type='agent',color='aqua',pygame=pygame)
# player3 = TrainedAgent(env, model_filepath='/Users/paulsomers/gridworlds/generic/_files/models/networkb.pb',color='aqua')
# player3 = ACTR(env, data=human_data, mismatch_penalty=20,noise=0.25,multiprocess=True,processes=5)
#player3 = TrainedAgent(env,color='aqua',model_name='net_vs_pred_best_noop')
#player4 = AIAgent(env,entity_type='agent',color='pink',pygame=pygame)
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data',position='near-goal')
#advisary2 = ChasingBlockingAdvisary(env,entity_type='advisary',color='pink',obs_type='data')

env.setRecordHistory()
# player3.setRecordHistory(history_dict={'actions':[],'saliences':[],'stuck':[]})
player3.setRecordHistory(history_dict={'actions':[]})
advisary.setRecordHistory(history_dict={'actions':[]})



wins = 0
losses = 0
scoreboard_font_wins = pygame.font.Font('freesansbold.ttf',16)
scoreboard_font_losses = pygame.font.Font('freesansbold.ttf',16)
scoreboard_font_steps = pygame.font.Font('freesansbold.ttf',16)
txtX = 30
txtY = 300
def show_score(display,wins,losses,min_steps,x,y):

    wins_txt = scoreboard_font_wins.render("Wins: " + repr(wins), True, (255,255,255))
    losses_txt = scoreboard_font_losses.render("Losses: " + repr(losses), True, (255,255,255))
    best_score = scoreboard_font_steps.render('Map minimum:' + repr(min_steps), True, (255,255,255))
    display.blit(best_score, (x, y-15))
    display.blit(wins_txt, (x,y))
    display.blit(losses_txt,(x,y+15))

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

size_factor = 30

initial_image_data = env.reset(config=mission_configuration[0])
initial_img = PIL.Image.fromarray(initial_image_data)
size = tuple((np.array(initial_img.size) * size_factor).astype(int))
initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))

initial_img = np.flip(np.rot90(initial_img),0)
#one noop
pygame.init()
window_size = (initial_img.shape[:2][0]+0, initial_img.shape[:2][1]+50)
display = pygame.display.set_mode(window_size,0,32)
background = pygame.surfarray.make_surface(initial_img)
background = background.convert()
display.blit(background,(0,0))
pygame.display.update()
if not mission_configuration[0] == None:
    show_score(display, wins, losses, mission_configuration[0]['steps'], txtX, txtY)#mission_configuration[0]['steps']
else:
    show_score(display, wins, losses, None, txtX, txtY)
game_done = False
running = True

import math

# t = Thread(target=run_player2)
# t.start()




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
            player3.stuck = 1
        if steps > 50:
            data['environment_episode_data'].append(env.history.copy())
            data['player_episode_data'].append(player3.history.copy())
            data['advisary_episode_data'].append(advisary.history.copy())
            env.setRecordHistory()
            player3.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
            advisary.setRecordHistory(history_dict={'actions': []})

            episode_done = True

            obs = env.reset(config=mission_configuration[i+1])
            obs = PIL.Image.fromarray(obs)
            size = tuple((np.array(obs.size) * size_factor).astype(int))
            obs = np.array(obs.resize(size, PIL.Image.NEAREST))
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.fill((0,0,0))
            display.blit(surf, (0, 0))
            pygame.display.update()
            if not mission_configuration[i+1] == None:
                show_score(display, wins, losses, mission_configuration[i+1]['steps'],txtX, txtY)#mission_configuration[i+1]['steps']
            else:
                show_score(display, wins, losses, None, txtX, txtY)
        pygame.display.update()
        key_pressed = 0
        pygame.time.delay(0)

        obs, r, done, info = env.step([])


        obs = PIL.Image.fromarray(obs)
        size = tuple((np.array(obs.size) * size_factor).astype(int))
        obs = np.array(obs.resize(size, PIL.Image.NEAREST))
        surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
        display.fill((0, 0, 0))
        display.blit(surf, (0, 0))
        pygame.display.update()
        if not mission_configuration[i] == None:
            show_score(display, wins, losses, mission_configuration[i]['steps'],txtX, txtY)#mission_configuration[i+1]['steps']
        else:
            show_score(display, wins, losses, None, txtX, txtY)
        for event in pygame.event.get():


            if event.type == pygame.QUIT:
                running = False


        if key_pressed and not game_done:
            player3.action = key_pressed
            if done:
                obs = env.reset(config=mission_configuration[i+1])
                surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
                display.fill((0, 0, 0))
                display.blit(surf, (0,0))
                if not mission_configuration[i] == None:
                    show_score(display, wins, losses, mission_configuration[i]['steps'], txtX, txtY)#mission_configuration[i+1]['steps']
                else:
                    show_score(display, wins, losses, None, txtX, txtY)
            else:
                #pygame.surfarray.blit_array(background,obs)
                surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs),0))
                #pygame.transform.rotate(surf,180)
                display.fill((0, 0, 0))
                display.blit(surf, (0,0))
                if not mission_configuration[i] == None:
                    show_score(display, wins, losses, mission_configuration[i]['steps'], txtX, txtY)#mission_configuration[i+1]['steps']
                else:
                    show_score(display, wins, losses, None, txtX, txtY)


        # pygame.time.delay(100)
        pygame.display.update()
        if not mission_configuration[i] == None:
            show_score(display,wins,losses,mission_configuration[i]['steps'],txtX, txtY)#mission_configuration[i+1]['steps']
        else:
            show_score(display, wins, losses, None, txtX, txtY)
        # clock.tick(100)
        if done:
            data['environment_episode_data'].append(env.history.copy())
            data['player_episode_data'].append(player3.history.copy())
            data['advisary_episode_data'].append(advisary.history.copy())
            wins = wins + data['environment_episode_data'][-1]['reward'][-1] if data['environment_episode_data'][-1]['reward'][-1] > 0 else wins
            losses = losses + 1 if data['environment_episode_data'][-1]['reward'][-1] < 0 else losses
            env.setRecordHistory()
            player3.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
            advisary.setRecordHistory(history_dict={'actions':[]})
            episode_done = True
            if i+1 >= episodes:
                obs = env.reset()
            else:
                obs = env.reset(config=mission_configuration[i+1])
            obs = PIL.Image.fromarray(obs)
            size = tuple((np.array(obs.size) * size_factor).astype(int))
            obs = np.array(obs.resize(size, PIL.Image.NEAREST))
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.fill((0, 0, 0))
            display.blit(surf, (0, 0))
            pygame.display.update()
            try:
                if not mission_configuration[i+1] == None:
                    show_score(display, wins, losses, mission_configuration[i+1]['steps'],txtX, txtY)#mission_configuration[i+1]['steps']
                else:
                    show_score(display, wins, losses, None, txtX, txtY)
            except IndexError:
                show_score(display, wins, losses, None, txtX, txtY)


#print("quitting?", player3.quit)
timestr = time.strftime("%Y%m%d-%H%M%S")
if write_data:
    pickle.dump(data,open(timestr + '_' + outputFileName + repr(participantNumber) + '.ptd','wb'))
pygame.quit()
