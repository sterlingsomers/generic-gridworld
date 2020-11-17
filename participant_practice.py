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

#some imports for the participant stuff
import glob, os

#A new class to extend old classes

# human_data = pickle.load(open('symbolic_data_sterlingV220200207-152603.lst','rb'))
os.chdir('./practice_data')
#determine the participant number
participant_files = []
for file in glob.glob('*.ptd'):
    participant_files.append(file)
participantNumber = len(participant_files)


data = {'environment_episode_data':[],'player_episode_data':[],'stuck':[],'AI_episode_data':[]}
episodes = 3
outputFileName = 'practice-'
write_data = True





# env = envs.generic_env.GenericEnv(map='small-empty',features=[{'entity_type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
env = envs.generic_env.GenericEnv(map='small-practice',wallrule=True)#,features=[{'entity_type':'obstacle','start_number':5,'color':'pink','moveTo':'moveToObstacle'}])
goal = CountingGoal(env,entity_type='goal',color='green',position='specific',position_coords=(5,6))
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
player3 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame,position='specific',position_coords=(8,1))
# player3 = ACTR(env, data=human_data, mismatch_penalty=20,noise=0.25,multiprocess=True,processes=5)
#player3 = TrainedAgent(env,color='aqua',model_name='net_vs_pred_best_noop')
#player4 = AIAgent(env,entity_type='agent',color='pink',pygame=pygame)
player4 = AIAgent(env,entity_type='agent',color='blue',obs_type='data',position='specific',position_coords=(5,1))
#advisary2 = ChasingBlockingAdvisary(env,entity_type='advisary',color='pink',obs_type='data')

env.setRecordHistory()
player3.setRecordHistory(history_dict={'actions':[],'saliences':[],'stuck':[]})
player4.setRecordHistory(history_dict={'actions':[]})



wins = 0
losses = 0
scoreboard_font_wins = pygame.font.Font('freesansbold.ttf',16)
scoreboard_font_losses = pygame.font.Font('freesansbold.ttf',16)
instructions_font = pygame.font.Font('freesansbold.ttf',14)
txtX = 30
txtY = 300
def show_score(display,wins,losses,x,y):

    wins_txt = scoreboard_font_wins.render("Wins: " + repr(wins), True, (255,255,255))
    losses_txt = scoreboard_font_losses.render("Losses: " + repr(losses), True, (255,255,255))
    instruction_txt = instructions_font.render('Arrive at (green) goal at the', True, (240, 252, 3))
    instruction_txt2 = instructions_font.render('same time as blue square.', True, (240, 252, 3))
    instruction_txt3 = instructions_font.render('Use arrow keys to move.', True, (240, 252, 3))
    instruction_txt4 = instructions_font.render('Use spacebar to issue non-move.', True, (240, 252, 3))
    display.blit(instruction_txt, (x,y+15+15+15))
    display.blit(instruction_txt2, (x, y +15 + 15 + 15+15))
    display.blit(instruction_txt3, (x, y + 15 + 15 + 15 + 15+15))
    display.blit(instruction_txt4, (x, y + 15 + 15 + 15 + 15+15+15))
    display.blit(wins_txt, (x,y))
    display.blit(losses_txt,(x,y+15))



human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

size_factor = 30

initial_image_data = env.reset()
initial_img = PIL.Image.fromarray(initial_image_data)
size = tuple((np.array(initial_img.size) * size_factor).astype(int))
initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))

initial_img = np.flip(np.rot90(initial_img),0)
#one noop
pygame.init()
window_size = (initial_img.shape[:2][0]+0, initial_img.shape[:2][1]+120)
display = pygame.display.set_mode(window_size,0,32)
background = pygame.surfarray.make_surface(initial_img)
background = background.convert()
display.blit(background,(0,0))
pygame.display.update()
show_score(display, wins, losses, txtX, txtY)
game_done = False
running = True

import math





done = False






clock = pygame.time.Clock()
#while not player3.quit:
for i in range(episodes):
    episode_done = False
    steps = 0
    while not episode_done:


        pygame.display.update()
        key_pressed = 0
        pygame.time.delay(0)

        obs, r, done, info = env.step([])
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1

        obs = PIL.Image.fromarray(obs)
        size = tuple((np.array(obs.size) * size_factor).astype(int))
        obs = np.array(obs.resize(size, PIL.Image.NEAREST))
        surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
        display.fill((0, 0, 0))
        display.blit(surf, (0, 0))
        pygame.display.update()
        show_score(display, wins, losses, txtX, txtY)
        for event in pygame.event.get():


            if event.type == pygame.QUIT:
                running = False


        # if key_pressed and not game_done:
        #     player3.action = key_pressed
        #     if done:
        #         obs = env.reset()
        #         surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
        #         display.fill((0, 0, 0))
        #         display.blit(surf, (0,0))
        #         show_score(display, wins, losses, txtX, txtY)
        #     else:
        #         #pygame.surfarray.blit_array(background,obs)
        #         surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs),0))
        #         #pygame.transform.rotate(surf,180)
        #         display.fill((0, 0, 0))
        #         display.blit(surf, (0,0))
        #         show_score(display, wins, losses, txtX, txtY)


        # pygame.time.delay(100)
        pygame.display.update()
        show_score(display,wins,losses,txtX, txtY)
        # clock.tick(100)
        if done:
            data['environment_episode_data'].append(env.history.copy())
            data['player_episode_data'].append(player3.history.copy())
            data['AI_episode_data'].append(player4.history.copy())
            env.setRecordHistory()
            player3.setRecordHistory(history_dict={'actions': [], 'saliences': [], 'stuck': []})
            player4.setRecordHistory(history_dict={'actions':[]})
            episode_done = True

            obs = env.reset()
            obs = PIL.Image.fromarray(obs)
            size = tuple((np.array(obs.size) * size_factor).astype(int))
            obs = np.array(obs.resize(size, PIL.Image.NEAREST))
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.fill((0, 0, 0))
            display.blit(surf, (0, 0))
            pygame.display.update()
            show_score(display, wins, losses, txtX, txtY)


#print("quitting?", player3.quit)
timestr = time.strftime("%Y%m%d-%H%M%S")
if write_data:
    pickle.dump(data,open(timestr + '_' + outputFileName + repr(participantNumber) + '.ptd','wb'))
pygame.quit()
