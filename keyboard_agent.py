#!/usr/bin/env python
from __future__ import print_function
from threading import Thread
import sys, gym, time
from pyglet.window import key

import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from envs.core import *



import pygame
import numpy as np

import PIL

#A new class to extend old classes
class AI_Agent(Agent):
    def __init__(self, env, obs_type='data',entity_type='', color='', position='random-free'):
        self.env = env
        self.value = env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type

    def getAction(self,obs):
        return random.choice([UP,DOWN,LEFT,RIGHT])

env = envs.generic_env.GenericEnv(dims=(20,20),features=[{'class':'feature','type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
player3 = HumanAgent(env,entity_type='agent',color='blue',pygame=pygame)
advisary = Advisary(env,entity_type='advisary',color='red')





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
    pygame.time.delay(250)

    obs, r, done, info = env.step()

    obs = PIL.Image.fromarray(obs)
    size = tuple((np.array(obs.size) * size_factor).astype(int))
    obs = np.array(obs.resize(size, PIL.Image.NEAREST))
    surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
    display.blit(surf, (0, 0))
    pygame.display.update()
    for event in pygame.event.get():


        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT: key_pressed = LEFT
            if event.key == pygame.K_RIGHT: key_pressed = RIGHT
            if event.key == pygame.K_DOWN: key_pressed = DOWN
            if event.key == pygame.K_UP: key_pressed = UP
            if event.key == pygame.K_r:
                key_pressed = True
                obs = env.reset()
                obs = PIL.Image.fromarray(obs)
                size = tuple((np.array(obs.size) * size_factor).astype(int))
                obs = np.array(obs.resize(size, PIL.Image.NEAREST))
                surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
                display.blit(surf, (0, 0))
                game_done = False
                break

    if key_pressed and not game_done:
        player3.action = key_pressed
        if done:
            #obs = env.reset()
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
        break

pygame.quit()


print("done")