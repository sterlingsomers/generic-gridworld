#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time
from pyglet.window import key

import envs.generic_env
from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#
import pygame
import numpy as np
# from scipy.misc import imresize
import PIL
# import skimage

#a = Graph(depth=10)
env = envs.generic_env.GenericEnv(obstacles={'number':1},goals={'number':1})
#env = gym.make('Gridworld1-v1')

# if not hasattr(env.action_space, 'n'):
#     raise Exception('Keyboard agent only supports discrete action spaces')
# ACTIONS = env.action_space.n
# ROLLOUT_TIME = 1000
# SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
# # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

initial_image_data = env.reset()
initial_img = PIL.Image.fromarray(initial_image_data)
size = tuple((np.array(initial_img.size) * 10).astype(int))
initial_img = np.array(initial_img.resize(size, PIL.Image.NEAREST))
# initial_img = (np.array(initial_image_data.size).astype(int))#imresize(env.reset(), 1000, interp='nearest')
# initial_img = skimage.transform.resize(initial_img, (initial_img.shape[0] * 10, initial_img.shape[1] * 10), order=3)
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

while running:
    key_pressed = 0
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
                obs = PIL.Image.fromarray(obs)#imresize(obs, 1000, interp='nearest')
                size = tuple((np.array(obs.size) * 10).astype(int))
                obs = np.array(obs.resize(size, PIL.Image.NEAREST))
                surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
                display.blit(surf, (0, 0))
                game_done = False
                break

    if key_pressed and not game_done:
        obs, r, done, info = env.step(key_pressed)
        obs = PIL.Image.fromarray(obs)
        size = tuple((np.array(obs.size) * 10).astype(int))
        obs = np.array(obs.resize(size, PIL.Image.NEAREST))
        game_done = done
        print("reward", r)
        display.blit(background, (0,0))
        if done:
            #obs = env.reset()
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs), 0))
            display.blit(surf, (0,0))
        else:
            #pygame.surfarray.blit_array(background,obs)
            surf = pygame.surfarray.make_surface(np.flip(np.rot90(obs),0))
            #pygame.transform.rotate(surf,180)
            display.blit(surf, (0,0))


        pygame.time.delay(100)
        pygame.display.update()

pygame.quit()


print("done")