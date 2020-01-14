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

class ChasingAdvisary(Advisary):

    def getAction(self,obs):
        my_location = np.where(self.env.current_grid_map == self.value)

        agents = self.getAgents()
        distance_to_agents = {}
        for agent in agents:
            agent_location = np.where(self.env.current_grid_map == agent)
            path  = self.env.getPathTo((my_location[0],my_location[1]),(agent_location[0],agent_location[1]),free_spaces=self.env.free_spaces)
            points_in_path = np.where(path == -1)
            points_in_path = list(zip(points_in_path[0],points_in_path[1]))
            distance_to_agents[agent] = {'dist':len(points_in_path),'path':path}
        #determine which is closest here.
        #for speed, just use the only agent
        path = distance_to_agents[4]['path']
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if path[self.env.action_map[direction]((my_location[0],my_location[1]))] == -1:
                return direction

class ChasingBlockingAdvisary(Advisary):
    def getAction(self, obs):
        my_location = np.where(self.env.current_grid_map == self.value)
        goal_val = self.env.getGoalValue()
        goal_location = np.where(self.env.current_grid_map == goal_val)
        agents = self.getAgents()
        distance_to_agents = {}
        agents_to_goal = {}
        for agent in agents:
            agent_location = np.where(self.env.current_grid_map == agent)
            path = self.env.getPathTo((my_location[0], my_location[1]), (agent_location[0], agent_location[1]),
                                      free_spaces=self.env.free_spaces)
            points_in_path = np.where(path == -1)
            if len(points_in_path) < 2:
                print("NOOP")
                return NOOP
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            agent_goal_transform = (int(agent_location[0])-int(goal_location[0]),int(agent_location[1]-goal_location[1]))
            distance_to_agents[agent] = {'dist': len(points_in_path), 'path': path, 'goal_transform':agent_goal_transform}

        #assume we've determined the closest agent
        #use #4 for now
        if distance_to_agents[agents[0]]['dist'] > 3:

            goal_transform = list(distance_to_agents[agents[0]]['goal_transform'])
            if goal_transform[0] < 0:
                goal_transform[0] = -1
            elif goal_transform[0] > 1:
                goal_transform[0] = 1
            if goal_transform[1] < 0:
                goal_transform[1] = -1
            elif goal_transform[1] > 1:
                goal_transform[1] = 1
            goal_location = [int(goal_location[0]), int(goal_location[1])]
            goal_location[0] = goal_location[0] + goal_transform[0]
            goal_location[1] = goal_location[1] + goal_transform[1]
            if int(my_location[0]) == int(goal_location[0]) and int(my_location[1]) == int(goal_location[1]):
                print("noooop")
                return NOOP
            path = self.env.getPathTo((my_location[0], my_location[1]), (goal_location[0], goal_location[1]),
                                      free_spaces=self.env.free_spaces)
            #if no path was found
            if not list(np.where(path == -1)[0]):
                print("NOOP 2")
                return NOOP
            for direction in [UP, DOWN, LEFT, RIGHT]:

                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    print("direction", direction)
                    return direction
        else:
            path = distance_to_agents[agents[0]]['path']
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    print("diection2", direction)
                    return direction
        return 2










class BlockingAdvisary(Advisary):
    def getGoal(self,obs):
        target_value = None
        for value in self.env.value_to_objects:
            if 'class' in self.env.value_to_objects[value]:
                if self.env.value_to_objects[value]['class'] == 'goal':
                    target_value = value
                    break
        target_location = np.where(self.env.current_grid_map == target_value)
        return target_location


    def getAction(self,obs):
        my_location = np.where(self.env.current_grid_map == self.value)
        agents = self.getAgents()

        distance_by_id = {}
        for agent_value in agents:
            target_location = np.where(self.env.current_grid_map == agent_value)
            distance_by_id[agent_value] = cityblock(my_location,target_location)
        for agent_value in distance_by_id:
            if distance_by_id[agent_value] <= 5:
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
        goal_location = self.getGoal(obs)
        permutation = (random.randint(-1,1),random.randint(0,1))
        target_location = (goal_location[0] + permutation[0], goal_location[1] + permutation[1])
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



env = envs.generic_env.GenericEnv(map='house',features=[{'class':'feature','type':'goal','start_number':1,'color':'green','moveTo':'moveToGoal'}])
# player1 = AI_Agent(env,obs_type='data',entity_type='agent',color='blue')
# player2 = Agent(env,entity_type='agent',color='orange')
player3 = HumanAgent(env,entity_type='agent',color='orange',pygame=pygame)
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data')





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