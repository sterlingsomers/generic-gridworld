import numpy as np
import random
import math
import functools
from threading import Lock
# from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from scipy.spatial.distance import cityblock, cdist
import PIL
import time
import os

import dill as pickle
from multiprocessing import Pool
from functools import partial

import ccm
log = ccm.log()
from ccm.lib.actr import *

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4
directions = [NOOP, DOWN, UP, LEFT, RIGHT]

class Entity:
    current_position = (0,0)
    action_chosen = None

    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free',position_coords=[]):
        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.position_coords = position_coords
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.position = position
        self.active = True
        self.history = {}
        self.record_history = False
        self.stuck = 0

    def setRecordHistory(self,on=True,history_dict={'actions':[],'agent_value':0,'stuck':[]},write_files=False,prefix=''):
        self.record_history = on
        self.history = history_dict
        self.history['agent_value'] = self.value

    def hitWall(self):
        return 0

    def stepCheck(self):
        return 0

    def moveToMe(self,entity_object):
        self.env.done = True
        self.env.reward -= 1

    def getAction(self,obs):
        if self.active:
            return random.choice([UP])
        else:
            return 0

    def getAgents(self):
        agents = []
        for entity in self.env.entities:
            if isinstance(self.env.entities[entity],Agent):
                agents.append(entity)
        return agents

    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        self.active = False
        return 1

    def place(self, position='random-free',position_coords=[]):
        # self.history['steps'] = []

        if position == 'random-free':
            free_spaces = []
            for free_space in self.env.free_spaces:
                found_spaces = np.where(self.env.current_grid_map == free_space)
                free_spaces.extend(list(zip(found_spaces[0], found_spaces[1])))
            the_space = random.choice(free_spaces)
            self.env.current_grid_map[the_space] = self.value
            self.current_position = the_space
        if position == 'near-goal':
            goal_value = self.env.getGoalValue()
            goal_locations = np.where(self.env.current_grid_map == goal_value)
            goal_locations = list(zip(goal_locations[0], goal_locations[1]))
            specific_goal_location = random.choice(goal_locations)
            neighbors = self.env.allNeighbors(specific_goal_location[0],specific_goal_location[1])
            free_spaces = []
            for free_space in self.env.free_spaces:
                found_spaces = np.where(self.env.current_grid_map == free_space)
                free_spaces.extend(list(zip(found_spaces[0], found_spaces[1])))
            intersection_of_spaces = [x for x in neighbors if x in free_spaces]
            if not intersection_of_spaces:
                self.place(position=position)
            the_space = random.choice(intersection_of_spaces)
            self.env.current_grid_map[the_space] = self.value
            self.current_position = the_space
        if position == 'specific':
            if not position_coords:
                raise ValueError("Coordinates must be specified")
            self.env.current_grid_map[position_coords] = self.value
            self.current_position = position_coords


    def update(self):
        pass

class ActiveEntity(Entity):
    def __init__(self, env, obs_type='image',entity_type='entity', color='', position='random-free',position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)
        self.env.active_entities[self.value] = self

    def getAction(self, obs):
        record_dict = self._getAction(obs)
        #action = self._getAction(obs)
        if self.record_history:
            for key,value in record_dict.items():
                self.history[key].append(value)
        return record_dict['actions']

class Goal(Entity):
    def __init__(self, env, obs_type='image',entity_type='goal', color='', position='random-free', position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)

    def moveToMe(self,entity_object):
        if isinstance(entity_object, Advisary):
            entity_object.intended_position = entity_object.current_position
            return 0
        print('entity hit goal', entity_object)
        self.env.done = True
        self.env.reward += 1

    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        self.env.done = True
        self.env.reward += 1

    def getAction(self,obs):
        return 0

class CountingGoal(Goal):
    def __init__(self, env, obs_type='image',entity_type='goal', color='', position='random-free', position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)
        self.env.active_entities[self.value] = self
        self.count = 0

    def stepCheck(self):
        if self.count == 1:
            self.env.done = True
            self.env.reward = - 1
            self.count = 0
        self.count = 0


    def moveToMe(self,entity_object):
        self.count += 1
        if self.count >= 2:
            self.env.done = True
            self.env.reward += 1

    def getAction(self,obs):
        return 0


class Agent(ActiveEntity):
    def __init__(self, env, obs_type='image',entity_type='agent', color='', position='random-free',position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)

    def moveToMe(self,entity_object):
        self.env.done = True
        self.env.reward -= 1


    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        # self.env.schedule_cleanup(self.value)
        # print('agent says done in moveto',self.value)
        # print('AGENT IS BEING ATTACKED')
        self.env.done = True
        self.env.reward = -1
        return 1

class AIAgent(Agent):
    def __init__(self, env, obs_type='image', entity_type='agent', color='', position='random-free',position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)


    def moveToMe(self,entity_object):
        print('enity', entity_object, 'hit', self)
        if isinstance(entity_object,Agent):
            entity_object.intended_position = entity_object.current_position
            return 1
        return super().moveToMe(entity_object)

    def getAction(self,obs):
        #go straight for the goal
        my_location = np.where(self.env.current_grid_map == self.value)
        goal_val = self.env.getGoalValue()
        goal_location = np.where(self.env.current_grid_map == goal_val)
        path = self.env.getPathTo((my_location[0], my_location[1]), (int(goal_location[0]), goal_location[1]),
                                  free_spaces=self.env.free_spaces)
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                #print("diection2", direction)
                return direction
        return random.choice([UP,DOWN,LEFT,RIGHT])

class ACTR_AGENT(Agent):
    import pyactup


    def __init__(self, env, obs_type='image',entity_type='agent', color='orange', position='random-free',data=[],mismatch_penalty=1,temperature=1,noise=0.0,decay=0.0,multiprocess=False,processes=4):
        super().__init__(env, obs_type, entity_type, color, position)
        self.mismatch_penalty = mismatch_penalty
        self.temperature = temperature
        self.noise = noise
        self.decay = decay
        self.memory = self.pyactup.Memory(noise=self.noise,decay=decay,temperature=temperature,threshold=-100.0,mismatch=mismatch_penalty,optimized_learning=False)
        self.pyactup.set_similarity_function(self.angle_similarity, *['goal_rads','advisary_rads','clockwise','counterclockwise'])
        self.pyactup.set_similarity_function(self.distance_similarity, *['goal_distance','advisary_distance'])
        # self.pyactup.set_similarity_function(self.vect_similarity, 'goal_vector')
        self.multiprocess = multiprocess
        self.processes = processes
        self.data = data
        self.last_observation = np.zeros(self.env.current_grid_map.shape)
        self.last_imagined_map = np.zeros(self.env.current_grid_map.shape)
        # print("here205")
        self.action_map = {1: lambda x: ((x[0] + 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           2: lambda x: ((x[0] - 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           3: lambda x: (x[0] % self.env.dims[0], (x[1] - 1) % self.env.dims[1]),
                           4: lambda x: (x[0] % self.env.dims[0], (x[1] + 1) % self.env.dims[1]),
                           0: lambda x: (x[0], x[1])}
        # print('here211')
        self.relative_action_categories = {UP:{'lateral':[LEFT,RIGHT],'away':DOWN},
                                           DOWN:{'lateral':[LEFT,RIGHT],'away':UP},
                                           LEFT:{'lateral':[UP,DOWN],'away':RIGHT},
                                           RIGHT:{'lateral':[UP,DOWN],'away':LEFT}}


        #Before using the distances, they have to be normalized (0 to 1)
        #Normalize by dividing by the max in the data
        distances = []
        for x in self.data:
            distances.append(x['goal_distance'])
            distances.append(x['advisary_distance'])
        #distances = [x['goal_distance'],x['advisary_distance'] for x in self.data]
        if distances:
            self.max_distance = max(distances)
            for datum in self.data:
                datum['goal_distance'] = datum['goal_distance'] / self.max_distance
                datum['advisary_distance'] = datum['advisary_distance'] / self.max_distance

        #for now, don't add any chunks at all
        # for chunk in self.data:
        #     self.memory.learn(**chunk)
    def vect_similarity(self,x,y):
        return 1

    def angle_similarity(self,x,y):
        PI = math.pi
        TAU = 2*PI
        result = min((2 * PI) - abs(x-y), abs(x-y))
        normalized = result / TAU
        xdeg = math.degrees(x)
        ydeg = math.degrees(y)
        resultdeg = math.degrees(result)
        normalized2 = resultdeg / 180
        #print("sim anle", 1 - normalized2)
        return 1 - normalized2

    def distance_similarity(self,x,y):
        x = x/self.max_distance
        result = 1 - abs(x-y)
        #print("sim distance", result, x, y)
        return result


    # def gridmap_to_symbols(self,gridmap, agent, value_to_objects):
    #     agent_location = np.where(gridmap == agent)
    #     agent_location = (int(agent_location[0]), int(agent_location[1]))
    #     goal_location = 0
    #     advisary_location = 0
    #     return_dict = {}
    #     for stuff in value_to_objects:
    #         if 'entity_type' in value_to_objects[stuff]:
    #             if value_to_objects[stuff]['entity_type'] == 'goal':
    #                 goal_location = np.where(gridmap == stuff)
    #             if value_to_objects[stuff]['entity_type'] == 'advisary':
    #                 advisary_location = np.where(gridmap == stuff)
    #     if goal_location:
    #         goal_location = (int(goal_location[0]), int(goal_location[1]))
    #         goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])
    #         path_agent_to_goal = self.env.getPathTo(agent_location, goal_location, free_spaces=[0])
    #         points_in_path = np.where(path_agent_to_goal == -1)
    #         points_in_path = list(zip(points_in_path[0], points_in_path[1]))
    #         return_dict['goal_vect'] = np.array(points_in_path) - np.array([agent_location[0],agent_location[1]])
    #         return_dict['goal_distance'] = len(points_in_path) / self.max_distance
    #     if advisary_location:
    #         advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
    #         advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
    #                                    advisary_location[1] - agent_lotight cation[1])
    #         path_agent_to_advisary = self.env.getPathTo(agent_location, advisary_location, free_spaces=[0])
    #         points_in_path = np.where(path_agent_to_advisary == -1)
    #         points_in_path = list(zip(points_in_path[0], points_in_path[1]))
    #         return_dict['advisary_rads'] = advisary_rads
    #         return_dict['advisary_distance'] = len(points_in_path) / self.max_distance
    #
    #     # the distances need to be normalized


        # return return_dict

    def multiprocess_blend_salience(self,probe_chunk,action):
        this_history = []
        # probe_chunk = self.gridmap_to_symbols(self.env.current_grid_map.copy(), self.value, self.env.value_to_objects)
        self.memory.activation_history = this_history
        blend_value = self.memory.blend(action, **probe_chunk)
        return action * 2

    def multi_blends(self,slot,probe,memory,value_to_objects):
        activation_history = []
        memory.activation_history = activation_history
        blend = memory.blend(slot,**probe)
        salience = self.compute_S(probe, [x for x in list(probe.keys()) if not x == slot],
                                  activation_history,
                                  slot,
                                  self.mismatch_penalty,
                                  self.temperature)
        return (blend, salience)


    def compute_S(self,probe, feature_list, history, Vk, MP, t):
        chunk_names = []

        PjxdSims = {}
        PI = math.pi
        for feature in feature_list:
            Fk = probe[feature]
            for chunk in history:
                dSim = None
                vjk = None
                for attribute in chunk['attributes']:
                    if attribute[0] == feature:
                        vjk = attribute[1]
                        break

                if Fk == vjk:
                    dSim = 0.0
                else:
                    if 'rads' in feature:
                        a_result = np.argmin(((2 * PI) - abs(vjk-Fk), abs(vjk-Fk)))
                        if not a_result:
                            dSim = (vjk - Fk) / abs(Fk - vjk)
                        else:
                            dSim = (Fk - vjk) / abs(Fk - vjk)
                    else:
                        dSim = (vjk - Fk) / abs(Fk - vjk)

                # if Fk == vjk:
                #     dSim = 0
                # else:
                #     dSim = -1 * ((Fk-vjk) / math.sqrt((Fk - vjk)**2))

                Pj = chunk['retrieval_probability']
                if not feature in PjxdSims:
                    PjxdSims[feature] = []
                PjxdSims[feature].append(Pj * dSim)
                pass

        # vio is the value of the output slot
        fullsum = {}
        result = {}  # dictionary to track feature
        Fk = None
        for feature in feature_list:
            Fk = probe[feature]
            if not feature in fullsum:
                fullsum[feature] = []
            inner_quantity = None
            Pi = None
            vio = None
            dSim = None
            vik = None
            for chunk in history:
                Pi = chunk['retrieval_probability']
                for attribute in chunk['attributes']:
                    if attribute[0] == Vk:
                        vio = attribute[1]

                for attribute in chunk['attributes']:
                    if attribute[0] == feature:
                        vik = attribute[1]
                # if Fk > vik:
                #     dSim = -1
                # elif Fk == vik:
                #     dSim = 0
                # else:
                #     dSim = 1
                # dSim = (Fk - vjk) / sqrt(((Fk - vjk) ** 2) + 10 ** -10)
                if Fk == vik:
                    dSim = 0.0
                else:
                    #dSim = (vik - Fk) / abs(Fk - vik)
                    if 'rads' in feature:
                        a_result = np.argmin(((2 * PI) - abs(vjk-Fk), abs(vjk-Fk)))
                        if not a_result:
                            dSim = (vik - Fk) / abs(Fk - vik)
                        else:
                            dSim = (Fk - vjk) / abs(Fk - vik)
                    else:
                        dSim = (vik - Fk) / abs(Fk - vik)
                #
                # if Fk == vik:
                #     dSim = 0
                # else:
                #     dSim = -1 * ((Fk-vik) / math.sqrt((Fk - vik)**2))

                inner_quantity = dSim - sum(PjxdSims[feature])
                fullsum[feature].append(Pi * inner_quantity * vio)

            result[feature] = sum(fullsum[feature])

        # sorted_results = sorted(result.items(), key=lambda kv: kv[1])
        return result

    def getPathTo(self,map,start_location,end_location, free_spaces=[], exclusion_points=[]):
        '''An A* algorithm to get from one point to another.
        free_spaces is a list of values that can be traversed.
        start_location and end_location are tuple point values.

        Returns a map with path denoted by -1 values. Inteded to use np.where(path == -1).'''
        pathArray = np.full(map.shape,0)

        for free_space in free_spaces:
            zeros = np.where(map == free_space)
            zeros = list(zip(zeros[0],zeros[1]))
            for point in zeros:
                pathArray[point] = 1

        #Because we started with true (1), we start with a current value of 1 (which will increase to two)
        current_value = 1
        target_value = 0
        pathArray[start_location] = 2
        directions = [UP, DOWN, LEFT, RIGHT]
        random.shuffle(directions)
        stop = False
        while True:
            current_value += 1
            target_value = current_value + 1
            test_points = np.where(pathArray == current_value)
            test_points = list(zip(test_points[0],test_points[1]))
            random.shuffle(test_points)
            still_looking = False
            for test_point in test_points:
                for direction in directions:
                    if self.action_map[direction](test_point) in exclusion_points:
                        continue
                    if pathArray[self.action_map[direction](test_point)] and pathArray[self.action_map[direction](test_point)] + current_value <= target_value:
                        pathArray[self.action_map[direction](test_point)] = target_value
                        still_looking = True
                    # if not end_location[0].tolist():
                    #     print('emtpiness')
                    # print(self.action_map[direction](test_point), end_location)
                    # print(test_point, end_location, direction)
                    # try:
                    #exclusion points
                    if self.action_map[direction](test_point) in exclusion_points:
                        continue

                    if self.action_map[direction](test_point) == (int(end_location[0]),int(end_location[1])):
                        pathArray[end_location] = - 1
                        still_looking = True
                        stop = True
                        break
                    # except Exception:
                    #     print('ERROR')

            if not still_looking:
                return pathArray
            if stop:
                break
        current_point = end_location
        while True:
            for direction in directions:
                if pathArray[self.action_map[direction](current_point)] == target_value - 1:
                    pathArray[current_point] = -1
                    current_point = self.action_map[direction](current_point)
                    target_value -= 1
                if current_point == start_location:
                    # pathArray[current_point] = -1
                    return pathArray

    def gridmap_to_symbols(self, gridmap, agent, value_to_objects):
        action_map = {1: lambda x: ((x[0] + 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      2: lambda x: ((x[0] - 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      3: lambda x: (x[0] % gridmap.shape[0], (x[1] - 1) % gridmap.shape[1]),
                      4: lambda x: (x[0] % gridmap.shape[0], (x[1] + 1) % gridmap.shape[1]),
                      0: lambda x: (x[0], x[1])}

        #agent_location = np.where(gridmap == agent)
        agent_location = self.env.active_entities[agent].current_position#(int(agent_location[0]), int(agent_location[1]))
        goal_location = []
        advisary_location = []
        return_dict = {}
        for stuff in value_to_objects:
            if 'entity_type' in value_to_objects[stuff]:
                if value_to_objects[stuff]['entity_type'] == 'goal':
                    goal_location = np.where(gridmap == stuff)
                if value_to_objects[stuff]['entity_type'] == 'advisary':
                    advisary_location = np.where(gridmap == stuff)
        if goal_location:
            goal_location = (int(goal_location[0]), int(goal_location[1]))
            # goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])
            path_agent_to_goal = self.getPathTo(gridmap, agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))

            # goal_vector = np.array(list(goal_location)) - np.array([agent_location[0],agent_location[1]])
            # goal_unit_vector = goal_vector / np.linalg.norm(goal_vector)
            # return_dict['goal_vector'] = (goal_unit_vector[0], goal_unit_vector[1])
            # return_dict['goal_vector'] = tuple((np.array(goal_location) - np.array([agent_location[0],agent_location[1]])) / np.linalg.norm(np.array(goal_location) - np.array([agent_location[0],agent_location[1]])))
            return_dict['goal_rads'] = 0.0
            return_dict['goal_distance'] = len(points_in_path)
            # print('here500')
        if advisary_location:#and goal_location
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            # advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
            #                            advisary_location[1] - agent_location[1])
            path_agent_to_advisary = self.getPathTo(gridmap, agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            # return_dict['advisary_rads'] = advisary_rads
            return_dict['advisary_distance'] = len(points_in_path)
            p1 = [agent_location[0],agent_location[1]]
            p0 = [goal_location[0], goal_location[1]]
            p2 = advisary_location
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) #in radians
            # angle = np.degrees(angle)
            return_dict['advisary_rads'] = angle
        # print('here517')
        for wall, dir in {'up_wall_distance': 2, 'left_wall_distance': 3, 'down_wall_distance': 1,
                          'right_wall_distance': 4}.items():
            current_position = agent_location
            dist = 0
            while True:
                dist += 1
                current_position = action_map[dir](current_position)
                if gridmap[current_position] == 1:
                    return_dict[wall] = dist
                    break

        # the distances need to be normalized (after return)
        # print('here530')
        return return_dict

    def determine_action_chunk(self,old_obs,new_obs):
        chunk = {}
        last_predator = np.where(old_obs == 4)
        last_predator = (int(last_predator[0]), int(last_predator[1]))
        new_predator = np.where(new_obs == 4)
        new_predator = (int(new_predator[0]), int(new_predator[1]))

        last_action_category = None

        my_old_position = np.where(old_obs == 3)
        my_old_position = (int(my_old_position[0]), int(my_old_position[1]))

        my_new_position = self.current_position

        old_goal = np.where(old_obs == 2)
        old_pred_to_goal_path = self.getPathTo(old_obs, last_predator, old_goal, free_spaces=[0])
        old_points_in_path_pred_to_goal = np.where(old_pred_to_goal_path == -1)
        old_points_in_path_pred_to_goal = list(
            zip(old_points_in_path_pred_to_goal[0], old_points_in_path_pred_to_goal[1]))
        old_distance_pred_to_goal = len(old_points_in_path_pred_to_goal)

        old_pred_to_me_path = self.getPathTo(old_obs, last_predator, my_old_position, free_spaces=[0])
        points_in_old_pred_to_me_path = np.where(old_pred_to_me_path == -1)
        points_in_old_pred_to_me_path = list(zip(points_in_old_pred_to_me_path[0], points_in_old_pred_to_me_path[1]))
        old_distance_pred_to_me = len(points_in_old_pred_to_me_path)

        new_pred_to_me_path = self.getPathTo(new_obs, new_predator, my_new_position)
        new_points_in_pred_to_me_path = np.where(new_pred_to_me_path == -1)
        new_points_in_pred_to_me_path = list(zip(new_points_in_pred_to_me_path[0], new_points_in_pred_to_me_path[1]))
        new_distance_pred_to_me = len(new_points_in_pred_to_me_path)

        new_goal = np.where(new_obs == 2)
        new_pred_to_goal_path = self.getPathTo(new_obs, new_predator, new_goal)
        new_points_in_pred_to_goal_path = np.where(new_pred_to_goal_path == -1)
        new_points_in_pred_to_goal_path = list(
            zip(new_points_in_pred_to_goal_path[0], new_points_in_pred_to_goal_path[1]))
        new_distance_pred_to_goal = len(new_points_in_pred_to_goal_path)

        chunk['attack'] = 1  # attack = 1, block = 0
        # the following checks to see if it stayed next to the goal
        # we are classifying that as not-attack (attack = 0)
        goal_position = [int(new_goal[0]), int(new_goal[1])]
        predator_position = (int(new_predator[0]), int(new_predator[1]))
        X = 10
        Y = 10  # size of board...
        neighbors = lambda x, y: [(x2, y2) for x2 in range(x - 1, x + 2)
                                  for y2 in range(y - 1, y + 2)
                                  if (-1 < x <= X and
                                      -1 < y <= Y and
                                      (x != x2 or y != y2) and
                                      (0 <= x2 <= X) and
                                      (0 <= y2 <= Y))]
        neighs = neighbors(goal_position[0], goal_position[1])
        if predator_position in neighs:
            chunk['attack'] = 0

        p1 = [my_old_position[0], my_old_position[1]]
        p0 = [int(old_goal[0]), int(old_goal[1])]
        p2 = [last_predator[0], last_predator[1]]
        p3 = [new_predator[0], new_predator[1]]
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        angle1 = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        # angle2 = np.math.atan2(np.linalg.det([v0,v2]),np.dot(v0,v2))
        # diffangle = angle2 - angle1
        chunk['clockwise'] = angle1 if angle1 < 0 else 0
        chunk['counterclockwise'] = angle1 if angle1 > 0 else 0
        chunk['direct'] = 1 if angle1 == 0 else 0

        return chunk

    def whatmove(self,gridmap,me,goal,predator,desired_rads):
        if desired_rads == 0:
            return 0
        action_map = {1: lambda x: ((x[0] + 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      2: lambda x: ((x[0] - 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      3: lambda x: (x[0] % gridmap.shape[0], (x[1] - 1) % gridmap.shape[1]),
                      4: lambda x: (x[0] % gridmap.shape[0], (x[1] + 1) % gridmap.shape[1]),
                      0: lambda x: (x[0], x[1])}
        move_to_rads = {UP:0, DOWN:0, LEFT:0, RIGHT:0}
        v0 = np.array(goal) - np.array(me)
        v1 = np.array(predator) - np.array(me)

        for move_func in action_map:
            new_pred = action_map[move_func](predator)
            v2 = np.array(new_pred) - np.array(me)
            move_to_rads[move_func] = np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))

        sorted_move = sorted(move_to_rads.items(), key=lambda kv: abs(kv[1]-desired_rads))

        return sorted_move[0][0]





    def _getAction(self,obs):
        # print('actr action')
        #The following creates chunks that describe the transition from the previous time-step
        #to the current time step.
        #The intention is to use this to predict where the agent will be on the NEXT time-step
        chunk = {}
        advisary_action_map = {'advisary_up': UP, 'advisary_down': DOWN, 'advisary_noop': NOOP, 'advisary_left': LEFT,
                               'advisary_right': RIGHT}
        goal_neighbours = [(1,0),(1,1),(0,1),(0,-1),(-1,0),(-1,-1),(-1,1),(1,-1)]
        encode_chunk = True #used to store only when prediction was wrong


        p1, p2, p3 = None, None, None
        my_new_position = self.current_position
        new_predator = np.where(self.env.current_grid_map == 4)
        new_predator = (int(new_predator[0]), int(new_predator[1]))

        if self.last_observation.any():
            if self.last_imagined_map.any():
                imagined_predator = np.where(self.last_imagined_map == 4)
                imagined_predator_points = list(zip(imagined_predator[0], imagined_predator[1]))
                for points in imagined_predator_points:
                    if self.env.current_grid_map[points] == 4:
                        encode_chunk = False #if at least one of the predictions was right, no need to store
            #The chunk encodes what the state was at time t-1, and what the predator did, observed at time t
            chunk = self.determine_action_chunk(self.last_observation,self.env.current_grid_map)
            old_goal = np.where(self.last_observation == 2)
            my_old_position = np.where(self.last_observation == 3)
            my_old_position = (int(my_old_position[0]), int(my_old_position[1]))
            p0 = [int(old_goal[0]), int(old_goal[1])]
            p1 = [my_new_position[0], my_new_position[1]]
            p3 = [new_predator[0], new_predator[1]]




            obs_chunk = self.gridmap_to_symbols(self.last_observation.copy(), self.value, self.env.value_to_objects)
            for key in obs_chunk:
                if 'distance' in key:
                    obs_chunk[key] = obs_chunk[key] / self.max_distance

            chunk = {**obs_chunk, **chunk}




        else:
            self.last_observation = self.env.current_grid_map.copy()
            #observation = self.gridmap_to_symbols(self.env.current_grid_map, self.value, self.env.value_to_objects)

        self.last_observation = self.env.current_grid_map.copy()

        self.memory.activation_history = []

        if chunk:
            encode_chunk = True
            if encode_chunk:
                print('encoding 1' , chunk)
                self.memory.learn(**chunk)
        # if chunk:
        #     encode_chunk = True #encode everything
        #     if encode_chunk:
        #         if not chunk['advisary_noop']: #combined with encode_chunk = True - encodes everything except NOOP
        #             self.memory.learn(**chunk)

        #made the observation, now do the blend
        if not self.multiprocess:
            self.memory.advance(0.1)
            # blend_values = {}
            # probe_chunk = self.gridmap_to_symbols(self.env.current_grid_map.copy(), self.value,
            #                                       self.env.value_to_objects)
            #
            # for advisary_action in ['advisary_up', 'advisary_down', 'advisary_noop', 'advisary_left','advisary_right']:
            #     blend_values[advisary_action] = self.memory.blend(advisary_action, **probe_chunk)

            print('Current map. value', self.value)
            print(np.array2string(self.env.current_grid_map))

            #will the predator attack?
            probe_chunk = self.gridmap_to_symbols(self.env.current_grid_map.copy(), self.value, self.env.value_to_objects)
            for key in probe_chunk:
                if 'distance' in key:
                    probe_chunk[key] = probe_chunk[key] / self.max_distance
            attack_value = self.memory.blend('attack', **probe_chunk)

            probe_chunk['attack'] = attack_value

            #now, given the attack status, predict the movement
            for direction in ['direct', 'clockwise', 'counterclockwise']:
                probe_chunk[direction] = self.memory.blend(direction, **probe_chunk)




            #make a pretend map, modify it with the predicted movement of the agent
            imaginary_map = self.env.current_grid_map.copy()
            #but you can only update the pretend map, if you have any basis to predict
            if not probe_chunk['attack'] == None:

                threshold = 0.33
                if probe_chunk['attack'] == 0 and probe_chunk['direct']:
                    #noop
                    pass
                elif probe_chunk['attack'] == 0 and probe_chunk['direct'] == 0:
                    #what move will cause the predicted clockwise or counterclockwise motion?
                    clock_move = self.whatmove(self.env.current_grid_map,p1, p0, p3, probe_chunk['clockwise'])
                    counter_clock_move = self.whatmove(self.env.current_grid_map,p1, p0, p3, probe_chunk['counterclockwise'])





            print("Projected Map")
            print("assuming:", probe_chunk)
            print(np.array2string(imaginary_map))


            #Should I go towards the goal? G1
            goal_position = np.where(self.env.current_grid_map == 2)
            goal_position = list(zip(goal_position[0], goal_position[1]))
            # #I'll allow the A* to put a path through the predator, then check if the action produced would crash
            # #which we could have just ensured it didn't by excluding 4 in the free_spaces
            # path_to_goal = self.getPathTo(imaginary_map, (self.current_position[0], self.current_position[1]),
            #                               goal_position[0], free_spaces=[0,4],exclusion_points=[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)])
            # points_in_path = np.where(path_to_goal == -1)
            # points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            # direction_to_goal = -1
            # potential_action = -1
            # for direction in directions:
            #     if path_to_goal[self.env.action_map[direction]((self.current_position[0], self.current_position[1]))] == -1:
            #         # print("diection2", direction)
            #         print(np.array2string(path_to_goal))
            #         potential_action = direction
            #         direction_to_goal = direction
            # #G1.a
            # #Will that action crash me into where I think the predator will be?
            # #This is what the A* could have done
            # if imaginary_map[self.action_map][potential_action](self.current_position) == 4:
            #     print('find another action')
            #     potential_action = -1
            #
            # #G2 - Should I go lateral?
            # #we assume that athe direction
            # if potential_action == -1:
            #     pass

            ##### an alternate approach
            #which actions bring me towards?
            current_position = self.current_position
            current_distance = np.linalg.norm(np.array(current_position) - np.array(goal_position[0]))
            direction_distances = {}
            for direction in directions:
                new_position = self.action_map[direction](current_position)
                if self.env.current_grid_map[new_position] == 1: #if i hit a wall, no move
                    new_position = self.current_position
                new_distance = np.linalg.norm(np.array(new_position) - np.array(goal_position[0]))
                direction_distances[direction] = new_distance

            # closers = [k for k,v in direction_distances.items() if v < current_distance]
            # closers.sort()
            #don't do that anymore. just sort by distance. then let the imagined position
            #figure it out
            closers = sorted(direction_distances, key=direction_distances.get)
            #closers is now an ordered list of actions that bring the agent closer to the target
            #Try them in order, to see if it will crash.  First one that doesn't crash the agent (given prediction)
            for action in closers:
                #if the imagined position is my position, evade
                new_predator = np.where(imaginary_map == 4)
                new_predator = list(zip(new_predator[0], new_predator[1]))#(int(new_predator[0]), int(new_predator[1]))
                # if new_predator == self.current_position:
                #     break #go on to the evade optoins
                #That break is no longer needed.
                position_function = self.action_map[action]
                intended_agent_position = position_function(self.current_position)
                for pred in new_predator:
                    ##Does the intended position get me killed?
                    #Do we cross paths
                    if self.current_position == pred and intended_agent_position == self.env.active_entities[4].current_position:
                        continue
                    #Will I go to its projected position
                    if not imaginary_map[intended_agent_position] == 4 and not imaginary_map[intended_agent_position] == 1:
                        print('action:', action)
                        return {'actions':action}


            #IF I get this far, that means I have to try to evade
            # evades = [k for k,v in direction_distances.items()]
            # evades.sort()
            # for action in evades:
            #     position_function = self.action_map[action]
            #     intended_agent_position = position_function(self.current_position)
            #     if self.env.current_grid_map[intended_agent_position] == 1: #if i hit a wall, no move
            #         intended_agent_position = self.current_position
            #     #Does the intended position get me killed?
            #     if not imaginary_map[intended_agent_position] == 4:
            #         return {'actions':action}
            # print('here')
            #YOU don't need evades.  in closers, just pick the action that minimizes that distances, and desn't crash



        else:
            probe_chunk = self.gridmap_to_symbols(self.env.current_grid_map.copy(), self.value,
                                                  self.env.value_to_objects)
            vto = self.env.value_to_objects.copy()
            env = self.env
            self.env = None
            while True:
                try:

                    p = Pool(processes=self.processes)
                    multi_p = partial(self.multi_blends, memory=self.memory, probe=probe_chunk,value_to_objects=vto)
                    BS = p.map(multi_p, possible_actions)
                    p.close()
                    p.join()
                    blends = [x[0] for x in BS]
                    saliences = {action:salience for action,salience in zip(possible_actions,[b[1] for b in BS])}
                    self.env = env
                except BlockingIOError:
                    print("Blocking IO Error")
                    time.sleep(0.1)
                    continue
                break





        # for x,y in zip(possible_actions, blends):
        #     print(x,y)
        # if possible_actions[np.argmax(blends)] == 'noop':
        # print('argmax', blends, np.argmax(blends), possible_actions[np.argmax(blends)])
        # print(saliences[possible_actions[np.argmax(blends)]])

        argmax_action = possible_actions[np.argmax(blends)]
        action_value = eval(argmax_action.upper())

        return {'actions':round(action_value),'saliences':0,'stuck':self.stuck}


    def moveToMe(self,entity_object):
        print('enity', entity_object, 'hit', self)
        if isinstance(entity_object,Advisary):
            entity_object.intended_position = self.current_position
            self.intended_position =  self.current_position
        #In this game, if anything moves to me, the game must be over
        self.env.done = True
        self.env.reward = - 1
            # return 1
        # return super().moveToMe(entity_object)

class pythonACTR(Agent):

    import pyactup
    class EnvironmentWrapper(ccm.Model):
        #forget the visi
        goal = ccm.Model(isa='goal',rads=0,distance=0)
        adversary = ccm.Model(isa='adversary',rads=0,distance=0)


    class VisionModule(ccm.Model):
        pass

    class ACTUP_AGENT(ACTR):
        fact1 = {'up': 1, 'down': 0}
        fact2 = {'up': 0, 'down': 1}
        fact3 = {'up': 0.8, 'down': 0.1}
        fact4 = {'up': 0.2, 'down': 1}
        facts = [fact1, fact2, fact3, fact4]
        focus = Buffer()
        DMbuffer = Buffer()
        DM = ACTUP(DMbuffer, noise=0.0, temperature=1, learning_time_increment=0, retrieval_time_increment=0.0,
                   mismatch=5, decay=0.5, threshold=-10.00)
        # pyactup.set_similarity_function(up_similarity, 'up')
        # pyactup.set_similarity_function(up_similarity, 'down')

        def init():
            for afact in facts:
                DM.learn(**afact)
            # DM.advance()
            focus.set('step:go_towards')

        #we don't need a vision module. for simplicity, just assume the environment is visible
        def do_nothing(focus='step:go_towards'):
            print('do nothing', self.parent)
            self.external_reference.action_to_forward = 1
            focus.set('step:request')

        def request(focus='step:request'):
            DM.blend('down', up=1.0)
            focus.set('step:do_nothing')

    def __init__(self, env, obs_type='image',entity_type='agent', color='orange', position='random-free',data=[],mismatch_penalty=1,temperature=1,noise=0.0,decay=0.0,multiprocess=False,processes=4):
        super().__init__(env, obs_type, entity_type, color, position)
        self.mismatch_penalty = mismatch_penalty
        self.temperature = temperature
        self.noise = noise
        self.decay = decay
        self.memory = self.pyactup.Memory(noise=self.noise,decay=decay,temperature=temperature,threshold=-100.0,mismatch=mismatch_penalty,optimized_learning=False)
        # self.pyactup.set_similarity_function(self.angle_similarity, *['goal_rads','advisary_rads','clockwise','counterclockwise'])
        # self.pyactup.set_similarity_function(self.distance_similarity, *['goal_distance','advisary_distance'])
        # self.pyactup.set_similarity_function(self.vect_similarity, 'goal_vector')
        self.multiprocess = multiprocess
        self.processes = processes
        self.data = data
        self.last_observation = np.zeros(self.env.current_grid_map.shape)
        self.last_imagined_map = np.zeros(self.env.current_grid_map.shape)
        # print("here205")
        self.action_map = {1: lambda x: ((x[0] + 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           2: lambda x: ((x[0] - 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           3: lambda x: (x[0] % self.env.dims[0], (x[1] - 1) % self.env.dims[1]),
                           4: lambda x: (x[0] % self.env.dims[0], (x[1] + 1) % self.env.dims[1]),
                           0: lambda x: (x[0], x[1])}
        # print('here211')
        self.relative_action_categories = {UP:{'lateral':[LEFT,RIGHT],'away':DOWN},
                                           DOWN:{'lateral':[LEFT,RIGHT],'away':UP},
                                           LEFT:{'lateral':[UP,DOWN],'away':RIGHT},
                                           RIGHT:{'lateral':[UP,DOWN],'away':LEFT}}

        self.action_to_forward = 0


        #Before using the distances, they have to be normalized (0 to 1)
        #Normalize by dividing by the max in the data
        distances = []
        for x in self.data:
            distances.append(x['goal_distance'])
            distances.append(x['advisary_distance'])
        #distances = [x['goal_distance'],x['advisary_distance'] for x in self.data]
        if distances:
            self.max_distance = max(distances)
            for datum in self.data:
                datum['goal_distance'] = datum['goal_distance'] / self.max_distance
                datum['advisary_distance'] = datum['advisary_distance'] / self.max_distance

        self.production_system = self.ACTUP_AGENT()
        self.production_system.external_environment = self.env
        self.production_system.external_reference = self
        self.production_environment = self.EnvironmentWrapper()
        self.production_environment.agent = self.production_system
        ccm.log_everything(self.production_system)

    def getPathTo(self,map,start_location,end_location, free_spaces=[], exclusion_points=[]):
        '''An A* algorithm to get from one point to another.
        free_spaces is a list of values that can be traversed.
        start_location and end_location are tuple point values.

        Returns a map with path denoted by -1 values. Inteded to use np.where(path == -1).'''
        pathArray = np.full(map.shape,0)

        for free_space in free_spaces:
            zeros = np.where(map == free_space)
            zeros = list(zip(zeros[0],zeros[1]))
            for point in zeros:
                pathArray[point] = 1

        #Because we started with true (1), we start with a current value of 1 (which will increase to two)
        current_value = 1
        target_value = 0
        pathArray[start_location] = 2
        directions = [UP, DOWN, LEFT, RIGHT]
        random.shuffle(directions)
        stop = False
        while True:
            current_value += 1
            target_value = current_value + 1
            test_points = np.where(pathArray == current_value)
            test_points = list(zip(test_points[0],test_points[1]))
            random.shuffle(test_points)
            still_looking = False
            for test_point in test_points:
                for direction in directions:
                    if self.action_map[direction](test_point) in exclusion_points:
                        continue
                    if pathArray[self.action_map[direction](test_point)] and pathArray[self.action_map[direction](test_point)] + current_value <= target_value:
                        pathArray[self.action_map[direction](test_point)] = target_value
                        still_looking = True
                    # if not end_location[0].tolist():
                    #     print('emtpiness')
                    # print(self.action_map[direction](test_point), end_location)
                    # print(test_point, end_location, direction)
                    # try:
                    #exclusion points
                    if self.action_map[direction](test_point) in exclusion_points:
                        continue

                    if self.action_map[direction](test_point) == (int(end_location[0]),int(end_location[1])):
                        pathArray[end_location] = - 1
                        still_looking = True
                        stop = True
                        break
                    # except Exception:
                    #     print('ERROR')

            if not still_looking:
                return pathArray
            if stop:
                break
        current_point = end_location
        while True:
            for direction in directions:
                if pathArray[self.action_map[direction](current_point)] == target_value - 1:
                    pathArray[current_point] = -1
                    current_point = self.action_map[direction](current_point)
                    target_value -= 1
                if current_point == start_location:
                    # pathArray[current_point] = -1
                    return pathArray

    def gridmap_to_symbols(self, gridmap, agent, value_to_objects):
        action_map = {1: lambda x: ((x[0] + 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      2: lambda x: ((x[0] - 1) % gridmap.shape[0], (x[1]) % gridmap.shape[1]),
                      3: lambda x: (x[0] % gridmap.shape[0], (x[1] - 1) % gridmap.shape[1]),
                      4: lambda x: (x[0] % gridmap.shape[0], (x[1] + 1) % gridmap.shape[1]),
                      0: lambda x: (x[0], x[1])}

        #agent_location = np.where(gridmap == agent)
        agent_location = self.env.active_entities[agent].current_position#(int(agent_location[0]), int(agent_location[1]))
        goal_location = []
        advisary_location = []
        return_dict = {}
        for stuff in value_to_objects:
            if 'entity_type' in value_to_objects[stuff]:
                if value_to_objects[stuff]['entity_type'] == 'goal':
                    goal_location = np.where(gridmap == stuff)
                if value_to_objects[stuff]['entity_type'] == 'advisary':
                    advisary_location = np.where(gridmap == stuff)
        if goal_location:
            goal_location = (int(goal_location[0]), int(goal_location[1]))
            # goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])
            path_agent_to_goal = self.getPathTo(gridmap, agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))

            # goal_vector = np.array(list(goal_location)) - np.array([agent_location[0],agent_location[1]])
            # goal_unit_vector = goal_vector / np.linalg.norm(goal_vector)
            # return_dict['goal_vector'] = (goal_unit_vector[0], goal_unit_vector[1])
            # return_dict['goal_vector'] = tuple((np.array(goal_location) - np.array([agent_location[0],agent_location[1]])) / np.linalg.norm(np.array(goal_location) - np.array([agent_location[0],agent_location[1]])))
            return_dict['goal_rads'] = 0.0
            return_dict['goal_distance'] = len(points_in_path)
            # print('here500')
        if advisary_location:#and goal_location
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            # advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
            #                            advisary_location[1] - agent_location[1])
            path_agent_to_advisary = self.getPathTo(gridmap, agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            # return_dict['advisary_rads'] = advisary_rads
            return_dict['advisary_distance'] = len(points_in_path)
            p1 = [agent_location[0],agent_location[1]]
            p0 = [goal_location[0], goal_location[1]]
            p2 = advisary_location
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) #in radians
            # angle = np.degrees(angle)
            return_dict['advisary_rads'] = angle
        # print('here517')
        for wall, dir in {'up_wall_distance': 2, 'left_wall_distance': 3, 'down_wall_distance': 1,
                          'right_wall_distance': 4}.items():
            current_position = agent_location
            dist = 0
            while True:
                dist += 1
                current_position = action_map[dir](current_position)
                if gridmap[current_position] == 1:
                    return_dict[wall] = dist
                    break

        # the distances need to be normalized (after return)
        # print('here530')
        return return_dict

    def gridmap_to_symbols(self,gridmap, agent, value_to_objects):
        agent_location = np.where(gridmap == agent)
        agent_location = (int(agent_location[0]), int(agent_location[1]))
        goal_location = 0
        advisary_location = 0
        return_dict = {}
        for stuff in value_to_objects:
            if 'entity_type' in value_to_objects[stuff]:
                if value_to_objects[stuff]['entity_type'] == 'goal':
                    goal_location = np.where(gridmap == stuff)
                if value_to_objects[stuff]['entity_type'] == 'advisary':
                    advisary_location = np.where(gridmap == stuff)
        if goal_location:
            goal_location = (int(goal_location[0]), int(goal_location[1]))
            goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])
            path_agent_to_goal = self.env.getPathTo(agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['goal_rads'] = goal_rads
            return_dict['goal_distance'] = len(points_in_path) / self.max_distance
        if advisary_location:
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
                                       advisary_location[1] - agent_location[1])
            path_agent_to_advisary = self.env.getPathTo(agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['adversary_rads'] = advisary_rads
            return_dict['adversary_distance'] = len(points_in_path) / self.max_distance

        # the distances need to be normalized


        return return_dict

    def _getAction(self,obs):
        #Update the ennvironment self.production.parent.<property> = <value>
        #then run the production system for an amount of time self.production_environment.run(seconds.miliseconds)

        environment_dictionary = self.gridmap_to_symbols(self.env.current_grid_map.copy(), self.value,
                                                  self.env.value_to_objects)
        #there is no partial matching in productions in pythonACTR (yet!)
        #I therefore make the values categorical (left, right, up, down)
        goal_angle = math.degrees(environment_dictionary['goal_rads'])


        self.production_system.parent.goal.rads = environment_dictionary['goal_rads']
        self.production_system.parent.goal.distance = environment_dictionary['goal_distance']
        self.production_system.parent.adversary.rads = environment_dictionary['adversary_rads']
        self.production_system.parent.adversary.distance = environment_dictionary['adversary_distance']
        self.production_environment.run(0.100)
        return {'actions':self.action_to_forward,'saliences':0,'stuck':self.stuck}




class HumanAgent(Agent):
    obs = None
    def __init__(self, env, obs_type='image',entity_type='agent', color='', position='random-free',pygame='None'):
        super().__init__(env, obs_type, entity_type, color, position)
        self.pygame = pygame
        self.quit = False

    def moveToMe(self,entity_object):
        print('enity', entity_object, 'hit', self)
        if isinstance(entity_object,Agent):
            entity_object.intended_position = entity_object.current_position
            self.intended_position =  self.current_position
            return 1
        return super().moveToMe(entity_object)


    def _getAction(self,obs):
        #this updates the picture
        # print("human getAction")
        key_pressed = None
        while key_pressed == None:
            event = self.pygame.event.wait()
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_LEFT: key_pressed = LEFT
                if event.key == self.pygame.K_RIGHT: key_pressed = RIGHT
                if event.key == self.pygame.K_DOWN: key_pressed = DOWN
                if event.key == self.pygame.K_UP: key_pressed = UP
                if event.key == self.pygame.K_SPACE: key_pressed = NOOP
                if event.key == self.pygame.K_r: key_pressed = 'reset'
                if event.key == self.pygame.K_q: key_pressed = 'quit'

        if key_pressed == 'reset':
            self.env.reset()
            return 0
        if key_pressed == 'quit':
            self.quit = True
            return 0
        # print("human pressed", key_pressed)
        return {'actions':key_pressed}


    # @record_action




class Advisary(ActiveEntity):
    def moveToMe(self,entity_object):
        return super().moveToMe(entity_object)

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
        #who is moving?

        self.env.done = 1
        self.env.reward = -1
        # print('PREADATOR IS BEING ATTACKED')

    def _getAction(self,obs):
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


class ChasingAdvisary(Advisary):

    def _getAction(self,obs):
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
    def moveToMe(self,entity_object):
        self.current_position = self.intended_position
        #del self.env.active_entities[entity_object.value]
        print('CBA: entity', entity_object, 'hit me')
        self.env.done = True
        self.reward = -1

        #super().moveToMe(entity_object)

    def _getAction(self, obs):
        my_location = np.where(self.env.current_grid_map == self.value)
        goal_val = self.env.getGoalValue()
        # print('goal_val',goal_val)
        directions = [UP, DOWN, LEFT, RIGHT]
        random.shuffle(directions)
        goal_location = np.where(self.env.current_grid_map == goal_val)
        # if goal_location[0].size==0:
        #     print('DEBUG')
        agents = self.getAgents()
        # print('agents', agents)
        distance_to_agents = {}
        agents_to_goal = {}
        for agent in agents:
            agent_location = np.where(self.env.current_grid_map == agent)
            # print('myloc', my_location)
            # print('agentloc', agent_location)
            path_to_agent = self.env.getPathTo((my_location[0], my_location[1]), (agent_location[0], agent_location[1]),
                                      free_spaces=self.env.free_spaces)
            points_in_path = np.where(path_to_agent == -1)
            if len(points_in_path) < 2:
                # print("NOOP")
                return NOOP
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            if len(points_in_path) == 0: #no points exist, therefore no path
                points_in_path_length = 10**10
            else:
                points_in_path_length = len(points_in_path)

            # print('goalloc', goal_location)
            agent_to_goal = self.env.getPathTo((agent_location[0], agent_location[1]), (goal_location[0], goal_location[1]),
                                               free_spaces=self.env.free_spaces + [self.value])
            agent_to_goal_points = np.where(agent_to_goal == - 1)
            points_to_goal_path = list(zip(agent_to_goal_points[0], agent_to_goal_points[1]))

            distance_to_agents[agent] = {'dist': points_in_path_length, 'raw_path_to_agent':path_to_agent, 'path_to_agent': points_in_path, 'agent_to_goal':points_to_goal_path}

        #distance_to_agents now has the distance to the agent, the path to the agent, AND the points in the agents path to the goal
        #the first rule is to check if my own location is beyond 3 steps to the goal
        path_to_goal = self.env.getPathTo((my_location[0],my_location[1]), (int(goal_location[0]),int(goal_location[1])),free_spaces=self.env.free_spaces)
        path_to_goal_points = np.where(path_to_goal == -1)
        path_to_goal_points = list(zip(path_to_goal_points[0], path_to_goal_points[1]))
        if len(path_to_goal_points) > 3:
            for direction in directions:
                if path_to_goal[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    # print("diection2", direction)
                    return {'actions':direction}

        #find the closest agent to intercept
        distance_list = sorted(distance_to_agents.keys(), key=lambda k: distance_to_agents[k]['dist'])
        target_agent_val = distance_list[0]
        #print('target_', target_agent_val)

        if distance_to_agents[target_agent_val]['dist']  > 2:
            #go for the goal

            target_location = (-1, -1)
            goal_location = [int(goal_location[0]), int(goal_location[1])]
            agent_path = distance_to_agents[target_agent_val]['agent_to_goal']

            if (int(my_location[0]),int(my_location[1])) in agent_path:
                #print("already in the way")
                return {'actions':NOOP}
            for direction in directions:
                if self.env.action_map[direction](goal_location) in agent_path:
                    target_location = self.env.action_map[direction](goal_location)

            if target_location == (-1, -1):
                #print("target location -1 -1 NOOP")
                return {'actions':NOOP}
            if int(my_location[0]) == int(target_location[0]) and int(my_location[1]) == int(target_location[1]):
                #print("already at target location")
                return {'actions':NOOP}

            # print('targloc', target_location)
            path = self.env.getPathTo((my_location[0], my_location[1]), (target_location[0], target_location[1]),
                                      free_spaces=self.env.free_spaces)
            #if no path was found
            if not list(np.where(path == -1)[0]):
                #print("No path NOOP")
                return {'actions':NOOP}
            for direction in [UP, DOWN, LEFT, RIGHT]:

                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    #print("Getting in path")
                    #print(np.array2string(path))
                    #print('direction2', direction)
                    return {'actions':direction}
        else: #go for the agent
            path = distance_to_agents[target_agent_val]['raw_path_to_agent']
            #print("Going for agent")
            #print(np.array2string(path))
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    # print("diection2", direction)
                    return {'actions':direction}
        # return 2


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


    def _getAction(self,obs):
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


class NetworkAgent(Agent):

    def __init__(self,env,obs_type='image', entity_type='', color='', position='random-free'):
        self.env = env
        self.value = env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.position = position
        self.active = True



    def moveToMe(self,entity_object):
        self.env.done = True
        self.env.reward -=1


class TrainedAgent(Agent):

    def __init__(self, env, obs_type='image', entity_type='', color='', position='random-free',model_name=''):
        import tensorflow
        from actorcritic.agent import ActorCriticAgent, ACMode
        from actorcritic.runner import Runner, PPORunParams

        super().__init__(env, obs_type, entity_type, color, position)

        self.tf = tensorflow

        self.ActorCriticAgent = ActorCriticAgent
        self.ACMode = ACMode

        self.Runner = Runner
        self.PPORunParams = PPORunParams

        self.full_checkpoint_path = os.path.join("_files/models", model_name)


        self.obs_type = obs_type
        self.position = position
        #self.agent = ActorCriticAgent()
        #self.agent.buid_model()
        self.active = True
        self.record_history = False


        self.tf.reset_default_graph()
        self.config = self.tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = self.tf.Session(config=self.config)

        self.agent = ActorCriticAgent(
            mode=self.ACMode.A2C,
            sess=self.sess,
            spatial_dim=8,
            unit_type_emb_dim=5,
            loss_value_weight=0.5,
            entropy_weight_action_id=0.01,
            entropy_weight_spatial=0.00000001,
            scalar_summary_freq=5,
            all_summary_freq=10,
            summary_path='_files/summaries',
            max_gradient_norm=10.0,
            num_actions=5,
            num_envs=1,
            nsteps=16,
            obs_dim=(10,10),
            policy="FullyConv"
        )
        self.agent.build_model()

        if os.path.exists(self.full_checkpoint_path):
            self.agent.load(self.full_checkpoint_path, False)

        self.runner = Runner(
            envs=self.env,
            agent=self.agent,
            discount=0.95,
            n_steps=16,
            do_training=False,
            ppo_par=None,
            policy_type="FullyConv",
            n_envs=1
        )


    def _getAction(self,obs):
        print('getaction called...')
        obs = self.runner.obs_processer.process([obs])
        action_ids, value_estimate, fc, action_probs = self.agent.step_eval(obs)
        print('action_ids',action_ids)
        return action_ids[0]

    def moveToMe(self, entity_object):
        self.env.done = True
        self.env.reward -= 1