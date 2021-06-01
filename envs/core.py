import numpy as np
import random
import math
import functools
from threading import Lock
# from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from scipy.spatial.distance import cityblock
import PIL
import time
import os

#import dill as pickle
from multiprocessing import Pool
from functools import partial

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4
directions = [NOOP, DOWN, UP, LEFT, RIGHT]

class Entity:
    current_position = (0,0)
    action_chosen = None

    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free'):
        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
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

    def moveToMe(self,entity_object):
        self.env.done = True
        self.env.reward -= 1

    def getAction(self,obs):
        if self.active:
            return random.choice([UP])
        else:
            return 0



    def moveTo(self,current_position,intended_position):
        current_position_value = self.env.current_grid_map[current_position[0], current_position[1]]
        intended_position_value = self.env.current_grid_map[intended_position[0], intended_position[1]]
        self.env.current_grid_map[current_position] = 0.0
        self.env.current_grid_map[intended_position] = current_position_value
        self.active = False
        return 1

    def place(self, position='random-free',positions=[]):
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


    def update(self):
        pass

class ActiveEntity(Entity):
    def __init__(self, env, obs_type='image',entity_type='entity', color='', position='random-free'):
        super().__init__(env, obs_type, entity_type, color, position)
        self.env.active_entities[self.value] = self

    def getAction(self, obs):
        record_dict = self._getAction(obs)
        #action = self._getAction(obs)
        if self.record_history:
            for key,value in record_dict.items():
                self.history[key].append(value)
        return record_dict['actions']

class Goal(Entity):
    def __init__(self, env, obs_type='image',entity_type='goal', color='', position='random-free'):
        super().__init__(env, obs_type, entity_type, color, position)

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


class Agent(ActiveEntity):
    def __init__(self, env, obs_type='image',entity_type='agent', color='', position='random-free'):
        super().__init__(env, obs_type, entity_type, color, position)

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
    def __init__(self, env, obs_type='image', entity_type='agent', color='', position='random-free'):
        super().__init__(env, obs_type, entity_type, color, position)


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

class ACTR(Agent):
    import pyactup


    def __init__(self, env, obs_type='image',entity_type='agent', color='orange', position='random-free',data=[],mismatch_penalty=1,temperature=1,noise=0.0,decay=0.0,multiprocess=False,processes=4):
        super().__init__(env, obs_type, entity_type, color, position)
        self.mismatch_penalty = mismatch_penalty
        self.temperature = temperature
        self.noise = noise
        self.decay = decay
        self.memory = self.pyactup.Memory(noise=self.noise,decay=decay,temperature=temperature,threshold=-100.0,mismatch=mismatch_penalty,optimized_learning=False)
        self.pyactup.set_similarity_function(self.angle_similarity, *['goal_rads','advisary_rads'])
        self.pyactup.set_similarity_function(self.distance_similarity, *['goal_distance','adversary_distance'])
        self.pyactup.set_similarity_function(self.value_similarity, 'value','action')
        self.multiprocess = multiprocess
        self.processes = processes
        self.data = data
        self.last_observation = np.zeros(self.env.current_grid_map.shape)
        self.last_imagined_map = np.zeros(self.env.current_grid_map.shape)
        self.should_store = None #In this version we don't store everything
        self.action_map = {1: lambda x: ((x[0] + 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           2: lambda x: ((x[0] - 1) % self.env.dims[0], (x[1]) % self.env.dims[1]),
                           3: lambda x: (x[0] % self.env.dims[0], (x[1] - 1) % self.env.dims[1]),
                           4: lambda x: (x[0] % self.env.dims[0], (x[1] + 1) % self.env.dims[1]),
                           0: lambda x: (x[0], x[1])}

        self.relative_action_categories = {UP:{'lateral':[LEFT,RIGHT],'away':DOWN},
                                           DOWN:{'lateral':[LEFT,RIGHT],'away':UP},
                                           LEFT:{'lateral':[UP,DOWN],'away':RIGHT},
                                           RIGHT:{'lateral':[UP,DOWN],'away':LEFT}}

        #action 0=noop, -1=away, 1=towards



        #Before using the distances, they have to be normalized (0 to 1)
        #Normalize by dividing by the max in the data
        self.max_distance = 14

        #for now, don't add any chunks at all
        # for chunk in self.data:
        #     self.memory.learn(**chunk)

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

    def value_similarity(self,x,y):
        return 1 - abs(x-y)/2


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
    #         return_dict['goal_rads'] = goal_rads
    #         return_dict['goal_distance'] = len(points_in_path) / self.max_distance
    #     if advisary_location:
    #         advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
    #         advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
    #                                    advisary_location[1] - agent_location[1])
    #         path_agent_to_advisary = self.env.getPathTo(agent_location, advisary_location, free_spaces=[0])
    #         points_in_path = np.where(path_agent_to_advisary == -1)
    #         points_in_path = list(zip(points_in_path[0], points_in_path[1]))
    #         return_dict['advisary_rads'] = advisary_rads
    #         return_dict['advisary_distance'] = len(points_in_path) / self.max_distance

        # the distances need to be normalized


        return return_dict

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
                    if 'angle' in feature:
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
                    if 'angle' in feature:
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
            path_agent_to_goal = self.getPathTo(gridmap, agent_location, goal_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_goal == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['goal_rads'] = goal_rads
            return_dict['goal_distance'] = len(points_in_path) / self.max_distance
        if advisary_location:
            advisary_location = (int(advisary_location[0]), int(advisary_location[1]))
            advisary_rads = math.atan2(advisary_location[0] - agent_location[0],
                                       advisary_location[1] - agent_location[1])
            path_agent_to_advisary = self.getPathTo(gridmap, agent_location, advisary_location, free_spaces=[0])
            points_in_path = np.where(path_agent_to_advisary == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            return_dict['adversary_rads'] = advisary_rads
            return_dict['adversary_distance'] = len(points_in_path) / self.max_distance

        # for wall, dir in {'up_wall_distance': 2, 'left_wall_distance': 3, 'down_wall_distance': 1,
        #                   'right_wall_distance': 4}.items():
        #     current_position = agent_location
        #     dist = 0
        #     while True:
        #         dist += 1
        #         current_position = action_map[dir](current_position)
        #         if gridmap[current_position] == 1:
        #             return_dict[wall] = dist / self.max_distance
        #             break

        # the distances need to be normalized

        return return_dict

    def _getAction(self,obs):
        print(np.array2string(self.env.current_grid_map))
        self.last_observation = self.env.current_grid_map.copy()
        symbolic_obs = self.gridmap_to_symbols(self.env.current_grid_map.copy(),self.value,self.env.value_to_objects)

        #pick the action plan that maximizes value
        value_estimates = {}
        for i in [-1,1]:
            value_estimates[i] = self.memory.blend('value',goal_distance=symbolic_obs['goal_distance'],adversary_distance=symbolic_obs['adversary_distance'],action=i)

        max_action = max(value_estimates, key=value_estimates.get)


        ##### an alternate approach
        #which actions bring me towards?
        goal_position = np.where(self.env.current_grid_map == 2)
        goal_position = list(zip(goal_position[0], goal_position[1]))
        current_position = self.current_position
        current_distance = np.linalg.norm(np.array(current_position) - np.array(goal_position[0]))
        direction_distances = {}
        for direction in directions:
            new_position = self.action_map[direction](current_position)
            if self.env.current_grid_map[new_position] == 1: #if i hit a wall, no move
                new_position = self.current_position
            new_distance = np.linalg.norm(np.array(new_position) - np.array(goal_position[0]))
            direction_distances[direction] = new_distance

        #closers = sorted(direction_distances, key=direction_distances.get)
        #no noop for now
        chosen_action = 0
        if goal_estimate > 0.5:
            towards = []
            for action in direction_distances:
                if direction_distances[action] < current_distance:
                    towards.append(action)
            chosen_action = random.choice(towards)
        elif goal_estimate <= 0.5:
            away = []
            for action in direction_distances:
                if direction_distances[action] > current_distance:
                    away.append(action)
            chosen_action = random.choice(away)

        print('CA',chosen_action)
        return {'actions':round(chosen_action),'saliences':0,'stuck':self.stuck}


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
        self.env.reward = -1

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