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

import dill as pickle
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
        self.env.info['fire'] = -1 # someone moves to the predator which is an Advisary which is an Entity

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

    def place(self, position='random-free',position_coords=()):
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
        #print('ACTS=', record_dict, 'entity:', self.value, 'self',self)
        return record_dict['actions']


class Goal(Entity):
    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free', position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)
        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        # self.value = self.env.object_values[-1] + 1
        # self.env.object_values.append(self.value)
        # self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        # self.env.entities[self.value] = self
        # self.color = color
        # # self.moveTo = 'moveToDefault'
        # self.entity_type = entity_type
        # self.obs_type = obs_type
        # self.position = position
        # self.active = True

    def moveToMe(self,entity_object):
        if isinstance(entity_object, Advisary):
            entity_object.intended_position = entity_object.current_position
            return 0
        # print('entity hit goal', entity_object)
        self.env.done = True
        self.env.reward += 1
        self.env.info['goal'] = 1

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

        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        # self.value = self.env.object_values[-1] + 1
        # self.env.object_values.append(self.value)
        # self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        # self.env.entities[self.value] = self
        # self.color = color
        # # self.moveTo = 'moveToDefault'
        # self.entity_type = entity_type
        # self.obs_type = obs_type
        # self.position = position
        # self.pygame = pygame

    def moveToMe(self,entity_object):
        # print('enity', entity_object, 'hit', self)
        if isinstance(entity_object,Agent):
            entity_object.intended_position = entity_object.current_position
            return 1
        if isinstance(entity_object, RunAwayGoal):
            self.env.done = True
            self.env.reward = 0.5
            return 1
        if isinstance(entity_object, NetworkAgent):
            entity_object.intended_position = entity_object.current_position
            return 1
        return super().moveToMe(entity_object)

    def getAction(self,obs):
        # print('obs=', obs)
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


class HumanAgent(Agent):
    obs = None
    def __init__(self, env, obs_type='image',entity_type='agent', color='', position='random-free',position_coords=[],pygame='None',mapping={'\uf700':UP,'\uf702':LEFT,'\uf701':DOWN,'\uf703':RIGHT,' ':NOOP,'r':'reset','q':'quit'}):
        self.size_factor = 10
        self.mapping = mapping
        super().__init__(env, obs_type, entity_type, color, position, position_coords)
        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        # self.value = self.env.object_values[-1] + 1
        # self.env.object_values.append(self.value)
        # self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        # self.env.entities[self.value] = self
        # self.color = color
        # # self.moveTo = 'moveToDefault'
        # self.entity_type = entity_type
        # self.obs_type = obs_type
        # self.action = 0
        # self.position = position
        self.pygame = pygame
        self.keymappings = {}
        # self.pygame.font.init()
        # self.font = self.pygame.font.SysFont('Arial', 30)
        self.display = 0
        self.walls_hit = 0


        self.quit = False

    def setDisplay(self,display):
        self.display = display
    def stepCheck(self):
        if self.walls_hit:
            self.env.done = True
            self.env.reward = - 1
            self.walls_hit = 0

    def hitWall(self):
        self.walls_hit += 1

    def moveToMe(self,entity_object):
        # print('enity', entity_object, 'hit', self)
        if isinstance(entity_object,Agent):
            entity_object.intended_position = entity_object.current_position
            self.intended_position =  self.current_position
            return 1
        return super().moveToMe(entity_object)

    def getAction(self,obs):
        #this updates the picture
        # print("human getAction")
        if self.display:
            snapshot = self.display.copy()
            textsurface = self.font.render(self.color, False, (255,255,255))
            size = self.pygame.display.get_surface().get_size()
            self.display.blit(textsurface, (10,220))
            self.pygame.display.update()

        key_pressed = None
        while key_pressed == None:
            event = self.pygame.event.wait()
            if event.type == self.pygame.KEYDOWN:
                 if event.unicode in self.mapping:
                    key_pressed = self.mapping[event.unicode]
                # if event.key == self.pygame.K_LEFT: key_pressed = LEFT
                # if event.key == self.pygame.K_RIGHT: key_pressed = RIGHT
                # if event.key == self.pygame.K_DOWN: key_pressed = DOWN
                # if event.key == self.pygame.K_UP: key_pressed = UP
                # if event.key == self.pygame.K_SPACE: key_pressed = NOOP
                # if event.key == self.pygame.K_r: key_pressed = 'reset'
                # if event.key == self.pygame.K_q: key_pressed = 'quit'

        if key_pressed == 'reset':
            self.env.reset()
            return 0
        if key_pressed == 'quit':
            self.quit = True
            return 0
        # print("human pressed", key_pressed)
        if self.display:
            self.display.blit(snapshot, (0,0))
            self.pygame.display.update()
        return {'actions':key_pressed}


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


class RunAwayGoal(ActiveEntity):

    def getAgents(self):
        agents = []
        for entity in self.env.entities:
            if isinstance(self.env.entities[entity], Agent):
                agents.append(self.env.entities[entity])
        return agents

    def _getAction(self,obs):
        #find the closest agent
        # print("here")
        # print('obs _getAction=', obs)
        directions = [UP, DOWN, LEFT, RIGHT]
        agents = self.getAgents()
        distance_to_agent = {}
        for agent in agents:
            agent_location = agent.current_position
            path_to_agent = self.env.getPathTo(self.current_position, agent.current_position, free_spaces=self.env.free_spaces)
            points_in_path = np.where(path_to_agent == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            distance_to_agent[agent] = len(points_in_path)

        min_agent = min(distance_to_agent, key=distance_to_agent.get)
        #min_agent (should be) the closest agent (in steps)
        #now I want the edge that is furthest away from that agent and I want to go there
        #find the edges, and then find which edge is the furthest PATH for that agent to get to
        #go there
        # print("here2")
        edges = self.env.edges
        edge_distances = {}
        edge_paths = {}
        for edge in edges:
            # print('get edge path', min_agent.current_position, edge )
            path_to_edge = self.env.getPathTo(min_agent.current_position, edge, free_spaces=self.env.free_spaces + [self.value])
            points_in_path = np.where(path_to_edge == -1)
            points_in_path = list(zip(points_in_path[0], points_in_path[1]))
            # print('pip', points_in_path)
            if len(points_in_path) == 0:
                # print('if')
                edge_distances[edge] = -1
                edge_paths[edge] = path_to_edge

            else:
                # print('else')
                edge_distances[edge] = len(points_in_path)
                edge_paths[edge] = path_to_edge

        # print('max')
        max_point = max(edge_distances, key=edge_distances.get)
        # print('here3')
        #max_point should be the point most distant from the closest agent
        # print('max points', max_point)
        target_path = self.env.getPathTo(self.current_position, max_point, free_spaces=self.env.free_spaces)

        # print(target_path)
        random.shuffle(directions)
        for direction in directions:
            if target_path[self.env.action_map[direction]((self.current_position[0], self.current_position[1]))] == -1:
                return {'actions':direction}

        return {'actions':0}

    def moveToMe(self,entity_object):
        # super().moveToMe(entity_object)
        self.env.done = True
        if isinstance(entity_object, AIAgent):
            self.env.reward = 0.5
            return 1
        if isinstance(entity_object, NetworkAgent):
            self.env.reward = 1.0
            return 1


class ChasingAdvisary(Advisary):

    def _getAction(self,obs):
        # print('obs=', obs)
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
        # print('CBA: entity', entity_object, 'hit me')
        super().moveToMe(entity_object)

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
                return {'actions':NOOP}
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
        #the first rule is to check if my own location is beyond 5 steps to the goal
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
        # print('target_', target_agent_val)

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
                    # print("direction", direction)
                    return {'actions':direction}
        else: #go for the agent
            path = distance_to_agents[target_agent_val]['raw_path_to_agent']
            # print("Going for agent")
            # print(np.array2string(path))
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
    def __init__(self,env,obs_type='image', entity_type='', color='', position='random-free', position_coords=[]):
        super().__init__(env, obs_type, entity_type, color, position, position_coords)

        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        # self.value = self.env.object_values[-1] + 1 # Weird here: in MAC this works with just env.object... even with the wrapper (and the env. TAB gives no suggestions). But in koalemos works
        # self.env.object_values.append(self.value)
        # self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        # self.env.entities[self.value] = self
        # self.color = color
        # # self.moveTo = 'moveToDefault'
        # self.entity_type = entity_type
        # self.obs_type = obs_type
        # self.position = position
        # self.active = True
        # self.record_history = False



    def moveToMe(self,entity_object): # Means smth moves to the network
        environment = self.env
        # if self.env.__class__.__name__ == 'GenericEnv':
        #     environment = self.env
        # else:
        #     environment = self.env.env
        if isinstance(entity_object,Agent): # Notes: Any agent hitting our agent
            # print('entity_object.intended_position=', entity_object.intended_position)
            entity_object.intended_position = entity_object.current_position
            return 1
        if isinstance(entity_object, Goal):
            # print('2nd done')
            environment.done = True
            environment.reward = 1
            return 1
        if isinstance(entity_object, ChasingBlockingAdvisary):
            # print('3rd')
            # print('self 3rd=',self.env)
            environment.done = True
            environment.reward -=1
            # print('net being captured')
            environment.info['fire'] = -1


class TrainedAgent(Agent):
    def __init__(self, env, obs_type='image', entity_type='', color='', position='random-free'):
        if env.__class__.__name__ == 'GenericEnv':
            self.env = env
        else:
            self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color, 'entity_type': entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.position = position
        #self.agent = ActorCriticAgent()
        #self.agent.buid_model()
        self.active = True

    def getAction(self,obs):
        pass
        #network stuff
        #return action

    def moveToMe(self, entity_object):
        self.env.done = True
        self.env.reward -=1