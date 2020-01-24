import numpy as np
import random
import math
from threading import Lock
# from envs.generic_env import UP, DOWN, LEFT, RIGHT, NOOP
from scipy.spatial.distance import cityblock
import PIL
import time

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

class Entity:
    current_position = (0,0)

    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free'):
        self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.active = True

    # def register_environment
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

    def place(self, position='random-free'):
        if position == 'random-free':
            free_spaces = []
            for free_space in self.env.free_spaces:
                found_spaces = np.where(self.env.current_grid_map == free_space)
                free_spaces.extend(list(zip(found_spaces[0], found_spaces[1])))
            the_space = random.choice(free_spaces)
            self.env.current_grid_map[the_space] = self.value
            self.current_position = the_space

    def update(self):
        pass
        # print("regular update")

class Goal(Entity):
    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free'):
        self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.active = True

    def moveToMe(self,entity_object):
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


class Agent(Entity):
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
    def __init__(self, env, obs_type='data',entity_type='', color='', position='random-free',pygame='None'):
        self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.pygame = pygame



    def getAction(self,obs):
        #go straight for the goal
        my_location = np.where(self.env.current_grid_map == self.value)
        goal_val = self.env.getGoalValue()
        goal_location = np.where(self.env.current_grid_map == goal_val)
        path = self.env.getPathTo((my_location[0], my_location[1]), (int(goal_location[0]), goal_location[1]),
                                  free_spaces=self.env.free_spaces + [4])
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                #print("diection2", direction)
                return direction
        return random.choice([UP,DOWN,LEFT,RIGHT])


class HumanAgent(Agent):
    obs = None
    def __init__(self, env, obs_type='image',entity_type='', color='', position='random-free',pygame='None'):
        self.size_factor = 10
        self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.action = 0
        self.pygame = pygame

    def getAction(self,obs):
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
                if event.key == self.pygame.K_r: key_pressed = 'reset'

        if key_pressed == 'reset':
            self.env.reset()
            return 0
        # print("human pressed", key_pressed)
        return key_pressed



class Advisary(Entity):
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

    def getAction(self,obs):
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
        # print('goal_val',goal_val)

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

            # print('goalloc', goal_location)
            agent_to_goal = self.env.getPathTo((agent_location[0], agent_location[1]), (goal_location[0], goal_location[1]),
                                               free_spaces=self.env.free_spaces+ [self.value])
            agent_to_goal_points = np.where(agent_to_goal == - 1)
            points_to_goal_path = list(zip(agent_to_goal_points[0], agent_to_goal_points[1]))

            distance_to_agents[agent] = {'dist': len(points_in_path), 'raw_path_to_agent':path_to_agent, 'path_to_agent': points_in_path, 'agent_to_goal':points_to_goal_path}

        #assume we've determined the closest agent
        #use the only one that exists
        if distance_to_agents[agents[0]]['dist'] > 3:
            #go for the goal
            target_location = (-1, -1)
            goal_location = [int(goal_location[0]), int(goal_location[1])]
            agent_path = distance_to_agents[agent]['agent_to_goal']
            directions = [UP, DOWN, LEFT, RIGHT]
            if (int(my_location[0]),int(my_location[1])) in agent_path:
                # print("in there....")
                return NOOP
            for direction in directions:
                if self.env.action_map[direction](goal_location) in agent_path:
                    target_location = self.env.action_map[direction](goal_location)

            if target_location == (-1, -1):
                return NOOP
            if int(my_location[0]) == int(target_location[0]) and int(my_location[1]) == int(target_location[1]):
                # print("noooop")
                return NOOP

            # print('targloc', target_location)
            path = self.env.getPathTo((my_location[0], my_location[1]), (target_location[0], target_location[1]),
                                      free_spaces=self.env.free_spaces)
            #if no path was found
            if not list(np.where(path == -1)[0]):
                # print("NOOP 2")
                return NOOP
            for direction in [UP, DOWN, LEFT, RIGHT]:

                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    # print("direction", direction)
                    return direction
        else: #go for the agent
            path = distance_to_agents[agents[0]]['raw_path_to_agent']
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if path[self.env.action_map[direction]((my_location[0], my_location[1]))] == -1:
                    # print("diection2", direction)
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


class NetworkAgent(Agent):

    def __init__(self,env,obs_type='image', entity_type='', color='', position='random-free'):
        self.env = env.env
        self.value = self.env.object_values[-1] + 1
        self.env.object_values.append(self.value)
        self.env.value_to_objects[self.value] = {'color': color,'entity_type':entity_type}
        self.env.entities[self.value] = self
        self.color = color
        # self.moveTo = 'moveToDefault'
        self.entity_type = entity_type
        self.obs_type = obs_type
        self.active = True

    def moveToMe(self,entity_object):
        self.env.done = True
        self.env.reward -=1