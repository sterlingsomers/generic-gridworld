#!/usr/bin/env python
from __future__ import print_function
import envs.generic_env
from envs.core import *
import pickle



env = envs.generic_env.GenericEnv(dims=(10,10))
goal = Goal(env,entity_type='goal',color='green')
player3 = AIAgent(env,entity_type='agent',color='orange',pygame=None)
advisary = ChasingBlockingAdvisary(env,entity_type='advisary',color='red',obs_type='data')

all_data = []
agents_to_track = [player3]
step_data = []
episodes = 100

env.reset()
for i in range(episodes):
    print("Episode", i)

    while 1:
        obs, reward, done, info = env.step([])
        # print("done", done)
        for agent in agents_to_track:
            step_data.append([type(agent).__name__, agent.value, agent.action_chosen[0], agent.action_chosen[1],env.value_to_objects])
        if done:
            break
    all_data.append(step_data)
    env.reset()


pickle.dump(all_data, open('1000_AIAgent.lst', 'wb'))
