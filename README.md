# Generic Gridworld Gym for Reinforcement Learning/ Multiagent Learning and more 
(readme not in final version)
## Intention
The intention of this project is to make a generic gridworld that can be (easily) setup for whatever
gridworld task you're interested in modeling. It can support various scenarios such as learning to avoid obstacles, predators, acomplishing goals/tasks, collaboration/competition with other agents etc. It is compatible with Openai Gym and baselines modules.

## Default Example
The default example is a 10x10 grid with border walls.  

run keyboard_agent.py in the root directory to run the example.



## File Description
### /envs/core.py
core.py has all the agent classes.
##### class Entity
This is the base class of all non-features. 
...

## Version Notes
This version is expanded to include runnable (trained) network agents in the same way as normal agents. Should be able to load the weights as shown in the example.
This version also includes a ACTR in the core capable of doing multiproessing (in mac architecture seems stable)

keyboard_agent - 

data_collection_keyboard - 

datacollect_runner - 
