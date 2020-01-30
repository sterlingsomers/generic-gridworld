#This project neds a readme. 

##Intention
The intention of this project is to make a generic gridworld that can be (easily) setup for whatever
gridworld task you're interested in modeling. 

##Default Example
The default example is a 10x10 grid with border walls.  

run keyboard_agent.py in the root directory to run the example.



#File Description
###/envs/core.py
core.py has all the agent classes.
#####class Entity
This is the base class of all non-features. 
...

#Version Notes


#Sample Code
--version update (inheritance_update)

##Generating an environment
####Creating an empty environment with border walls, with specific dimensions
env = envs.generic_env.GenericEnv(dims=(10,10))

####