from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# data = pd.read_csv('/Users/constantinos/Documents/Projects/MFEC_02orig/gridworld_05-13-15-08_0p99/results.csv')
d1 = pd.read_csv('/Users/constantinos/Documents/Projects/genreal_grid/stable_models'
                 '/run_ppo_bc_human_rl_feats_predator_PPO2_1-tag-episode_reward.csv')
r1 = d1['Value'].values
epoch1 = d1['Step'].values

d2 = pd.read_csv('/Users/constantinos/Documents/Projects/genreal_grid/stable_models'
                 '/run_ppo_feats_predator_PPO2_1-tag-episode_reward.csv')
r2 = d2['Value'].values
epoch2 = d2['Step'].values

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed
fig = plt.figure
y1 = smooth(r1, 0.8)
plt.plot(epoch1,r1, alpha=0.2)#, label='orig')
plt.plot(epoch1,y1, color= '#000080',label='bc+ppo (smoothed)', linewidth=3)

y2 = smooth(r2, 0.8)
plt.plot(epoch2,r2, alpha=0.2, color= '#940054')#, label='orig')
plt.plot(epoch2,y2, color= '#940054',label='ppo (smoothed)', linewidth=3)

plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')
# plt.show()
plt.savefig('behavioral_cloning.pdf',bbox_inches='tight')