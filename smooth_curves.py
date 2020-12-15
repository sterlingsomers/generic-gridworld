from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('/Users/constantinos/Documents/Projects/MFEC_02orig/gridworld_05-13-15-08_0p99/results.csv')
r = data[' episode_nums'].values
epoch = data['epoch'].values

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

y = smooth(r, 0.8)
plt.plot(epoch,r, alpha=0.2, label='orig')
plt.plot(epoch,y, color= '#000080',label='smoothed')
plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()