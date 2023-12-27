import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
from operator import add
import pandas as pd
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

fig, ax = plt.subplots()
x = [random.uniform(0, 10) for _ in range(10)]
y = [random.uniform(0, 10) for _ in range(10)]
scat = ax.scatter(x, y)
plt.xlim([0, 11])
plt.ylim([0, 11])
pos = pd.DataFrame(columns=['x', 'y','xd','yd','xv','yv','xa','ya','xyd'])
n=5
pos['x'] = [random.uniform(0, 10) for _ in range(n)]
pos['y'] = [random.uniform(0, 10) for _ in range(n)]
pos['xv'] = 0
pos['yv'] = 0
pos['xa'] = 0
pos['ya'] = 0
pos['xd'] = [np.zeros(n) for _ in range(n)]
pos['yd'] = [np.zeros(n) for _ in range(n)]
pos['xyd'] = [np.zeros(n) for _ in range(n)]
k=1
x0=2.5
maxd=1
dt=10/1000
def update(frame):
    global x,y,pos
    for i in range(len(pos)):
        pos.loc[:, 'xd'].iloc[i] = pos['x'] - pos['x'].iloc[i]
        pos.loc[:, 'yd'].iloc[i] = pos['y'] - pos['y'].iloc[i]
        pos.loc[:, 'xyd'].iloc[i] = (pos.loc[i,'xd']**2+pos.loc[i,'yd']**2)**0.5
        pos.loc[i,'xyd'][pos['xyd'][i]>x0*maxd] = np.nan
        pos.loc[i,'xyd'][pos['xyd'][i]==0.0] = np.nan
    pos['xa'] = (k*(x0-pos['xyd'])*pos['x']/pos['xyd']).apply(np.nansum)
    pos['ya'] = (k*(x0-pos['xyd'])*pos['y']/pos['xyd']).apply(np.nansum)
    pos['x'] += pos['xv']*dt + 0.5*pos['xa']*dt*dt
    pos['y'] += pos['yv']*dt + 0.5*pos['ya']*dt*dt
    pos['xv'] += pos['xa']*dt
    pos['yv'] += pos['ya']*dt
    pos[pos['x']>10]=10
    pos[pos['x']<0] = 0
    pos[pos['y']>10] = 10
    pos[pos['y']<0] = 0
    data = np.stack([pos['x'], pos['y']]).T
    scat.set_offsets(data)
    #scat = ax.scatter(x,y)
    return scat

ani = animation.FuncAnimation(fig=fig, func=update,frames=60, interval=dt*1000)
#ani.save()
plt.show()
