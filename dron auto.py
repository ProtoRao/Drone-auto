import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
from operator import add
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

fig, ax = plt.subplots(1,2,figsize=(12.5,6.125))
lims=5
ax[0].set_xlim(-lims, 10+lims)
ax[0].set_ylim(-lims, 10+lims)
pos = pd.DataFrame(columns=['x', 'y','xd','yd','xv','yv','xa','ya','xyd'])

n=10

pos['x'] = [random.uniform(0, 10) for _ in range(n)]
pos['y'] = [random.uniform(0, 10) for _ in range(n)]
scat = ax[0].scatter(pos['x'], pos['y'],color='b')

line=[ [0]*n for i in range(n)]
for i in range(len(pos)):
    for j in range(len(pos)):
        line[i][j] = ax[0].plot([pos['x'][i], pos['x'][j]], [pos['y'][i], pos['y'][j]])[0]
# pos['x'] = [2,5,8]
# pos['y'] = [2,7.2,2]
pos['xv'] = 0
pos['yv'] = 0
pos['xa'] = 0
pos['ya'] = 0
pos['xd'] = [np.zeros(n) for _ in range(n)]
pos['yd'] = [np.zeros(n) for _ in range(n)]
pos['xyd'] = [np.zeros(n) for _ in range(n)]

# ax2.set_xlim(-lims, 10+lims)
# ax2.set_ylim(-lims, 10+lims)
line2 = ax[1].plot([],[])[0]
line3 = ax[1].plot([],[],'k')[0]
l=[]
l.append(np.nanmean(pos['xyd'][0]))

k=1
x0=5
maxd = 1.5
mina = 0.1
fps=24
dt=1/fps

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = current_value - self.setpoint

        # Proportional term
        P = self.kp * error

        # Integral term
        self.integral += error * dt
        I = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative

        # PID control signal
        control_signal = P + I + D

        # Update previous error for the next iteration
        self.prev_error = error

        return control_signal

kp = 0.5#0.1
ki = 0.05#0.01
kd = 0.25#0.05

# Setpoint (target distance)
setpoint = x0

# Create PID controller
pid_controller = PIDController(kp, ki, kd, setpoint)

    
def update(frame):
    global pos
    for i in range(len(pos)):
        pos.loc[:, 'xd'].iloc[i] = pos['x'] - pos['x'].iloc[i]
        pos.loc[:, 'yd'].iloc[i] = pos['y'] - pos['y'].iloc[i]
        pos.loc[:, 'xyd'].iloc[i] = (pos.loc[i,'xd']**2+pos.loc[i,'yd']**2)**0.5
    pos['xyd'] = pos['xyd'].apply(lambda arr: np.where(arr > x0 * maxd, np.nan, arr))
    pos['xyd'] = pos['xyd'].apply(lambda arr: np.where(arr == 0, np.nan, arr))
    pos['xa'] = (pid_controller.update(pos['xyd'], dt)*pos['xd']/pos['xyd']).apply(np.nansum)
    pos['ya'] = (pid_controller.update(pos['xyd'], dt)*pos['yd']/pos['xyd']).apply(np.nansum)
    # pos['xv'] = np.where(abs(pos['xa']) < mina, 0, pos['xv'])
    # pos['yv'] = np.where(abs(pos['ya']) < mina, 0, pos['yv'])
    pos['x'] += pos['xv']*dt + 0.5*pos['xa']*dt*dt
    pos['y'] += pos['yv']*dt + 0.5*pos['ya']*dt*dt
    pos['xv'] += pos['xa']*dt
    pos['yv'] += pos['ya']*dt
    data = np.stack([pos['x'], pos['y']]).T
    ax[0].set_xlim(pos['x'].min()-2,pos['x'].max()+2)
    ax[0].set_ylim(pos['y'].min()-2,pos['y'].max()+2)
    scat.set_offsets(data)
    for i in range(len(pos)):
        for j in range(len(pos)):
            if pos['xyd'][i][j]<(x0*(1+0.05)) and pos['xyd'][i][j]>(x0*(1-0.05)):
                line[i][j].set_data([pos['x'][i], pos['x'][j]], [pos['y'][i], pos['y'][j]])
                line[i][j].set_color('r')
            elif pos['xyd'][i][j]<(x0 * maxd):
                line[i][j].set_data([pos['x'][i], pos['x'][j]], [pos['y'][i], pos['y'][j]])
                line[i][j].set_color('c')
            else:
                line[i][j].set_data([0],[0])
    # os.system('cls')
    # print(pos)
    l.append(pos['xyd'].apply(lambda x: np.nanmean(x)).mean())
    ax[1].set_xlim(0,len(l)*1.0)
    ax[1].set_ylim(0,max(l)*1.5)
    line2.set_data([list(range(len(l)))],[l])
    line3.set_data([list(range(len(l)))],[x0]*len(l))
    #print(l)
    return scat,line2

ani = animation.FuncAnimation(fig=fig, func=update,frames=fps, interval=dt*1000)
#ani.save()
plt.show()
        