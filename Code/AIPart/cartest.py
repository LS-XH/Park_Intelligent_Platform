from car import Cars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from car import *


# 设置图形
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(40, 200)
ax.set_ylim(-3,21)



Cars=Cars(None,[
    Car(0,50,5),
    Car(0,70,3),
    Car(1,70,1),
    Car(0,90,1),
])

xs = []
ys = []
lines=[]

# init
for i in range(4):
    xs.append([])
    ys.append([])
    lines.append(ax.plot([], [], lw=2)[0])



def simulate():
    Cars.simulate(dt=0.01)

    for i,car in enumerate(Cars.cars):
        xs[i].append(car.p_x)
        ys[i].append(car.p_y)

        lines[i].set_data(xs[i],ys[i])




# 动画更新函数
def update(frame):
    simulate()
    return lines

ani = FuncAnimation(fig, update, frames=100,
                    blit=True, interval=20)

plt.title('cav')
plt.xlabel('x')
plt.grid()
plt.show()