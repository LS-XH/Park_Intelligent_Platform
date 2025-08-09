from car import Cars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from car import *


# 设置图形
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(40, 200)
ax.set_ylim(-3,21)



cars=Cars(None,[
    Car(0,50,5),
    Car(0,70,3),
    Car(1,70,1,"3"),
    Car(0,90,1),
])

xs = []
ys = []
lines= []
points= []
texts= []

# init
for i in range(4):
    xs.append([])
    ys.append([])
    lines.append(ax.plot([], [], lw=2)[0])
    points.append(ax.plot(1,1 , 'o', markersize=5)[0])
    texts.append(ax.text(0, 0, '', horizontalalignment='left', verticalalignment='top'))



def simulate(frame):
    cars.simulate(dt=0.01)

    for i,car in enumerate(cars.cars):
        xs[i].append(car.p_x)
        ys[i].append(car.p_y)

        points[i].set_data(xs[i][-2:-1], ys[i][-2:-1])
        lines[i].set_data(xs[i],ys[i])
        texts[i].set_text("%f,%f"%(car.a_x,car.a_y))
        texts[i].set_position((xs[i][-1], ys[i][-1]))




# 动画更新函数
def update(frame):
    simulate(frame)

    return *lines,*points,*texts

ani = FuncAnimation(fig, update, frames=100,
                    blit=True, interval=40)

plt.title('cav')
plt.xlabel('x')
plt.grid()
plt.show()