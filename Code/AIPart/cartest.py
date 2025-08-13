
from car import Cars
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from car import *




from graph import Graph


# 设置图形
fig, ax = plt.subplots(figsize=(20, 4))
ax.set_xlim(40, 160)
ax.set_ylim(-12,17)

#尾迹时间
wake = 0

# 向量增长比例
vectork_v = 0.2
vectork_a = 1

#帧速，模拟速度
#fps * t = frame_speed
dt = 0.01
frame_speed = 1

ax.plot([0,1000],[0,0],color='black')
ax.plot([0,1000],[3,3],color='black')
ax.plot([0,1000],[6,6],color='black')
ax.plot([0,1000],[9,9],color='black')
ax.plot([0,1000],[12,12],color='black')



c_list = [
    Car(1,50,4.5,"0"),
    Car(0,50,1.5,"1"),
    Car(1,70,1.5,"2"),  ###############
    Car(1,70,4.5,"3"),  ###############
    Car(2,70,7.5,"4"),
    Car(1,75,4.5,"5"),
    Car(2,90,1.5,"6"),    #########
    Car(2,90,7.5,"7"),
    Car(0,110,1.5,"8"),
    Car(1,110,4.5,"9"),
]


cars=Cars2(None,c_list)

xs = []
ys = []
lines= []
points= []
vectors_v = []
vectors_a = []
tangents = []
texts= []


def to_tangent(x,y,w,h):
    return [x+w/2,x+w/2,x-w/2,x-w/2,x+w/2],[y+h/2,y-h/2,y+h/2,y-h/2,y+h/2]
def from_centre(x:float,y:float,w:float,h:float):
    return x-w/2,y-h/2

# init
for i in range(len(c_list)):
    xs.append([])
    ys.append([])
    lines.append(ax.plot([], [], lw=2)[0])
    points.append(ax.plot(1,1 , 'o', markersize=5)[0])
    vectors_v.append(ax.arrow(1, 1, 1, 1, head_width=0, head_length=0, ec='blue'))
    vectors_a.append(ax.arrow(1, 1, 1, 1, head_width=0, head_length=0, ec='red'))
    tangents.append(Rectangle((0,0),4,2))
    ax.add_patch(tangents[i])
    texts.append(ax.text(0, 0, '', horizontalalignment='left', verticalalignment='top'))



def simulate(frame):
    cars.simulate(dt=dt)

    for i,car in enumerate(cars.cars):
        xs[i].append(car.p_x)
        ys[i].append(car.p_y)

        points[i].set_data(xs[i][-2:-1], ys[i][-2:-1])

        if len(xs[i])>wake/dt+1:
            lines[i].set_data(xs[i][int(-(wake/dt+1)):-1].copy(), ys[i][int(-(wake/dt+1)):-1].copy())
        else:
            lines[i].set_data(xs[i].copy(), ys[i].copy())
        vectors_v[i].set_data(x=xs[i][-1], y=ys[i][-1], dx=vectork_v * car.v_x, dy=vectork_v * car.v_y)
        vectors_a[i].set_data(x=xs[i][-1], y=ys[i][-1], dx=vectork_a * car.a_x, dy=vectork_a * car.a_y)
        tangents[i].set_xy(from_centre(xs[i][-1], ys[i][-1],4,2))
        tangents[i].set_angle(np.degrees(np.atan(car.v_y/car.v_x)))
        texts[i].set_text(car.text)
        texts[i].set_position((xs[i][-1], ys[i][-1]))




# 动画更新函数
def update(frame):
    simulate(frame)

    return *points,*lines,*tangents,*texts,*vectors_v,*vectors_a

ani = FuncAnimation(fig, update, frames=100,
                    blit=True, interval=1000*dt/frame_speed)

plt.title('cav')
plt.xlabel('x')
plt.grid()
plt.show()