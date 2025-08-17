from sympy.physics.units.definitions import curie
from typing_extensions import Optional
import numpy as np
import torch
import random

from Interface.physics import RigidBody
from car.cavcar import Car,Road
from Interface.car import CarsBase,Delegation
import Interface
import car.tendency as td
from graph.graph import ROAD_WIDTH


import time



class CAVLane(Delegation):
    """
    比较特殊的委托，支持重复添加，但只会添加一次
    """
    def __init__(
            self,
            graph:Optional[Interface.GraphBase],
            cars:list[Car],
            stop_line,
            lane,
            ax:float=0.8,
            bx:float=4,
            ay:float=0.5,
            by:float=4
    ):
        super().__init__(cars)

        self.graph = graph
        self.stop_line = stop_line
        self.lane = lane
        self.ax = ax
        self.bx = bx
        self.ay = ay
        self.by = by

        for car in self.cars:
            car.p_y = (car.get_lane() + 0.5) * Car.lane_length
            car.v_y = 0


        self.convergence = True

    def cav_x(self,obj:Car,target:Car,r_x):
        return self.ax * ((target.p_x - obj.p_x - r_x) + self.bx * (target.v_x - obj.v_x))

    def cav_l(self,obj:Car,r_px):
        delta = (r_px - obj.p_x)
        if delta > 30:
            delta = 30
        return self.ax * (delta + self.bx * (0 - obj.v_x))

    def cav_align(self,obj:Car,lane):
        return self.ay * (((lane+0.5) * Car.lane_length - obj.p_y)+ self.by * (0 - obj.v_y))

    def simulate(self,dt:float=0.1,light = False):
        self.convergence = True
        self.cars.sort(key=lambda car: car.p_x+car.get_lane()*0.01,reverse=True)


        for i,car in enumerate(self.cars):
            car.a_x = 0
            car.a_y = 0

            # 对齐当前车道
            car.a_y = self.cav_align(car,self.lane)

            if i == 0:
                if not light:
                    car.a_x = self.cav_l(car,self.stop_line)
                else:
                    car.a_x = self.cav_l(car, self.stop_line)+10

            else:
                car.a_x = self.cars[0].a_x
                car.a_x += self.cav_x(car,self.cars[0],i*20)
                car.a_x += self.cav_x(car,self.cars[i-1],20)

                # 收敛判断
                if abs(car.a_x - self.cars[0].a_x) > 0.5:
                    self.convergence = False
    def simulate_one(self,dt:float=0.1,obj:Car=None,light:bool=False):
        self.convergence = True
        self.cars.sort(key=lambda car: car.p_x,reverse=True)

        if obj not in self.cars:raise Exception

        obj.a_x = 0
        obj.a_y = 0

        # 对齐当前车道
        obj.a_y = self.cav_align(obj, self.lane)

        if self.cars.index(obj) == 0:
            if not light:
                obj.a_x = self.cav_l(obj, self.stop_line)
            else:
                obj.a_x = self.cav_l(obj, self.stop_line) + 10

        else:
            obj.a_x = self.cars[0].a_x
            obj.a_x += self.cav_x(obj, self.cars[0], self.cars.index(obj) * 20)
            obj.a_x += self.cav_x(obj, self.cars[self.cars.index(obj) - 1], 20)


        # 收敛判断
        if abs(obj.a_x - self.cars[0].a_x) > 0.5:
            self.convergence = False
    def append(self,car:Car):
        if car not in self.cars: Delegation.append(self,car)

    def back(self,car:Car):
        if car in self.cars: Delegation.back(self,car)


class CAVRoad(Road):
    def __init__(
            self,
            graph:Optional[Interface.GraphBase],
            cars:list[Car]=(),
            stop_line:float=200,
            start_id:int=0,
            end_id:int=0,
    ):
        super().__init__(graph,cars)
        self.graph = graph
        self.start_id = start_id
        self.end_id = end_id

        # 线性变换到当前路的start_point
        self.tsf_start = RigidBody(p_x=self.graph.points[self.start_id].x,p_y=self.graph.points[self.start_id].y).transform(self.graph.road_basic[self.start_id,self.end_id])
        # 线性变换到当前路的end_point
        self.tsf_end = RigidBody(p_x=self.graph.points[self.end_id].x,p_y=self.graph.points[self.end_id].y).transform(self.graph.road_basic[self.start_id,self.end_id])

        # 起始点，加上起始路口半径，即为道路的x0
        self.start_x = self.tsf_start.p_x + self.graph.corssing_radius[self.start_id]
        # 起始点，减去lane*3，即为道路的y0
        self.start_y = self.tsf_start.p_y - ROAD_WIDTH/2


        # 停止线，cutlength
        self.stop_line = self.graph.cut_length[self.start_id,self.end_id]


        self.to_lane = td.AlignLane(ay=0.6,by=4)

        # 线性躲避位置
        self.avoid_l = td.LinearAvoidance(k=2,ax=1,bx=1.5,ay=2,by=5,safe_distance=10)
        self.avoid = td.RearEndedAvoidance(k=2,ax=20,bx=1.5,ay=2,by=5,safe_distance=500)

        # ax对变道时让速的程度 ， ay横向躲避高速目标
        self.avoid_v = td.VectorAvoidence(k=4,ax=40,bx=0,ay=4,by=0,safe_distance=200)

        # ax对变道时让速的程度
        self.avoid_a = td.AccAvoidence(k=1,ax=10,bx=0,ay=10,by=0,safe_distance=10)

        # 托管给CAVLane调控的对象
        self.cavs = [CAVLane(graph, [], self.stop_line, lane) for lane in range(3)]


        # 变道车辆，仅在初始化和新增车辆时添加
        # self.inchange=[car for car in self.cars if car.get_lane()!=car.obj_lane]
        self.inchange=[]



    def simulate(self,dt:float=0.1):
        """
        此委托，严格符合transfor传入，和back的返回
        调用子委托时，忽略cars的改动
        :param dt:
        :return:
        """
        for car in self.all_cars:
            car.vector_basis = (self.graph.road_basic[self.start_id,self.end_id])
            car.p_x = car.p_x - self.start_x
            car.p_y = car.p_y - self.start_y

        # 从变道中删除对象
        for car in self.inchange:
            # 误差小于允许值，直接忽略抖动，加入直行队列
            if abs(car.p_y - (car.obj_lane + 0.5) * Car.lane_length) < 0.2:
                # 从所有lane委托中删除
                for column in self.cavs:
                    column.back(car)

                # 添加到自己道路的委托
                self.cavs[car.get_lane()].append(car)
                self.inchange.remove(car)


        # 向变道中添加对象
        # 将每个车添加到自己的道路上
        for car in self.all_cars:
            if car.obj_lane != car.get_lane():
                if car not in self.inchange: self.inchange.append(car)
            lane = car.get_lane()
            if lane > 2:continue
            self.cavs[lane].append(car)






        # 添加变道车辆到所有lane托管
        # 三者共同托管
        for car in self.inchange:
            if car not in self.cars: continue
            for column in self.cavs:
                column.append(car)
            self.send(car)






        # cav计算车辆加速度(all car)(保持距离，对齐车道)
        for lane,CAVc in enumerate(self.cavs):
            CAVc.simulate(dt=dt,light = True)


        # 计算变道车辆，将会清空车的a_y
        for car in self.inchange:
            self.cavs[car.obj_lane].simulate_one(dt=dt,obj=car,light=True)

            # 抑制y
            # car.a_y = (self.to_lane(car,car.obj_lane) < (self.avoid_v.increment(car,self.cars) + self.avoid_a(car,self.cars))).a_y

        # 模拟车辆
        for car in self.all_cars:
            car.simulate(dt=dt)





        # 超过停止线，从此托管中移除，并返回父托管
        res = []
        for car in self.all_cars:
            if car.p_x > self.stop_line:
                res.append(car)
                if car in self.inchange: self.inchange.remove(car)
                for column in self.cavs:
                    column.back(car)


        for car in self.all_cars:
            car.p_x = car.p_x + self.start_x
            car.p_y = car.p_y + self.start_y
            car.vector_basis = (np.array([[1,0],[0,1]]))



        for car in res:
            self.back(car)
        return res
