from typing import Optional

from sympy.physics.units.definitions import curie
from typing_extensions import Optional
import numpy as np
import torch
import random

from Interface.physics import RigidBody
from car.cavcar import Car
from car.road import Road, CAVRoad
from car.crossing import Crossing
from Interface.car import CarsBase,Delegation
from Algorithm.bidirectional_dijkstra import bidirectional_dijkstra as find_road


import Interface
import car.tendency as td
from graph import Graph


class Cars(Delegation,CarsBase):
    def __init__(self,graph:Graph,cars:list[Car]):
        Delegation.__init__(self,cars)



        self.path = []



        self.graph = graph

        self.road = self.graph.degree.copy()

        self.road_delegation = np.ndarray(shape=(self.road.shape[0],self.road.shape[0]),dtype=Road)
        for row in range(self.road.shape[0]):
            self.crossing_delegation = Crossing(row)

            for col in range(self.road.shape[0]):
                if self.road[row][col] != 0:
                    self.road_delegation[row,col] = CAVRoad(graph = self.graph,cars=[])


        self.crossing_delegation = np.ndarray(shape=(len(self.road)),dtype=Road)





    def get_road(self,car:Car,distance_delta:float = 2)-> Optional[tuple[int, int]]:
        """
        获取一个车所在的道路，如果不在任何道路上则返回None
        :param car: 车辆
        :param distance_delta: 允许误差范围
        :return: 一个元组，为(start_id,end_id)，如果不在任何路上，返回None
        """
        for start_id,row in enumerate(self.graph.degree):
            for end_id,item in enumerate(row):
                if item != 0:
                    p1 = np.array([self.graph.points[end_id].x,self.graph.points[end_id].y])
                    p2 = np.array([self.graph.points[start_id].x,self.graph.points[start_id].y])

                    # 差的方向相同：点差点积
                    if np.dot(p1-car.position.flatten(),p1-p2) > 0:

                        # 法向量为顺时针，即为右车道的垂直方向
                        n = np.array([(p1-p2)[1],-(p1-p2)[0]])
                        distance = np.dot(p1-car.position.flatten(),n)/np.linalg.norm(n)

                        # 计算点到直线距离，小于2
                        if abs(distance)<Car.lane_length + distance_delta:
                            return (end_id,start_id) if distance > 0 else (start_id,end_id)

        return None

    def simulate(self,dt=0.01):
        # 通过路径，将闲置的车辆添加到对应道路的委托
        for car in self.cars:
            path = self.path[self.all_cars.index(car)]
            self.transfer(car,self.road_delegation[path[0],path[1]].item())


        # 执行road托管
        for s_id,row in enumerate(self.road_delegation):
            for e_id,road in enumerate(row):
                # 执行委托
                backs:[] = road.simulate(dt)

                # 获取完成委托，返回出来的车
                for back_car in backs:
                    car_id = self.all_cars.index(back_car)

                    # 添加到路口托管
                    back_car.from_id = self.path[car_id][0]
                    back_car.cross_id = self.path[car_id][1]
                    back_car.to_id = self.path[car_id][2]

                    self.crossing_delegation[self.path[car_id][1]].recieve(back_car)

                    # 已从end中出来，所以删除start，end将成为下次的start（删除走完的路径部分）
                    del self.path[car_id][0]




        # 执行crossing托管
        for crossing in self.crossing_delegation:
            # 执行委托
            backs: [] = crossing.simulate(dt)

            # 获取已经出来的车
            for back_car in backs:
                car_id = self.all_cars.index(back_car)

                # 重新添加到闲置车辆
                self.cars.append(back_car)





    def add_car(self,car:Car,end_id):
        # 新增车辆
        if car not in self.all_cars:
            road = self.get_road(car)
            if road is None: raise Exception("Car is not on any road")

            # 新增车到列表
            self.append(car)
            # 新增路径
            self.path.append(find_road(self.graph.length,road[1],end_id)[0])



        else:
            return

    @property
    def car_positions(self):
        res = []
        for car in self.all_cars:
            car.transform()
            res.append({
                "x":car.position.p_x,
                "y":car.position.p_y,
                "angle":np.degrees(np.arctan(car.velocity.p_y/car.velocity.p_x))
            })

        return res










