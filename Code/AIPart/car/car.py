from typing import Optional

from numpy.ma.core import shape
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
from graph.graph import LANE_WIDTH,ROAD_WIDTH

import time


class Cars(Delegation,CarsBase):
    def __init__(self,graph:Graph,cars:list):
        self.graph = graph
        self.path = []

        Delegation.__init__(self,[])

        for car in cars:
            self.add_car_by_road(car[0],car[1],car[2],car[3],car[4])


        # for car in cars:
        #     self.add_car(Car(x=car[0],y=car[1],id=str(len(self.all_cars))),car[2])



        self.road = self.graph.degree.copy()


        # 初始化 road
        self.road_delegation = np.ndarray(shape=(self.road.shape[0],self.road.shape[0]),dtype=Road)
        # 初始化crossing
        self.crossing_delegation = np.ndarray(shape=(len(self.road)),dtype=Road)
        for row in range(self.road.shape[0]):
            self.crossing_delegation[row] = Crossing(point_id=row,graph=self.graph,cars = [])

            for col in range(self.road.shape[0]):
                if self.road[row][col] != 0:
                    self.road_delegation[row,col] = CAVRoad(graph = self.graph,cars=[],start_id=row,end_id=col)







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
        for car in self.all_cars:
            if car not in self.cars:continue
            path = self.path[self.all_cars.index(car)]
            car.obj_lane=self.graph.crossing_turn[path[1],path[0],path[2]]["lane"]
            self.transfer(car,self.road_delegation[path[0],path[1]])

        # 执行road托管
        for s_id,row in enumerate(self.road_delegation):
            for e_id,road in enumerate(row):
                if road is None:continue
                # 执行委托
                backs:[] = road.simulate(dt)

                # 获取完成委托，返回出来的车
                for back_car in backs:
                    car_id = self.all_cars.index(back_car)

                    # 添加到路口托管
                    back_car.from_id = self.path[car_id][0]
                    back_car.cross_id = self.path[car_id][1]
                    back_car.to_id = self.path[car_id][2]

                    self.transfer(self.crossing_delegation[self.path[car_id][1]],back_car)

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





    def add_car(self,car:Car,destination,road=None):
        # 新增车辆
        if car not in self.all_cars:
            if road is None: road = self.get_road(car)
            if road is None: raise Exception("Car is not on any road")

            # 新增车到列表
            self.append(car)
            # 新增路径
            self.path.append([road[0]]+find_road(self.graph.length,road[1],destination))
        else:
            return


    def add_car_by_road(self,s_id,e_id,lane,process,destination):
        point = RigidBody(self.graph.points[s_id].x,self.graph.points[s_id].y)
        point.transform(self.graph.road_basic[s_id,e_id])
        # 相对道路的x，路口点加上长度*进度
        point.p_x += self.graph.length[s_id,e_id] * process
        # 相对道路的y，路口点减上lane的量
        point.p_y -= ROAD_WIDTH/2 - (lane+0.5) * LANE_WIDTH
        self.add_car(car = Car(x=point.p_x,y=point.p_y,base=point.vector_basis,id = str(len(self.all_cars))).transform(),destination=destination,road=[s_id,e_id])

    @property
    def car_positions(self):
        res = []
        for car in self.all_cars:
            res.append({
                "x":car.p_x,
                "y":car.p_y,
                "theta":np.degrees(np.arctan(car.v_y/car.v_x if car.v_x!=0 else 0)).item()
            })

        return res

    @property
    def flow_statistics(self):
        """
        获取当前tick的点的车辆数量邻接矩阵
        :return:
        """
        res = np.zeros(shape=(len(self.road_delegation),len(self.road_delegation)),dtype=int)

        for row,del_r in enumerate(self.road_delegation):
            for col,item in enumerate(del_r):
                res[row,col] = len(item.all_cars)


        return res











