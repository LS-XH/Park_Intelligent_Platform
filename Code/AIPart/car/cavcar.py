from sympy.physics.units.definitions import curie
from typing_extensions import Optional
import numpy as np
import torch
import random

from Interface.physics import RigidBody
from Interface.car import Road,Car
import Interface
import car.tendency as td






class CavGroup:
    def __init__(self, leader:Car, follows:list[Car]):
        self.leader = leader
        self.follows = follows
    def simulate(self):
        return



class Carsss(Road):
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        super().__init__(graph,cars)
        self.graph = graph
        self.cars = cars

        self.chatter = td.Chatter(0.4)
        self.forward = td.Forward(scale=0.001,bx=0)
        self.to_lane = td.AlignLane(ay=0.6,by=2)
        self.avoid = td.ComAvoidance(k=1,ax=0.1,bx=1.5,ay=2,by=5,safe_distance=10)

        self.brake = td.Brake(max_speed=5)
        self.cruise = td.Cruise(bx=0.01,cruise_speed=10)

    def simulate(self,dt:float=0.1):
        for car in self.cars:
            car.zero_a()

            if car.id == "0":
                a = 1
            if car.id == "1":
                a = 1
            if car.id == "2":
                a = 1
            if car.id == "3":
                a = 1
            if car.id == "4":
                a = 1
            if car.id == "5":
                a = 1
            if car.id == "6":
                a = 1
            if car.id == "8":
                a = 1
            if car.id == "9":
                a=1


            forward = self.forward(car)
            brake = self.brake(car)
            cruise = self.cruise(car)
            c_lane = self.to_lane.increment(car,obj_lane = car.obj_lane)
            chatter = self.chatter(car)

            if car.get_lane()==car.obj_lane:
                avoid = self.avoid.increment(car,self.get_cars(car.obj_lane)+self.will_to(car.obj_lane))
                forward = forward > avoid
                forward = forward < avoid
                car += cruise
            else:
                avoid = self.avoid.increment(car,self.cars)
                forward = forward > RigidBody(a_x=2)
                forward = forward < avoid
                c_lane = c_lane < avoid
                # car + brake

            car += forward
            car += c_lane

            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.id

            # car.text = "%.1f,%.1f\n%.1f,%.1f\n" % (car.v_x,car.v_y,car.a_x, car.a_y)

class Cars2(Road):
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        super().__init__(graph,cars)
        self.graph = graph
        self.cars = cars


        self.forward = td.Forward(scale=0.001,bx=0,lane_k=5)
        self.to_lane = td.AlignLane(ay=0.6,by=2)

        # 线性躲避位置
        self.avoid_l = td.LinearAvoidance(k=2,ax=1,bx=1.5,ay=2,by=5,safe_distance=10)
        self.avoid = td.RearEndedAvoidance(k=2,ax=20,bx=1.5,ay=2,by=5,safe_distance=500)

        # ax对变道时让速的程度 ， ay横向躲避高速目标
        self.avoid_v = td.VectorAvoidence(k=4,ax=40,bx=0,ay=4,by=0,safe_distance=200)

        # ax对变道时让速的程度
        self.avoid_a = td.AccAvoidence(k=1,ax=10,bx=0,ay=10,by=0,safe_distance=10)


        self.stop = td.DetermineStop(scale=2)

        self.cruise = td.Cruise(bx=0.01,cruise_speed=10)



        self.cavs = []

        self.all_car = cars.copy()
    def simulate(self,dt:float=0.1):
        #tendency
        for lane in range(3):
            #车道上无车辆
            if len(self.obj_lane(lane)) == 0:
                continue

            # 车道上还有闲杂车辆
            if sorted(self.get_cars(lane), key=lambda car: car.p_x) != sorted(self.obj_lane(lane),key=lambda car: car.p_x): continue

            # 判断目标为此到的车辆都在车道中心区域
            if sorted(self.on_lane(lane),key=lambda car: car.p_x) == sorted(self.obj_lane(lane),key=lambda car: car.p_x):
                self.cavs.append(CAVColumn(None, self.get_cars(lane), 700,lane, 0.5, 1.5))
                if lane == 1:
                    a=1
                self.cars = [car for car in self.cars if car.obj_lane != lane]






        for car in self.cars:
            car.zero_a()

            if car.id == "0":
                a=1
            if car.id == "1":
                a=1
            if car.id == "2":
                a=1


            forward = self.forward(car,car.get_lane())
            c_lane = self.to_lane.increment(car,obj_lane = car.obj_lane)

            # if car.get_lane() != car.obj_lane:forward += RigidBody(a_x=2)

            car += forward
            car += c_lane


        # supp
        for car in self.cars:
            if car.id == "1":
                a = 1
            if car.id == "2":
                a = 1
            if car.id == "3":
                a = 1
            if car.id == "4":
                a = 1
            if car.id == "5":
                a = 1
            if car.id == "6":
                a = 1
            if car.id == "7":
                a = 1

            res = RigidBody(a_x=car.a_x,a_y=car.a_y)

            avoid = self.avoid.increment(car, self.cars)
            avoid_l = self.avoid_l.increment(car, self.cars)

            avoid_v = self.avoid_v.increment(car, self.cars)
            avoid_a = self.avoid_a.increment(car, self.cars)
            cruise = self.cruise(car,car.get_lane())

            stop = self.stop.increment(car)



            if car.get_lane() == car.obj_lane:
                #三者引起的抑制，让速不让道，仅有x调整
                avoidence = RigidBody(a_x = avoid.a_x+avoid_v.a_x+avoid_a.a_x)
                # res += avoidence

                #定速巡航
                res += cruise

                # res += stop
            else:
                res += RigidBody(a_x = 2)
                #位置引起的抑制
                res = res < RigidBody(a_x = avoid_l.a_x)
                res = res < RigidBody(a_x = avoid.a_x)

                #躲避高速目标
                res = res > avoid_v
                res = res < avoid_v

                #y轴变道回避
                res = res < RigidBody(a_y = avoid_a.a_y)


            car.a_x = res.a_x
            car.a_y = res.a_y

            # if car.id == "9" and car.p_x>200:
            #     car.a_x = 0
            #     car.a_y = 0
            #     car.v_x = 0
            #     car.v_y = 0




            # if car.p_x >= 500:
            #     car.a_y = 0
            #     car.p_y = (car.obj_lane + 0.5) * Car.lane_length
            #
            #


        for car in self.cars:
            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.id

        for cav in self.cavs:
            cav.simulate(dt=dt)

class Cars21:
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        self.graph = graph
        self.cars = cars


        self.chatter = td.Chatter(0.4)
        self.forward = td.Forward(scale=0.001,bx=0)
        self.to_lane = td.AlignLane(ay=0.6,by=2)
        self.avoid = td.ComAvoidance(k=1,ax=0.1,bx=1.5,ay=2,by=5,safe_distance=10)

        self.avoid_v = td.VectorAvoidence(k=1,ax=2,bx=5,ay=2,by=5,safe_distance=10)

        self.brake = td.Brake(max_speed=5)
        self.cruise = td.Cruise(bx=0.01,cruise_speed=10)
    def simulate(self,dt:float=0.1):
        for car in self.cars:
            car.zero_a()

            if car.id == "2":
                a=1


            forward = self.forward(car)
            brake = self.brake(car)
            cruise = self.cruise(car)
            c_lane = self.to_lane.increment(car,obj_lane = car.obj_lane)
            chatter = self.chatter(car)


            avoid = self.avoid.increment(car, self.cars)
            avoid_v = self.avoid_v.increment(car, self.cars)

            if car.get_lane() == car.obj_lane:
                forward = forward > avoid
                forward = forward < avoid
                forward += RigidBody(a_x = avoid_v.a_x)
                car += cruise
            else:
                forward = forward > RigidBody(a_x=2)
                forward = forward < avoid_v

                c_lane = c_lane < avoid
                # car + brake

            car += forward
            car += c_lane

            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.id



class CAVColumn:
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car],stop_line,lane,ax:float=0.1,bx:float=4,ay:float=0.3,by:float=1.5):
        self.graph = graph
        self.cars = cars
        self.stop_line = stop_line
        self.lane = lane
        self.ax = ax
        self.bx = bx
        self.ay = ay
        self.by = by
        self.cars.sort(key=lambda car: car.p_x,reverse=True)

        for car in self.cars:
            car.p_y = (car.get_lane() + 0.5) * Car.lane_length
            car.v_y = 0

    def cav_x(self,obj:Car,target:Car,r_x):
        return self.ax * ((target.p_x - obj.p_x - r_x) + self.bx * (target.v_x - obj.v_x))

    def cav_l(self,obj:Car,r_px):
        delta = (r_px - obj.p_x)
        if delta > 30:
            delta = 30
        return self.ax * (delta + self.bx * (0 - obj.v_x))

    def cav_align(self,obj:Car,lane):
        return self.ay * ((obj.p_y - (lane+0.5) * Car.lane_length)+ self.by * (0 - obj.v_y))

    def simulate(self,dt:float=0.1):
        for i,car in enumerate(self.cars):
            car.a_x = 0
            car.a_y = 0
            if car.id == "2":
                a = 1

            # 对齐当前车道
            car.a_y = self.cav_align(car,self.lane)

            if i == 0:
                car.a_x = self.cav_l(car,self.stop_line)

            else:
                car.a_x = self.cars[0].a_x
                car.a_x += self.cav_x(car,self.cars[0],i*30)
                car.a_x += self.cav_x(car,self.cars[i-1],30)

        for car in self.cars:
            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.v_x



class CAV:
    def __init__(self, graph: Optional[Interface.GraphBase], cars: list[Car]):
        self.graph = graph
        self.cars = cars


        # p_x从大到小
        self.lanes = [[],[],[]]

        for car in cars:
            if len(self.lanes[car.obj_lane]) == 0:
                self.lanes[car.obj_lane].append(car)
            elif self.lanes[car.obj_lane][-1].p_x > car.p_x:
                self.lanes[car.obj_lane].append(car)
            else:
                for i,l in enumerate(self.lanes[car.obj_lane]):
                    if self.lanes[car.obj_lane][i].p_x < car.p_x:
                        self.lanes[car.obj_lane].insert(i,car)
                        break




        self.chatter = td.Chatter(0.1)
        self.forward = td.Forward(scale=3, bx=0.1)
        self.to_lane = td.AlignLane(ay=0.1, by=4)
        self.avoid = td.ComAvoidance(k=1, ax=10, bx=1.5, ay=0.5, by=1.5, safe_distance=30)

        self.brake = td.Brake(max_speed=5)
        self.cruise = td.Brake(max_speed=10)
        self.all_car = cars

    @staticmethod
    def cav(obj:Car,target:Car,r_x,r_y,ax:float=1,bx:float=1,ay:float=1,by:float=1):
        obj.a_x += ax * ((target.p_x - obj.p_x - r_x) + bx * (target.v_x - obj.v_x))
        obj.a_y += ay * ((target.p_y - obj.p_y - r_y) + by * (target.v_y - obj.v_y))


    def simulate(self, dt: float = 0.1):

        for l,lane in enumerate(self.lanes):
            for i,car in enumerate(lane):
                car.zero_a()
                if i == 0:
                    continue
                for tl,tlane in enumerate(self.lanes):
                    for ti,tcar in enumerate(lane):
                        #
                        # car.p_x = 10 * -(ti - i)
                        # car.p_y = 3 * -(tl - l+0.5)
                        if tcar == car:
                            continue
                        if i != 0:
                            # car.a_x+=lane[0].a_x
                            # car.a_y+=lane[0].a_y
                            a=1
                        self.cav(car,tcar,10*-(ti-i),3*-(tl-l),0.01,1.7,0.01,1.7)
                        a=1


                # car + chatter
                car.simulate(dt=dt)

                # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
                car.text = car.id

                # car.text = "%.1f,%.1f\n%.1f,%.1f\n" % (car.v_x,car.v_y,car.a_x, car.a_y)



class CarsA(Road):
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        super().__init__(graph,cars)
        self.graph = graph
        self.cars = cars


        self.forward = td.Forward(scale=0.001,bx=0,lane_k=5)
        self.to_lane = td.AlignLane(ay=0.6,by=2)

        # 线性躲避位置
        self.avoid_l = td.LinearAvoidance(k=2,ax=1,bx=1.5,ay=2,by=5,safe_distance=10)
        self.avoid = td.RearEndedAvoidance(k=2,ax=20,bx=1.5,ay=2,by=5,safe_distance=500)

        # ax对变道时让速的程度 ， ay横向躲避高速目标
        self.avoid_v = td.VectorAvoidence(k=4,ax=40,bx=0,ay=4,by=0,safe_distance=200)

        # ax对变道时让速的程度
        self.avoid_a = td.AccAvoidence(k=1,ax=10,bx=0,ay=10,by=0,safe_distance=10)


        self.stop = td.DetermineStop(scale=2)

        self.cruise = td.Cruise(bx=0.01,cruise_speed=10)



        self.cavs = []

        self.all_car = cars.copy()
    def simulate(self,dt:float=0.1):

        for car in self.cars:
            car.zero_a()

            if car.id == "0":
                a=1
            if car.id == "1":
                a=1
            if car.id == "2":
                a=1


            forward = self.forward(car,car.get_lane())
            c_lane = self.to_lane.increment(car,obj_lane = car.obj_lane)

            # if car.get_lane() != car.obj_lane:forward += RigidBody(a_x=2)

            car += forward
            car += c_lane


        # supp
        for car in self.cars:
            if car.id == "1":
                a = 1
            if car.id == "2":
                a = 1
            if car.id == "3":
                a = 1
            if car.id == "4":
                a = 1
            if car.id == "5":
                a = 1
            if car.id == "6":
                a = 1
            if car.id == "7":
                a = 1

            res = RigidBody(a_x=car.a_x,a_y=car.a_y)

            avoid = self.avoid.increment(car, self.cars)
            avoid_l = self.avoid_l.increment(car, self.cars)

            avoid_v = self.avoid_v.increment(car, self.cars)
            avoid_a = self.avoid_a.increment(car, self.cars)
            cruise = self.cruise(car,car.get_lane())

            stop = self.stop.increment(car)



            if car.get_lane() == car.obj_lane:
                #三者引起的抑制，让速不让道，仅有x调整
                avoidence = RigidBody(a_x = avoid.a_x+avoid_v.a_x+avoid_a.a_x)
                res += avoidence

                #定速巡航
                res += cruise

                # res += stop
            else:
                res += RigidBody(a_x = 2)
                #位置引起的抑制
                res = res < RigidBody(a_x = avoid_l.a_x)
                res = res < RigidBody(a_x = avoid.a_x)

                #躲避高速目标
                res = res > avoid_v
                res = res < avoid_v

                #y轴变道回避
                res = res < RigidBody(a_y = avoid_a.a_y)


            car.a_x = res.a_x
            car.a_y = res.a_y

            # if car.id == "9" and car.p_x>200:
            #     car.a_x = 0
            #     car.a_y = 0
            #     car.v_x = 0
            #     car.v_y = 0




            # if car.p_x >= 500:
            #     car.a_y = 0
            #     car.p_y = (car.obj_lane + 0.5) * Car.lane_length
            #
            #


        for car in self.cars:
            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.id

        for cav in self.cavs:
            cav.simulate(dt=dt)
