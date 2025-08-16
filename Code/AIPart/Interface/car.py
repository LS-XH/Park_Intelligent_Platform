from abc import abstractmethod, ABC
from typing import Tuple, Optional, ClassVar
from Interface.physics import RigidBody
from Interface.graph import GraphBase

def road_car(default_lane:int = 0):
    def decorator(cls):
        @property
        def obj_lane(self):
            return cls.__obj_lane
        @obj_lane.setter
        def obj_lane(self, value):
            cls.__obj_lane = value

        cls.__obj_lane = default_lane
        cls.obj_lane = obj_lane

        return cls

    return decorator

def crossing_car(default_from_id:int = -1,defult_cross_id:int = -1,default_to_id:int = -1):
    def decorator(cls):
        @property
        def from_id(self):
            return cls.__from_id
        @from_id.setter
        def from_id(self, value):
            cls.__from_id = value
        @property
        def cross_id(self):
            return cls.__cross_id
        @cross_id.setter
        def cross_id(self, value):
            cls.__cross_id = value
        @property
        def to_id(self):
            return cls.__to_id
        @to_id.setter
        def to_id(self, value):
            cls.__to_id = value

        cls.__from_id = default_from_id
        cls.__cross_id = defult_cross_id
        cls.__to_id = default_to_id

        cls.from_id = from_id
        cls.cross_id = cross_id
        cls.to_id = to_id
        return cls
    return decorator


@road_car(default_lane = -1)
@crossing_car(default_from_id = -1,default_to_id = -1)
class Car(RigidBody):
    car_length=4
    min_distance=1
    lane_length=3

    from_id:ClassVar[int]
    cross_id:ClassVar[int]
    to_id: ClassVar[int]

    def __init__(self,lane=0,x=0,y=0,id="",base=None,from_id:int = 0,to_id:int = 0):
        super(Car,self).__init__(p_x=x,p_y=y,vector_basis=base)
        self.id=id
        self.v_x = 10
        self.text = ""

        # 用于Road
        self.__obj_lane=lane

        # 用于Crossing
        self.__from = from_id
        self.__to = to_id


    @property
    def obj_lane(self)->int:
        """
        此车辆的目标车道
        :return:
        """
        return self.__obj_lane

    def lane2py(self,lane):
        """
        转换道路序号为其y轴的值
        :param lane:
        :return:
        """
        return (lane+0.5)*Car.lane_length

    def get_lane(self):
        """
        此车辆所在的车道
        :return:
        """
        return int(self.p_y/Car.lane_length)

    def on_lane(self,delta = 0.2)->Optional[int]:
        """
        是否在某车道中心，如在，则返回车道，不在则返回none
        :param delta: 允许的最大误差
        :return:
        """
        lane = self.get_lane()
        if abs(self.lane2py(lane)-self.p_y) < delta:
            return lane
        return None

    def zero_a(self):
        """
        清空加速度数据
        :return:
        """
        self.a_x=0
        self.a_y=0

    def simulate(self, dt: float = 0):
        super(Car,self).simulate(dt)
        self.zero_a()
class Delegation:
    def __init__(self,cars:[]):
        self.cars=[]
        self.all_cars = []

        for car in cars:
            self.append(car)

        self.delegate = []

    def append(self,car:Car):
        """
        新增元素,（接收）
        :param car:
        :return:
        """
        if car not in self.cars:
            self.cars.append(car)
            self.all_cars.append(car)

    def back(self,car:Car):
        """
        返回父委托的需要调用函数
        :param car:
        :return:
        """
        if car in self.cars:
            self.cars.remove(car)
            self.all_cars.remove(car)


    def send(self,transfer:Car):
        """
        转移到子委托后，调用此函数，从空闲对象中删除
        :param transfer:
        :return:
        """
        self.cars.remove(transfer)


    def transfer(self,transfer:Car,obj_delegation:'Delegation'):
        """
        委托间对象传递
        在当前委托，将会把对象从空闲对象中删除
        在子委托中，将会新增对象
        :param transfer:
        :param obj_delegation:
        :return:
        """
        if transfer in self.cars:
            obj_delegation.append(transfer)
            self.send(transfer)



    def simulate(self):
        pass
class Road(Delegation):
    def __init__(self,graph:Optional[GraphBase],cars:list[Car]):
        Delegation.__init__(self,cars)
        self.graph = graph
        self.cars = cars

    def get_cars(self,lane:int)->list[Car]:
        """
        获取在此车道上的车辆，只要中心在此车道的范围内即可
        :param lane:
        :return:
        """
        res = []
        for car in self.cars:
            if car.get_lane()==lane:
                res.append(car)
        return res
    def on_lane(self,lane:int,delta = 0.2)->list[Car]:
        """
        正在车道上行驶的车辆，要求误差小于delta
        :param lane:
        :param delta:
        :return:
        """
        res = []
        for car in self.cars:
            if abs(car.p_y - (lane+0.5)*Car.lane_length)<delta:
                res.append(car)

        return res
    def obj_lane(self,lane:int)->list[Car]:
        """
        目标车道为lane的车辆
        :param lane:
        :return:
        """
        res = []
        for car in self.cars:
            if car.obj_lane == lane:
                res.append(car)

        return res

    def will_to(self,lane:int)->list[Car]:
        """
        将要去某条车道的车辆
        :param lane:
        :return:
        """
        res = []
        for car in self.cars:
            # 同向，且相差为1
            if (car.get_lane() - car.obj_lane)*(car.get_lane() - lane) > 0 and abs(car.get_lane() - lane) == 1:
                res.append(car)
        return res

class CarsBase(ABC):
    @property
    @abstractmethod
    def car_positions(self):
        pass

    @abstractmethod
    def simulate(self, dt: float = 0):
        """
        模拟一帧，获取车辆属性
        :param dt: 一帧的实际时间
        :return: 一个字典
        {
            id1:{p_x,p_y,v_x,v_y,angle}

            id2:{p_x,p_y,v_x,v_y,angle}

            id3:{p_x,p_y,v_x,v_y,angle}

            ...
        }
        """
        pass


