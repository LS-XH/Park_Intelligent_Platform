from abc import abstractmethod

from Interface.physics import RigidBody



class Tendency:
    """
    趋势模块
    """
    def __init__(self):
        return
    def __call__(self,obj:RigidBody)->RigidBody:
        return obj+self.increment(obj)

    @abstractmethod
    def increment(self,obj:RigidBody)->RigidBody:
        pass

class Forward(Tendency):
    def __init__(
            self,
            scale:float=1,
            cruise_speed:float=1
    ):
        self.__scale = scale
        self.__cruise_speed = cruise_speed

        super().__init__()
        return
    @staticmethod
    def forward_av(scale:float=1,cruise_speed:float=1):
        """
        前行趋势
        :param scale: 前行趋势的规模
        :param cruise_speed: 巡航速度
        :return:
        """
        return scale

    def increment(self,obj:RigidBody):
        return RigidBody(a_x=self.forward_av(self.__scale,self.__cruise_speed))


class Align(Tendency):
    def __init__(
            self,
            target_car: RigidBody,
            r_h,
            ah: float = 1,
            bh: float = 1
    ):
        """
        横向对齐前方的一个目标
        :param target_car: 对齐的目标
        :param r_h: 期望横向偏移量
        :param ah: 邻接权重系数
        :param bh: 速度阻尼系数
        :return:
        """
        super().__init__()
        
        self.__ah = ah
        self.__bh = bh
        self.__r_h = r_h
        self.__target_car = target_car

    def increment(self,obj:RigidBody):
        return RigidBody(
            a_x = self.__ah * ((self.__target_car.p_y - obj.p_y - self.__r_h) + self.__bh * (self.__target_car.v_y - obj.v_y))
        )
    
class Avoidance(Tendency):
    def __init__(
            self,
            target_car: RigidBody,
            k: float = 1,
            ah: float = 1,
            bh: float = 1,
            av: float = 1,
            bv: float = 1,
            safe_distance: float = 3,
            epsilon: float = 0.01
    ):
        """
        躲避障碍物
        :param target_car:
        :param k: 整体权重
        :param ah: 纵向邻接权重系数
        :param bh: 纵向速度阻尼系数
        :param av: 横向邻接权重系数
        :param bv: 横向速度阻尼系数
        :param safe_distance: 安全距离
        :param epsilon: 防止分母为0的增量
        :return:
        """
        super().__init__()
        self.__target_car = target_car
        self.__k = k
        self.__ah = ah
        self.__bh = bh
        self.__av = av
        self.__bv = bv
        self.__safe_distance = safe_distance
        self.__epsilon = epsilon
        return
    def increment(self,obj:RigidBody):
        a_c,a_h=0,0
        # 纵向挤压
        if abs(self.__target_car.p_x-obj.p_x)<self.__safe_distance:
            a_c = self.__av*((self.__target_car.p_x-obj.p_x+self.__epsilon-self.__safe_distance)-self.__bv*(self.__target_car.v_x-obj.v_x))

        # 横向躲避
        if abs(self.__target_car.p_y-obj.p_y)<self.__safe_distance:
            a_h = self.__ah*((self.__target_car.p_y-obj.p_y-self.__safe_distance)-self.__bh*(self.__target_car.v_y-obj.v_y))

        return RigidBody(
            a_x=self.__k*a_c,
            a_y=self.__k*a_h,
        )