from abc import abstractmethod


class CarsBase:
    @abstractmethod
    def generate_a(self)->[float,float]:
        pass

    @property
    @abstractmethod
    def car_positions(self):
        pass

    @property
    @abstractmethod
    def car_angles(self):
        pass

    @abstractmethod
    def simulate(self,dt:float=0):
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

