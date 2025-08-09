from Interface import GraphBase
import math
import numpy as np
import json

class Graph(GraphBase):
    green_light=20

    def __init__(self):
        self.__points=[]
        self.__points_id={}
        self.__edges=[]

    def initialize_car(self):
        """
        初始化车的位置
        :return:
        """
        pass

    def initialize_crowd(self):
        """
        初始化人的位置
        :return:
        """
        pass


    def nodes_position(self)->np.matrix:
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        positions=[[p.x,p.y] for p in self.__points]
        return np.matrix(positions)

    def length(self)->np.matrix:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        n=len(self.__points)
        length_mat=np.zeros((n,n))
        for edge in self.__edges:
            i=self.__points_id[edge.start_id]
            j=self.__points_id[edge.end_id]
            length_mat[i,j]=edge.length
        return np.matrix(length_mat)

    def weight(self)->np.matrix:
        """
        权重邻接矩阵
        :return: dim=2
        """
        pass

    def degree(self)->np.matrix:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        pass

    def limit_speed(self)->np.matrix:
        n=len(self.__points)
        speed_mat=np.zeros((n,n))
        for edge in self.__edges:
            i=self.__points_id[edge.start_id]
            j=self.__points_id[edge.end_id]
            speed_mat[i, j]=edge.limit_speed
        return np.matrix(speed_mat)


    def traffic_light(self)->np.matrix:
        light_mat=np.zeros((len(self.__points),len(self.__points)))
        for end in range(len(self.__points)):
            count = 0
            for start in range(len(self.__points)):
                if self.degree



    def get_light(self,start_id:str="",end_id:str="")->float:
        """
        获取红绿灯所剩的时间，如果为负数，则是绿灯，其绝对值为所剩的时间
        :param start_id: 起始点的名称id
        :param end_id: 末端点（路口）的名称id
        :return:
        """
        pass
        # self.traffic_light[self.__points_id[start_id],self.__points_id[end_id]]


    def add_point(self,id:str,x:float,y:float,type:PointType=PointType.crossing):
        self.__points.append(Point(id,x,y,type=type))
        self.__points_id[id]=len(self.__points_id)
        return True

    def add_edge(self,start_id:str,end_id:str,length:float,degree:int,limit_speed:float):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception("")

        self.__edges.append(Edge(start_id,end_id,length,degree,limit_speed))

    def load_json(self,path:str):
        with open(path,'r') as f:
            data=json.load(f)
        for point_data in data['points']:
            id=str(point_data['id'])
            x=float(point_data['x'])
            y=float(point_data['y'])
            type=point_data['type']
            self.add_point(id,x,y,type=type)

        for edge_data in data['edges']:
            start_id=str(edge_data['start_id'])
            end_id=str(edge_data['end_id'])
            degree=int(edge_data['degree'])
            limit_speed=float(edge_data["limit_speed"])
            start_point=self.__points[self.__points_id[start_id]]
            end_point=self.__points[self.__points_id[end_id]]
            length=math.sqrt(
                (end_point.x-start_point.x)**2+
                (end_point.y-start_point.y)**2
            )
            self.add_edge(start_id,end_id,length,degree,limit_speed)