from Code.AIPart.Interface.people import PersonBase


class Crowd(PersonBase):


    def __init__(self, position:list, graph:any ):
        self.pos = position
        self.graph=graph

    def simulate(self, happened: list = None):

        pass


    def get_pos(self, node_id: int, ranges: int = 30) -> list:
        pass

    @property
    def position(self) -> list:
        return []


    def kill(self, node_id: int, ranges: int = 20):
        pass

    def get_emergency(self, happened: list = None):
        if happened:
            for happened in happened:
                e=666
        return []