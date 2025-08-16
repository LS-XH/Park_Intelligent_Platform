from car.car import Cars
from graph import Graph
from load_frame import *

graph = Graph()
init_cars = initialize_cars(30)
init_cars_list = cars_to_calculate(init_cars)


cars = Cars(graph, init_cars_list)



for i in range(100):
    cars.simulate(dt=0.1)
    print(cars.car_positions)