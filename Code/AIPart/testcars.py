from car import Cars
from graph import Graph
from load_frame import initialize_cars, cars_to_calculate

graph = Graph()
init_cars = initialize_cars(30)
init_cars_list = cars_to_calculate(init_cars)
CARS = Cars(graph, init_cars_list)

for _ in range(1000):
    CARS.simulate(dt=0.1)
    print(CARS.car_positions)