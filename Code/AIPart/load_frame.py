import json
import math
import random

import numpy as np


# random.seed(42)


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_neighbor_node(current_node_id, edges):
    end_points = []
    for edge_info in edges:
        current_start = edge_info['start_id']
        current_end = edge_info['end_id']
        if current_start == current_node_id:
            end_points.append(edge_info['end_id'])
        elif current_end == current_node_id:
            end_points.append(edge_info['start_id'])

    return end_points


def generate_random_with_gap(start, stop, num, min_gap):
    """
    生成指定范围内的随机实数，确保任意两个数据间隔 ≥ min_gap，且所有数据在[start, stop]范围内
    :param start:       范围起始值
    :param stop:        范围结束值
    :param num:         数据数量
    :param min_gap:     最小间隔
    :return:            排序后的随机实数数组（满足间隔约束）
    """
    if num <= 0:
        raise ValueError("数据数量必须为正数")
    if min_gap < 0:
        raise ValueError("最小间隔不能为负数")
    if num == 1:
        return np.array([np.random.uniform(start, stop)])

    required_range = (num - 1) * min_gap
    total_available = stop - start
    if required_range > total_available:
        raise ValueError(f"无法生成数据：所需最小范围 {required_range} 超过实际范围 {total_available}")

    extra_space = total_available - required_range
    if extra_space > 0:
        if num > 2:
            splits = np.random.uniform(0, extra_space, size=num - 2)
            splits.sort()
            offsets = np.concatenate([[0], splits, [extra_space]])
            offsets = np.diff(offsets)
        else:
            offsets = np.array([extra_space])
    else:
        offsets = np.zeros(num - 1)

    data = [start]
    current = start
    for i in range(num - 1):
        step = min_gap + offsets[i]
        current += step
        if i == num - 2:
            current = min(current, stop)
        data.append(current)

    return np.array(data).tolist()


def initialize_cars(num_cars):
    cars = []
    map_data = load_json("./graph/data.json")
    points_data = map_data["points"]
    edges_data = map_data["edges"]

    # 在某一个路段生成多辆车
    random_point_id = random.randint(0, len(points_data) - 1)
    random_point_id = 0
    random_point = points_data[random_point_id]
    random_point_neighbor = find_neighbor_node(random_point_id, edges_data)
    random_point_neighbor = random.choice(random_point_neighbor)
    random_point_neighbor = 1
    start_x, start_y = random_point['x'], random_point['y']
    end_x, end_y = points_data[random_point_neighbor]['x'], points_data[random_point_neighbor]['y']
    start_theta = math.degrees(math.atan2(end_y - start_y, end_x - start_x)) % 360
    first_generate_num_cars = random.randint(10, 20)
    random_percent = generate_random_with_gap(0.1, 0.5, first_generate_num_cars, 0.01)

    for i in range(first_generate_num_cars):
        random_destination_id = random.randint(0, len(points_data) - 1)
        cars.append([random_point_id, random_point_neighbor, random.choice([0, 1, 2]), random_percent[i], random_destination_id])

    # 随机生成其他车辆
    other_generate_num_cars = num_cars - first_generate_num_cars
    for i in range(other_generate_num_cars):
        random_start_id = random.randint(0, len(points_data) - 1)
        random_point = points_data[random_start_id]
        random_end_id = find_neighbor_node(random_start_id, edges_data)
        random_end_id = random.choice(random_end_id)

        start_x, start_y = random_point['x'], random_point['y']
        end_x, end_y = points_data[random_end_id]['x'], points_data[random_end_id]['y']
        theta = math.degrees(math.atan2(end_y - start_y, end_x - start_x)) % 360
        random_percent = random.uniform(0.2, 0.8)

        car_x = start_x + (end_x - start_x) * random_percent
        car_y = start_y + (end_y - start_y) * random_percent
        random_destination_id = random.randint(0, len(points_data) - 1)

        cars.append([random_start_id, random_end_id, random.choice([0, 1, 2]), random_percent, random_destination_id])

    return cars


def cars_to_calculate(cars):
    return [(sublist[0], sublist[1], sublist[2], sublist[3], sublist[4]) for sublist in cars]


def cars_to_unity(cars):
    return {
        i: {
            "x": sublist[0],
            "y": sublist[1],
            "theta": sublist[2]
        } for i, sublist in enumerate(cars)
    }


if __name__ == '__main__':
    my_cars = initialize_cars(30)
    my_cars = cars_to_calculate(my_cars)
    print(my_cars)
    # from matplotlib import pyplot as plt
    #
    # my_cars = initialize_cars(30)
    # my_cars = cars_to_calculate(my_cars)
    # data = load_json("./graph/data.json")
    # points_data = data["points"]
    #
    # for point in points_data:
    #     x, y = point['x'], point['y']
    #     plt.scatter(x, y, c='r', s=20)
    #
    # for car in my_cars:
    #     x, y = car[0], car[1]
    #     plt.scatter(x, y, c='b', s=5)
    #
    # plt.show()
