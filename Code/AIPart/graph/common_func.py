import json
import random


def read_json_data(path):
    """
    读取json文件并返回Python字典

    :param path:    json文件路径
    :return:        Python字典对象
    """

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()

    return data


def select_by_probability(prob_dict):
    """
    根据概率字典选择key

    :param prob_dict: 概率字典，格式为 {key: 概率值, ...}
    :return: 随机选择的key
    """

    keys = list(prob_dict.keys())
    probabilities = list(prob_dict.values())
    selected = random.choices(keys, weights=probabilities, k=1)[0]

    return selected


def get_random_end_points(start_point, edges):
    """
    根据起始点和所有路线，随机选择一个终点

    :param start_point:      起点
    :param edges:               所有路线
    :return:                    终点ID
    """
    end_points = []
    for edge, v in edges.items():
        start_point_id = edge.split('->')[0]
        if start_point_id == start_point:
            end_points.append(edge.split('->')[1])

    random_next_point_id = random.choice(end_points)

    return random_next_point_id


def generate_cars_list(num_cars):
    """
    根据城市热点概率随机生成车辆的分布列表

    :param num_cars:    目标生成车的数量
    :return:            字典，表示每一辆车的绝对坐标
    """

    hot_data = read_json_data('./hot_data.json')
    graph = read_json_data('./data.json')
    points = graph['points']
    edges = graph['edges']

    cars = {}
    for i in range(num_cars):
        selected_point = select_by_probability(hot_data)
        random_next_point_id = get_random_end_points(selected_point, edges)

        x1 = points[selected_point]['x']
        y1 = points[selected_point]['y']
        x2 = points[random_next_point_id]['x']
        y2 = points[random_next_point_id]['y']

        random_percent = random.random()

        x = x1 + (x2 - x1) * random_percent
        y = y1 + (y2 - y1) * random_percent

        cars[i] = {'x': x, 'y': y}

    return cars


def generate_people_list(num_people):
    """

    :param num_people:
    :return:
    """

    hot_data = read_json_data('./hot_data.json')
    graph = read_json_data('./data.json')
    points = graph['points']

    peoples = []
    for i in range(num_people):
        selected_point = select_by_probability(hot_data)
        selected_point_x = points[selected_point]['x']
        selected_point_y = points[selected_point]['y']

        dx = random.gauss(selected_point_x, 20)
        dy = random.gauss(selected_point_y, 20)
        coord = (dx, dy)

        peoples.append(coord)

    return peoples


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    graph = read_json_data('./data.json')
    points = graph['points']

    my_crowd = generate_people_list(3000)
    for coord in my_crowd:
        plt.scatter(coord[0], coord[1], s=20, color='b')

    for id, info in points.items():
        plt.scatter(info['x'], info['y'], s=100, color='r')

    plt.show()