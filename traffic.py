import networkx as nx
import pandas as pd
from collections import defaultdict
import numpy as np
from itertools import chain


def initialize_queues(cars_df):
    return cars_df.groupby('location')['car_num'].apply(list).to_dict(into=defaultdict(list))


def parse_file(filename):
    graph = nx.DiGraph()

    n = 0
    roads = []
    paths = []
    street_to_length = {}
    for line in open(filename, 'r'):
        if n == 0:
            T, n_intersections, n_roads, n_cars, n_points = map(int, line.split(' '))
        elif n <= n_roads:
            n1, n2, street_name, weight = line.split(' ')
            roads.append(dict(
                node_start=int(n1),
                node_end=int(n2),
                street_name=street_name,
                weight=int(weight)
            ))
            street_to_length[street_name] = int(weight)
            graph.add_edge(int(n1), int(n2), weight=int(weight), street_name=street_name)
        elif n <= n_roads + n_cars:
            paths.append(line)
        n += 1

    roads_df = pd.DataFrame.from_dict(roads)

    cars = []
    for path in paths:
        num, rest = path.split(' ', maxsplit=1)
        num = int(num)
        path_streets = rest.strip('\n').split(' ')
        cars.append(dict(
            num_streets=num,
            path_streets=path_streets
        ))
    cars_df = pd.DataFrame.from_dict(cars)
    cars_df['index'] = 0
    cars_df['location'] = cars_df['path_streets'].apply(lambda x: x[0])
    cars_df['distance_left'] = 0
    cars_df['total_distance_left'] = cars_df['path_streets'].apply(lambda x: sum(map(lambda y: street_to_length[y], x )) )
    cars_df['car_num'] = cars_df.index
    cars_df['done'] = False

    return graph, roads_df, cars_df, T, street_to_length


def simple_schedule(graph):
    schedules = []
    intersection_schedules = {}
    for node in graph.nodes:
        schedule = []
        intersection_schedule = []
        for u, v, data in graph.in_edges(node, data=True):
            schedule.append([data['street_name'], 1])
            intersection_schedule.append([data['street_name'] * 1])
        schedules.append([node, len(schedule), schedule])
        intersection_schedules[node] = intersection_schedule
    return schedules, intersection_schedules


def schedule_by_frequency(graph, cars_df):
    from collections import Counter
    all_paths = chain(*cars_df['path_streets'].tolist())
    counts = Counter(all_paths)

    schedules = []
    intersection_schedules = {}
    for node in graph.nodes:
        schedule = []
        intersection_schedule = []
        for u, v, data in graph.in_edges(node, data=True):
            street_name = data['street_name']
            green_time = counts[street_name]
            schedule.append([street_name, green_time])
            intersection_schedule.append([data['street_name'] * green_time])
        schedules.append([node, len(schedule), schedule])
        intersection_schedules[node] = intersection_schedule
    return schedules, intersection_schedules



def get_green_lights(intersection_schedules, t):
    return [v[t % len(v)][0] for v in intersection_schedules.values()]


def add_to_queues(queues, queues_to_add):
    for car_num, street_name in queues_to_add.items():
        queue = queues[street_name]
        queue.append(car_num)


graph, roads_df, cars_df, T, street_to_length = parse_file(r'hashcode.in')
street_to_node = roads_df[['street_name', 'node_end']].set_index('street_name').to_dict()['node_end']
node_to_streets = roads_df[['street_name', 'node_end']].groupby('node_end')['street_name'].apply(list).to_dict()


# schedules, intersection_schedules = simple_schedule(graph)
schedules, intersection_schedules = schedule_by_frequency(graph, cars_df)

def simulation(T, cars_df, street_to_length ):
    queues = initialize_queues(cars_df)
    history = {}
    metrics = []
    for t in range(T):
        print(t)
        history[t] = cars_df.copy()
        #     print(queues)
        # if t > 50:
        #     break
        people_not_at_lights_idx = cars_df['distance_left'] > 1
        people_approaching_lights = cars_df['distance_left'] == 1
        people_finishing = np.logical_and(cars_df['distance_left'] == 1, cars_df['index'] == (cars_df['num_streets'] - 1))

        #     print(t, len(people_not_at_lights))
        green_lights = get_green_lights(intersection_schedules, t)
        #     print(queues)
        starting_cars = [queues[green_light].pop(0) for green_light in green_lights if queues.get(green_light)]
        #     print('starting_cars')
        #     print(starting_cars)
        #     print(queues)

        #     print('starting_cars', len(starting_cars))
        #     print('people_approaching_lights', people_approaching_lights)

        starting_cars_idx = np.logical_and(cars_df['car_num'].isin(starting_cars), ~cars_df['done'])
        if starting_cars_idx.any():
            #         print(cars_df.loc[starting_cars_idx, :])
            cars_df.loc[starting_cars_idx, 'index'] += 1
            street_names = cars_df.loc[starting_cars_idx, ['index', 'path_streets']].apply(lambda x: x['path_streets'][x['index']],
                                                                             axis=1)
            cars_df.loc[starting_cars_idx, 'location'] = street_names
            cars_df.loc[starting_cars_idx, 'distance_left'] = street_names.map(street_to_length) - 1
            cars_df.loc[starting_cars_idx, 'total_distance_left'] -= 1
            if pd.isnull(cars_df['distance_left']).any():
                print('error')
        #         print(cars_df.loc[starting_cars_idx, :])
        #         print('-------------')

        else:
            print('no one')

        people_starting_length_1 = starting_cars_idx & (cars_df['distance_left'] == 1)
        driving_idx = (people_not_at_lights_idx |
                                  people_approaching_lights |
                                                people_finishing)
        cars_df.loc[driving_idx, 'distance_left'] -= 1
        cars_df.loc[driving_idx, 'total_distance_left'] -= 1

        queuing_before = len(list(chain(*queues.values())))

        queues_to_add = cars_df.loc[(people_approaching_lights | people_starting_length_1),
                                    ['location']]['location'].to_dict()
        add_to_queues(queues, queues_to_add)
        queuing_after = len(list(chain(*queues.values())))

        cars_df.loc[people_finishing, 'done'] = True
        num_lights = len({k: v for k, v in queues.items() if v}.keys())
        max_red_light = max(list({k: len(v) for k, v in queues.items() if v}.values()) + [0])
        # print('num_lights', num_lights)
        # print('max_red_light', max_red_light)

        metrics.append(dict(
            driving = driving_idx.sum(),
            queuing_before = queuing_before,
            queuing_after = queuing_after,
            num_queues_to_add = len(queues_to_add),
            num_starting = starting_cars_idx.sum(),
            max_index = cars_df['index'].max(),
            min_index = cars_df['index'].min(),
            mean_index = cars_df['index'].mean(),
            # max_queues = max(map(len, queues.values())),
            num_done = cars_df['done'].sum(),
            max_intersection_queue = cars_df[cars_df['distance_left'] == 0]['location'].map(street_to_node).value_counts().max(),
        ))

    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df

metrics_df= simulation(T, cars_df, street_to_length )
print(metrics_df)

metrics_df[['driving', 'queuing_after']].plot()