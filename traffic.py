import copy
import pickle
import typing
from collections import Counter
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd

ALPHA = 0.8


def initialize_queues(
    cars_df: pd.DataFrame, streets: typing.List
) -> typing.Dict[int, typing.List[int]]:
    queues = {s: [] for s in streets}
    queues.update(cars_df.groupby("location")["car_num"].apply(list).to_dict())
    return queues

def parse_file(
    filename: str,
) -> typing.Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, int, typing.Dict[str, int]]:
    graph = nx.DiGraph()

    n = 0
    roads = []
    paths = []
    street_to_length = {}
    for line in open(filename, "r"):
        if n == 0:
            T, n_intersections, n_roads, n_cars, n_points = map(int, line.split(" "))
        elif n <= n_roads:
            n1, n2, street_name, weight = line.split(" ")
            roads.append(
                dict(
                    node_start=int(n1),
                    node_end=int(n2),
                    street_name=street_name,
                    weight=int(weight),
                )
            )
            street_to_length[street_name] = int(weight)
            graph.add_edge(
                int(n1), int(n2), weight=int(weight), street_name=street_name
            )
        elif n <= n_roads + n_cars:
            paths.append(line)
        n += 1

    roads_df = pd.DataFrame.from_dict(roads)

    cars = []
    for path in paths:
        num, rest = path.split(" ", maxsplit=1)
        num = int(num)
        path_streets = rest.strip("\n").split(" ")
        cars.append(dict(num_streets=num, path_streets=path_streets))
    cars_df = pd.DataFrame.from_dict(cars)
    cars_df["index"] = 0
    cars_df["location"] = cars_df["path_streets"].apply(lambda x: x[0])
    cars_df["distance_left"] = 0
    cars_df["total_distance_left"] = cars_df["path_streets"].apply(
        lambda x: sum(map(lambda y: street_to_length[y], x))
    )
    cars_df["car_num"] = cars_df.index
    cars_df["done"] = False

    return graph, roads_df, cars_df, T, street_to_length


def simple_schedule(
    graph: nx.DiGraph,
) -> typing.Dict[int, typing.List[typing.Tuple[str, int]]]:
    schedules = []
    intersection_schedules = {}
    schedules_dict = {}
    for node in graph.nodes:
        schedule = []
        intersection_schedule = []
        for u, v, data in graph.in_edges(node, data=True):
            schedule.append([data["street_name"], 1])
            intersection_schedule.append([data["street_name"] * 1])
        schedules.append([node, len(schedule), schedule])
        schedules_dict[node] = schedule
        intersection_schedules[node] = list(chain(*intersection_schedule))
    return schedules_dict  # schedules, intersection_schedules, schedules_dict


def schedule_by_frequency(
    graph: nx.DiGraph, cars_df: pd.DataFrame
) -> typing.Dict[int, typing.List[typing.Tuple[str, int]]]:

    all_paths = chain(*cars_df["path_streets"].tolist())
    counts = Counter(all_paths)

    schedules = []
    schedules_dict = {}
    intersection_schedules = {}
    for node in graph.nodes:
        schedule = []
        intersection_schedule = []
        for u, v, data in graph.in_edges(node, data=True):
            street_name = data["street_name"]
            green_time = counts[street_name]
            schedule.append([street_name, green_time])
            if green_time > 0:
                intersection_schedule.append([data["street_name"]] * green_time)
        schedules.append([node, len(schedule), schedule])
        schedules_dict[node] = schedule
        if len(list(chain(*intersection_schedule))):
            intersection_schedules[node] = list(chain(*intersection_schedule))
    return schedules_dict  # schedules, intersection_schedules, schedules_dict


def get_green_lights(
    intersection_schedules: typing.Dict[int, typing.List[str]], t: int
) -> typing.List[str]:
    return [v[t % len(v)] for v in intersection_schedules.values()]


def add_to_queues(
    queues: typing.Dict[int, typing.List[int]],
    queues_to_add: typing.Dict[int, typing.List[int]],
) -> None:
    for car_num, street_name in queues_to_add.items():
        queue = queues[street_name]
        queue.append(car_num)


def add_to_queues_np(
    queues: typing.Dict[int, typing.List[int]],
    queues_to_add_locations: np.ndarray,
    queues_to_add_carnums: np.ndarray,
) -> None:
    for car_num, street_name in zip(queues_to_add_carnums, queues_to_add_locations):
        queue = queues[street_name]
        queue.append(car_num)


def mutate_schedules(
    schedules_dict: typing.Dict[int, typing.List[typing.Tuple[str, int]]]
):
    schedules_dict = copy.deepcopy(schedules_dict)
    for node, schedules in schedules_dict.items():
        for street in schedules:
            if np.random.random() > ALPHA:
                street[1] = np.max(
                    [0, schedules_dict[0][0][1] + np.random.choice([1, -1])]
                )
    return schedules_dict


def schedules_dict_to_schedules_list(
    schedules_dict: typing.Dict[int, typing.List[typing.Tuple[str, int]]]
) -> typing.List[typing.Tuple[int, int, typing.List[typing.Tuple[str, int]]]]:
    schedules_list = []
    for node, schedules in schedules_dict.items():
        filtered_schedules = [s for s in schedules if s[1] != 0]
        if len(filtered_schedules) > 0:
            schedules_list.append([node, len(filtered_schedules), filtered_schedules])
    return schedules_list


def schedules_dict_to_intersection_schedules(
    schedules_dict: typing.Dict[int, typing.List[typing.Tuple[str, int]]]
) -> typing.Dict[int, typing.List[str]]:
    intersection_schedules = {}
    for node, schedules in schedules_dict.items():
        intersection_schedule = []
        for street in schedules:
            street_name, green_time = street
            if green_time > 0:
                intersection_schedule.append([street_name] * green_time)

        node_list = list(chain(*intersection_schedule))
        if len(node_list):
            intersection_schedules[node] = node_list
    return intersection_schedules


def fast_simulation(
    T: int,
    cars_df: pd.DataFrame,
    street_to_length: typing.Dict[str, int],
    intersection_schedules: typing.Dict[int, typing.List[str]],
) -> typing.Tuple[pd.DataFrame, int]:
    score = 0

    streets = list(street_to_length.keys())
    queues = initialize_queues(cars_df, streets)
    metrics = []

    # numpy arrays
    distance_left = cars_df["distance_left"].values
    index = cars_df["index"].values
    num_streets = (cars_df["num_streets"]).values
    car_num = cars_df["car_num"].values
    done = cars_df["done"].values
    path_streets = cars_df["path_streets"].values
    location = cars_df["location"].values
    for t in range(T):
        # if t % 1000 == 0:
        #     print(t)
        people_not_at_lights_idx = distance_left > 1
        people_approaching_lights = distance_left == 1

        people_finishing = np.logical_and(
            people_approaching_lights, index == (num_streets - 1)
        )

        green_lights = get_green_lights(intersection_schedules, t)
        starting_cars = [
            queues[green_light].pop(0)
            for green_light in green_lights
            if queues.get(green_light)
        ]

        starting_cars_idx = np.isin(car_num, starting_cars) & ~done
        if starting_cars_idx.any():
            index[starting_cars_idx] += 1
            starting_street_paths = path_streets[starting_cars_idx]
            starting_index = index[starting_cars_idx]
            starting_streets = list(
                map(lambda x: x[0][x[1]], zip(starting_street_paths, starting_index))
            )

            location[starting_cars_idx] = starting_streets
            distance_left[starting_cars_idx] = list(
                map(lambda x: street_to_length[x] - 1, starting_streets)
            )

        people_starting_length_1 = starting_cars_idx & (distance_left == 0)
        driving_idx = (
                people_not_at_lights_idx | people_approaching_lights | people_finishing
        )  # excludes last street length 1

        distance_left[driving_idx] -= 1
        finish_length_1 = people_starting_length_1 & (index == (num_streets - 1))
        people_finishing |= finish_length_1

        heading_to_queue = ~people_finishing & (
                people_approaching_lights | people_starting_length_1
        )

        queues_to_add_locations = location[heading_to_queue]
        queues_to_add_carnums = car_num[heading_to_queue]

        add_to_queues_np(queues, queues_to_add_locations, queues_to_add_carnums)

        score += (T - t) * people_finishing.sum()
        done[people_finishing] = True

        metrics.append(
            dict(
                driving=driving_idx.sum(),
                num_starting=starting_cars_idx.sum(),
                max_index=np.max(index),
                min_index=np.min(index),
                mean_index=np.mean(index),
                num_done=np.sum(done),
            )
        )

    metrics_df = pd.DataFrame.from_dict(metrics)

    return metrics_df, score



graph, roads_df, cars_df, T, street_to_length = parse_file(r"hashcode.in")
street_to_node = (
    roads_df[["street_name", "node_end"]].set_index("street_name").to_dict()["node_end"]
)
node_to_streets = (
    roads_df[["street_name", "node_end"]]
    .groupby("node_end")["street_name"]
    .apply(list)
    .to_dict()
)

best_score = -1
best_schedules_dict = None

schedules_dict = simple_schedule(graph)
import time

start = time.time()
for i in range(10000):
    schedules = schedules_dict_to_schedules_list(schedules_dict)
    intersection_schedules = schedules_dict_to_intersection_schedules(schedules_dict)

    simple_metrics_df, score = fast_simulation(
        T, cars_df.copy(), street_to_length, intersection_schedules
    )

    if score > best_score:
        best_score = score
        best_schedules_dict = schedules_dict
        print("new best score", score, time.time() - start)
        pickle.dump(schedules_dict, open(f"{score}", "wb"))
    else:
        print(f"{score} < {best_score}", time.time() - start)

    schedules_dict = mutate_schedules(schedules_dict)
