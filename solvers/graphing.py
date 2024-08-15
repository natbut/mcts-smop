from copy import deepcopy

import numpy as np
import yaml
from scipy.stats import norm

from control.env_model import EnvironmentModel
from control.task import Task


class Graph:
    def __init__(self, vertices, rewards, works, edges, cost_distributions, cost_func=None):
        self.vertices = vertices
        self.edges = edges
        self.rewards = rewards
        self.works = works
        self.cost_distributions = cost_distributions
        self.cost_func = cost_func

    def sample_edge_stoch(self, edge):
        random_sample = self.cost_distributions[edge].rvs(size=5)
        return int(max(1, np.random.choice(random_sample)))

    def get_edge_mean(self, edge):
        # print("Distr Mean:", self.cost_distributions[edge].mean())
        try:
            return int(self.cost_distributions[edge].mean())
        except:
            # TODO Error for (vs, vg)
            # print("Error! NaN for edge", edge)
            return 0

    def get_stoch_cost_edgeWork(self, edge):
        # Use for
        random_sample = self.sample_edge_stoch(edge)
        total_dist = self.works[edge[1]] + random_sample
        if self.cost_func:
            return self.cost_func(total_dist)
        return total_dist

    def get_mean_cost_edgeWork(self, edge):
        total_dist = self.works[edge[1]] + self.get_edge_mean(edge)
        # print(edge, "Work:", self.works[edge[1]],
        #       " + Edge:", self.get_edge_mean(edge))
        if self.cost_func:
            # print("Predicted cost", edge, ":", self.cost_func(total_dist))
            return self.cost_func(total_dist)
        return total_dist


def generate_graph_from_model(env_model: EnvironmentModel,
                              task_dict: dict[Task],
                              dim_ranges,
                              agent_vel,
                              cost_func,
                              c,
                              disp=False
                              ):
    """
    Generate graph for planning from given task dict
    """
    # Generate a planning graph using edge cost distributions provided by environment model as task edges

    # Set up distances between tasks (use as means for planning graph edge distros)

    vertices = [t for t in task_dict.keys()]
    rewards = {}
    works = {}

    for t in vertices:
        rewards[t] = task_dict[t].reward
        works[t] = task_dict[t].work

    edges = [(v1, v2) for v1 in vertices for v2 in vertices if v1 != v2]

    means = {}
    vars = {}
    for t1 in vertices:
        for t2 in vertices:
            if t1 != t2:
                loc1 = task_dict[t1].location
                loc2 = task_dict[t2].location
                mean, var = env_model.get_travel_dist_distribution(loc1,
                                                                   loc2,
                                                                   dim_ranges,
                                                                   agent_vel,
                                                                   disp=disp
                                                                   )
                # print("Edge", (t1, t2), " Mean:", mean, " Var:", var)
                means[(t1, t2)] = mean
                vars[(t1, t2)] = var

                # print("\nTask", t1, " and Task", t2,
                #       " \nLoc1:", loc1, " Loc2:", loc2,
                #       " \nMean:", mean, " Var:", var)

    cost_distributions = set_up_cost_distributions(vertices,
                                                   means,
                                                   vars,
                                                   c)
    return Graph(vertices,
                 rewards,
                 works,
                 edges,
                 cost_distributions,
                 cost_func
                 )


def set_up_cost_distributions(vertices, means, vars, c=0.05):
    cost_distributions = {}
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                edge = (v1, v2)
                mean = means[edge]
                # TODO will want to change this variation handling - maybe move to env_model and preset edges differently?
                # Vars from env_model are primarily 0
                stddev = max((c * mean)**0.5, vars[edge]**0.5)
                # print("Setting cost dist", (v1, v2), ":", mean, stddev)
                cost_distributions[edge] = norm(loc=mean, scale=stddev)
                # print("Mean:", mean, " Norm Mean: ",
                #       cost_distributions[edge].mean())

    return cost_distributions


def generate_cost_distributions(vertices, mean_range=(1, 5), c=0.05, seed=42):
    """
    Create edge cost distributions between vertices in complete graph
    """
    np.random.seed(seed)
    cost_distributions = {}
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                mean = np.random.uniform(mean_range[0], mean_range[1])
                stddev = (c * mean)**0.5
                cost_distributions[(v1, v2)] = norm(loc=mean, scale=stddev)
    return cost_distributions


def create_sop_instance_fromTasks(task_list: list[Task], c):

    vertices = [t.id for t in task_list]
    rewards = {}
    works = {}

    for i, task in enumerate(task_list):
        rewards[vertices[i]] = task.reward
        works[vertices[i]] = task.work

    edges = [(v1, v2) for v1 in vertices for v2 in vertices if v1 != v2]

    means = {}
    for i, task1 in enumerate(task_list):
        for j, task2 in enumerate(task_list):
            if i != j:
                means[(vertices[i], vertices[j])
                      ] = task1.distances_to_tasks[task2.id]

    cost_distributions = set_up_cost_distributions(vertices,
                                                   means,
                                                   c
                                                   )

    return Graph(vertices,
                 rewards,
                 works,
                 edges,
                 cost_distributions
                 )


def create_sop_instance(num_vertices: int,
                        mean_range=None,
                        c=0.05,
                        reward_range=(0, 10),
                        work_range=(0, 0),
                        rand_seed=42
                        ) -> Graph:
    """
    Create graph with stochastic edge costs and given number of vertices
    """

    # Generate vertices
    vertices = ["vs"]
    for i in range(num_vertices):
        vertices.append("v" + str(i))
    vertices.append("vg")

    # Generate edges for complete graph
    edges = [(v1, v2) for v1 in vertices for v2 in vertices if v1 != v2]

    # Generate distributions
    cost_distributions = generate_cost_distributions(vertices,
                                                     mean_range,
                                                     c,
                                                     seed=rand_seed)

    # Generate rewards NOTE: vs and vg get rewards here
    rewards = {}
    for v in vertices:
        if reward_range[0] == reward_range[1]:
            rewards[v] = reward_range[0]
        else:
            rewards[v] = np.random.randint(reward_range[0], reward_range[1]+1)

    # Generate work costs for each node
    works = {}
    for v in vertices:
        if work_range[0] == work_range[1]:
            works[v] = work_range[0]
        else:
            works[v] = np.random.randint(work_range[0], work_range[1]+1)

    return Graph(vertices, rewards, works, edges, cost_distributions)


def create_dummy_graph(graph: Graph, c) -> Graph:
    dummy_graph = deepcopy(graph)
    # NOTE perhaps increase the difference between means?
    for edge in dummy_graph.edges:
        new_edge_mean = dummy_graph.sample_edge_stoch(edge)
        # Min edge mean should be >= 0
        new_mean_range = (new_edge_mean-1, new_edge_mean+1)
        new_cost_dist = generate_cost_distributions([edge[0], edge[1]],
                                                    new_mean_range,
                                                    c)
        dummy_graph.cost_distributions[edge] = new_cost_dist[edge]
    return dummy_graph


def create_true_graph(stoch_graph: Graph, c=0.05) -> Graph:
    # Use very low c for deterministic
    true_graph = deepcopy(stoch_graph)

    for edge in true_graph.edges:
        cost_sample = true_graph.sample_edge_stoch(edge)
        stddev = (c * cost_sample)**0.5
        true_graph.cost_distributions[edge] = norm(
            loc=cost_sample, scale=stddev)

    return true_graph


def create_sop_inst_from_config(config_filepath) -> Graph:

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

        return create_sop_instance(config["problem_size"],
                                   config["edges_mean_range"],
                                   config["c"],
                                   config["reward_range"],
                                   rand_seed=np.random.randint(100))
