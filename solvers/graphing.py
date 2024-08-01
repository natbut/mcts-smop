from copy import deepcopy

import numpy as np
import yaml
from scipy.stats import norm


class Graph:
    def __init__(self, vertices, rewards, works, edges, cost_distributions):
        self.vertices = vertices
        self.edges = edges
        self.rewards = rewards
        self.works = works
        self.cost_distributions = cost_distributions

    def sample_edge_stoch(self, edge):
        random_sample = self.cost_distributions[edge].rvs(size=5)
        return int(max(1, np.random.choice(random_sample)))

    def get_edge_mean(self, edge):
        return int(self.cost_distributions[edge].mean())

    def get_stoch_cost_edgeWork(self, edge):
        # Use for
        random_sample = self.cost_distributions[edge].rvs(size=5)
        return self.works[edge[1]] + int(max(1, np.random.choice(random_sample)))

    def get_mean_cost_edgeWork(self, edge):
        return self.works[edge[1]] + int(self.cost_distributions[edge].mean())


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
