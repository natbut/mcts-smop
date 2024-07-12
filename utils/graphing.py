from math import ceil

import numpy as np
from scipy.stats import norm


class Graph:
    def __init__(self, vertices, edges, rewards, cost_distributions):
        self.vertices = vertices
        self.edges = edges
        self.rewards = rewards
        self.cost_distributions = cost_distributions

    def get_stoch_cost(self, edge):
        random_sample = self.cost_distributions[edge].rvs(size=5)
        return ceil(max(0, np.random.choice(random_sample)))

    def get_mean_cost(self, edge):
        return ceil(self.cost_distributions[edge].mean())


def generate_cost_distributions(vertices, mean_range=(1, 5), c=0.05, seed=42):
    """
    Create edge cost distributions between vertices in complete graph
    """
    np.random.seed(seed)
    cost_distributions = {}
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                # Mean
                mean = np.random.uniform(mean_range[0], mean_range[1])
                # Stddev
                stddev = (c * mean)**0.5  # c=0.75 from Panadero papers

                # custom_range = (mean/2, mean)
                # (stddev_range[0], stddev_range[1])
                # stddev = np.random.uniform(custom_range[0], custom_range[1])
                cost_distributions[(v1, v2)] = norm(loc=mean, scale=stddev)
    return cost_distributions


def create_sop_instance(num_vertices: int,
                        mean_range=None,
                        c=0.05,
                        reward_range=(0, 10),
                        rand_seed=42
                        ) -> Graph:
    # Create graph with stochastic edge costs and given number of vertices

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
    rewards["vs"] = 0
    rewards["vg"] = 0

    return Graph(vertices, edges, rewards, cost_distributions)
