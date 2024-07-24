# Implementation of algorithms from:
# Solving Stochastic Orienteering Problems with Chance Constraints
# Using Monte Carlo Tree Search
# by Carpin and Thayer (2022)

import random

import numpy as np

from solvers.graphing import Graph


class Node:
    """
    Node class for MCTS
    """

    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.fail_prob = 0


def uctf_select(node: Node, z=2.0):
    best_value = float("-inf")
    best_node = None
    for child in node.children:
        uctf_value = child.reward * (1 - child.fail_prob) + z * np.sqrt(
            np.log(node.visits + 1) / (child.visits + 1)
        )
        # print('Chile', child.state, "UCTF:", uctf_value)
        if uctf_value > best_value:
            best_value = uctf_value
            best_node = child
    return best_node


def expand(node: Node, graph: Graph):
    # TODO may be able to speed things up here
    untried_actions = [
        v
        for v in graph.vertices
        if v not in [child.state for child in node.children]
        and (node.state, v) in graph.edges
    ]
    # print("untried actions", untried_actions)
    if untried_actions:
        action = random.choice(untried_actions)
        child_node = Node(action, parent=node)
        node.children.append(child_node)
        return child_node
    return node


def sampleTraverseTime(path: list[Node], graph: Graph) -> int:
    if len(path) == 1:
        return 0
    sampled_cost = sum(
        graph.get_cost((path[i], path[i+1])) for i in range(len(path)-1))

    return sampled_cost


def rollout(node: Node, goal: str, graph: Graph, budget: int, failure_probability: float, sample_iters=10, greedy_iters=10):
    paths = []

    for _ in range(sample_iters):
        # Setup
        current_state = node.state
        # Fill in path root to node
        path = []
        temp = node
        while temp != None:
            path.insert(0, temp.state)
            temp = temp.parent
        remaining_budget = budget - sampleTraverseTime(path, graph)

        # ROLLOUT STEPS
        # First pick random vertex that is both unvisited and not along path between root and node
        possible_actions = [
            v for v in graph.vertices if (current_state, v) in graph.edges and v not in path
        ]
        # print("Possible actions:", possible_actions)
        if not possible_actions:
            break

        v_rand = random.choice(possible_actions)
        path.append(v_rand)
        current_state = v_rand

        # Next, repeatedly pick vertex using greedy criterion until goal is reached or failure
        # While goal not reached
        while current_state != goal:
            # Assemble list of candidate edges (reachable from last vertex in path and not yet visited)
            v_l = path[-1]
            candidates = [v for v in graph.vertices if (
                v_l, v) in graph.edges and v not in path]

            # Sample edge costs, both for reward/cost ratio and for discarding for chance constraints
            best_vk = None
            best_ratio = 0
            for v_k in candidates:

                total_cost_vl_vk = 0
                total_cost_vk_vg = 0
                budget_exceeded = 0
                for _ in range(greedy_iters):
                    total_cost_vl_vk += graph.get_cost((v_l, v_k))
                    if v_k == goal:
                        total_cost_vk_vg += 0
                    else:
                        total_cost_vk_vg += graph.get_cost((v_k, goal))

                    if total_cost_vl_vk + total_cost_vk_vg > remaining_budget:
                        budget_exceeded += 1

                avg_cost_vl_vk = total_cost_vl_vk / greedy_iters
                percent_failed = budget_exceeded / greedy_iters
                # Enforce constraint on failure probability (probability of failed samples must be less than acceptable failure probability)
                if percent_failed <= failure_probability:
                    # Evaluate each ratio whose edge satisfies chance constraint. Select highest ratio and add vertex to path
                    ratio = graph.rewards[v_k] / avg_cost_vl_vk
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_vk = v_k

            if best_vk == None:  # If no options that satisfy chance constraints are found, go straight to goal
                best_vk = goal
            current_state = best_vk
            path.append(best_vk)

        # When path is complete, add it to paths list and start another sample
        paths.append(path)

    # Process stored paths for Q and F values
    total_reward = 0
    total_failures = 0
    for path in paths:
        total_reward = sum(graph.rewards[v] for v in path)
        sampled_cost = sum(graph.get_cost(
            (path[i], path[i+1])) for i in range(len(path)-1))
        if sampled_cost > budget:
            total_failures += 1

    average_reward = total_reward / sample_iters
    average_failures = total_failures / sample_iters

    return average_reward, average_failures


def backup(node: Node, reward, failure, allowed_fail_prob):
    node.reward = reward
    node.fail_prob = failure
    node.visits += 1

    while node is not None:
        parent = node.parent
        if parent == None:
            break

        # Better solution found (higher reward, lower failure)
        if parent.fail_prob >= node.fail_prob and parent.reward <= node.reward + graph.rewards[parent.state]:
            parent.fail_prob = node.fail_prob
            parent.reward = node.reward + graph.rewards[parent.state]  # TODO
        # Okay solution found (higher reward, failure still within constraint)
        elif parent.fail_prob < node.fail_prob and parent.reward <= node.reward + graph.rewards[parent.state] and node.fail_prob < allowed_fail_prob:
            parent.fail_prob = node.fail_prob
            parent.reward = node.reward + graph.rewards[parent.state]  # TODO
        # No backprop in all other cases
        else:
            parent.fail_prob = node.fail_prob
            parent.reward = node.reward
            break

        node = node.parent
        node.visits += 1


def sopcc(graph, start: str, goal: str, budget, iterations, sample_iterations=10, failure_probability=0.1):
    # SOPCC (Alg. 2)
    print("=== Running SOPCC ===")
    root = Node(start)  # initialize tree
    for _ in range(iterations):
        # Traverse tree (Select)
        node = root
        while node.children:  # recursively apply UCTF to reach leaf
            # print("Node", node.state, "children:",
            #   [c.state for c in node.children])
            node = uctf_select(node)
        # Expand already-visited leaves
        if node.visits > 0:
            # print("Node", node.state, "visited; expand")
            node = expand(node, graph)
        # Rollout
        reward, failure = rollout(
            node, goal, graph, budget, failure_probability)
        # Backup
        backup(node, reward, failure, failure_probability)

    # Return root child with highest score that satisfies constraints
    best_child = None
    best_reward = 0
    for child in root.children:
        if child.reward > best_reward and child.fail_prob <= failure_probability:
            best_child = child
            best_reward = child.reward
    if best_child != None:
        return best_child.state
    else:
        return goal


def mcts_with_sopcc(graph: Graph, start: str, goal: str, budget, iterations,  failure_probability):
    current_vertex = start
    # Alternating planning and execution (Alg. 1)
    while budget > 0 and current_vertex != goal:
        # Get next action using tree search from current vertex
        next_vertex = sopcc(
            graph, current_vertex, goal, budget, iterations, failure_probability
        )
        # Take next action, update budget
        travel_cost = graph.get_cost((current_vertex, next_vertex))
        print("Travel cost for ", (current_vertex, next_vertex), ":", travel_cost)
        budget -= travel_cost
        current_vertex = next_vertex
        print(f"Moved to {current_vertex} with remaining budget {budget}")

    if budget > 0:
        return "Success"
    else:
        return "Failure"


if __name__ == "__main__":
    # Parameters
    start = "vs"
    goal = "vg"
    budget = 10
    iterations = 100
    failure_probability = 0.1

    # Define the graph
    vertices = ["vs", "v1", "v2", "v3", "vg"]
    # TODO every vertex should have an edge directed from it to goal, but not necessarily to every other vertex
    edges = [  # TODO make undirected (or add reverse directions)
        ("vs", "v1"),
        ("vs", "v2"),
        ("v1", "v2"),
        ("v1", "vg"),
        ("v2", "v3"),
        ("v2", "vg"),
        ("v3", "vg"),
    ]
    rewards = {"vs": 0, "v1": 10, "v2": 20, "v3": 15, "vg": 30}
    cost_distributions = {
        ("vs", "v1"): [2, 3, 4],
        ("vs", "v2"): [1, 2, 3],
        ("v1", "v2"): [2, 2, 3],
        ("v1", "vg"): [4, 5, 6],
        ("v2", "v3"): [3, 4, 5],
        ("v2", "vg"): [3, 4, 5],
        ("v3", "vg"): [1, 2, 3],
    }

    graph = Graph(vertices, edges, rewards, cost_distributions)

    result = mcts_with_sopcc(graph, start, goal, budget,
                             iterations, failure_probability)
    print(result)
