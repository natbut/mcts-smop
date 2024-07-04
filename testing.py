from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from mcts_sop import Graph, mcts_with_sopcc
from my_DecMCTS import Tree


# === PROBLEM SETUP FUNCTIONS ===
def generate_cost_distributions(vertices, seed=42):
    # Create cost distributions for complete graph
    np.random.seed(seed)
    cost_distributions = {}
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                mean = np.random.uniform(1, 5)  # Mean between 1 and 5
                # Stddev between 0.5 and 1.5
                stddev = np.random.uniform(0.5, 1.5)
                cost_distributions[(v1, v2)] = norm(loc=mean, scale=stddev)
    return cost_distributions


# === DEC-MCTS SETUP FUNCTIONS ===
class State:
    """
    Object stored at each node (defines state of solution, not environment)
    """

    def __init__(self, act_seq, budget):
        self.action_seq = act_seq  # ordered list of tasks to visit
        # self.tasks = tasks
        self.remaining_budget = budget

    def __str__(self):
        return "x:" + str(self.action_seq) + " | rem_budget:" + str(self.remaining_budget)


def state_storer(data: dict, parent_state: State, action, id):
    """
    This calculates the object stored at a given node given parent node and action
    """
    # Root Node edge case
    if parent_state == None:
        # This state is also used Null action when calculating local reward
        return State([data["start"], data["end"]], data["budget"])

    state = deepcopy(parent_state)  # Make sure to deepcopy if editing parent
    state.action_seq.remove(data["end"])  # Enforce ending at end state
    state.action_seq += [action, data["end"]]

    # Update remaining budget
    actions = state.action_seq
    cost = 0
    for i in range(len(actions)-1):
        edge = (actions[i], actions[i+1])
        cost += data["graph"].get_mean_cost(edge)
    state.remaining_budget = data["budget"] - cost
    return state


def avail_actions(data, state: State, robot_id):
    """
    Create an available actions function (FOR EXPANSION)

    This returns a list of possible actions to take from a given state (from state_storer)
    """
    # Restrict available actions to those that do not exceed travel budget (is how Dec-MCTS paper formulated it). However, the paper does not enforce returning to a goal location. So we want to restrict to actions where cost to travel to action + cost from action to goal does not exceed remaining budget.
    # NOTE: This may be handled differently once stochastic edges are introduced
    choices = []
    graph = data["graph"]
    for task in graph.vertices:
        # check if task is already scheduled
        if task not in state.action_seq:
            # check if task is reachable within budget
            # back 2 because last action is vg
            last_action = state.action_seq[-2]
            cost = graph.get_mean_cost(
                (last_action, task)) + graph.get_mean_cost((task, data["end"]))
            if cost <= state.remaining_budget:
                choices.append(task)
    # print("Given actions", state.action_seq, " we have choices: ", choices)
    return choices


def reward(data: dict, states: dict[State]):
    """
    Create a reward function. This is the global reward given a list of  actions taken by the current robot, and every other robot states is a dictionary with keys being robot IDs, and values are the object returned by the state_storer function you provide
    """
    # Return sum of reward for each unique task visited (duplicates don't reward)
    # TODO Check against local utility function defined in paper
    # print("REWARD CALC")
    all_tasks_visited = []
    for robot in states:
        all_tasks_visited += states[robot].action_seq
    # print("all tasks visited:", all_tasks_visited)
    unique_tasks_visited = set(all_tasks_visited)
    # print("unique tasks visited:", unique_tasks_visited)
    graph = data["graph"]
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited)


def sim_get_actions_available(data: dict, states: dict[State], rob_id: int):
    """
    Return available actions during simulation (FOR ROLLOUT)
    """
    # Return list of unallocated tasks. As the graph is complete, all tasks can be returned. In the future, may only return a subset of tasks that neighbor the final task in rob_id's action sequence

    # Restrict available actions to those that do not exceed travel budget (is how Dec-MCTS paper formulated it). However, the paper does not enforce returning to a goal location. So we want to restrict to actions where cost to travel to action + cost from action to goal does not exceed remaining budget.
    # NOTE: This may be handled differently once stochastic edges are introduced
    reachable_unallocated_tasks = []
    graph = data["graph"]
    robot_state = states[rob_id]

    all_tasks_allocated = []
    for robot in states:
        all_tasks_allocated += states[robot].action_seq
    # print("all tasks visited:", all_tasks_visited)
    unique_tasks_allocated = set(all_tasks_allocated)

    # print("Tasks allocated:", unique_tasks_allocated)
    for task in graph.vertices:
        if task not in unique_tasks_allocated:
            last_action = robot_state.action_seq[-2]
            cost = graph.get_mean_cost(
                (last_action, task)) + graph.get_mean_cost((task, data["end"]))
            if cost <= robot_state.remaining_budget:
                reachable_unallocated_tasks.append(task)

    return reachable_unallocated_tasks


def sim_select_action(data: dict, options: list, state: State):
    """
    Choose an available option during simulation (can be random)
    FOR ROLLOUT
    """
    # TODO More intelligent selection choice (See SOPCC)
    # TODO from DecMCTS: Select next edge that does not exceed travel budget and maximizes ratio of increased reward to edge cost
    return np.random.choice(options)


if __name__ == "__main__":

    # === ORIENTEERING PROBLEM DEFINITION ===

    # Define small graph with stochastic edge costs
    vertices_small = ["vs", "v1", "v2", "v3", "vg"]
    edges_small = [(v1, v2)
                   for v1 in vertices_small for v2 in vertices_small if v1 != v2]
    rewards_small = {"vs": 0, "v1": 10, "v2": 20, "v3": 15, "vg": 10}
    cost_distributions_small = generate_cost_distributions(
        vertices_small, seed=42)

    # Define medium graph with stochastic edge costs
    vertices_medium = ["vs", "v1", "v2", "v3", "v4",
                       "v5", "v6", "v7", "v8", "v9", "v10", "vg"]
    edges_medium = [(v1, v2)
                    for v1 in vertices_medium for v2 in vertices_medium if v1 != v2]
    rewards_medium = {"vs": 0, "v1": 10, "v2": 20, "v3": 15, "v4": 30,
                      "v5": 15, "v6": 25, "v7": 30, "v8": 20, "v9": 45, "v10": 10, "vg": 10}
    cost_distributions_medium = generate_cost_distributions(
        vertices_medium, seed=42)

    # Define graphs for different problem sizes
    graph_small = Graph(vertices_small, edges_small,
                        rewards_small, cost_distributions_small)

    graph_medium = Graph(vertices_medium, edges_medium,
                         rewards_medium, cost_distributions_medium)

    # === DEC-MCTS PROBLEM SETUP ===
    graph = graph_medium
    start = "vs"
    end = "vg"
    budget = 20
    iterations = 10
    comm_n = 5  # Number of Action Sequences to communicate
    failure_probability = 0.1  # for stoch

    # data required to calculate global reward and available actions
    data1 = {"graph": deepcopy(graph), "start": start, "end": end,
             "budget": budget, "fail_prob": failure_probability}
    data2 = {"graph": deepcopy(graph), "start": start, "end": end,
             "budget": budget, "fail_prob": failure_probability}

    rew1 = [0]
    rew2 = [0]
    # === ALGORITHM 1 ===
    i = 0
    # while still traveling and energy remains
    while data1["start"] != data1["end"] and data2["start"] != data2["end"] and data1["budget"] > 0 and data2["budget"] > 0:
        print(" === STEP ", i, "===")
        i += 1

        # Initialize MCTS tree instances for each robot
        tree1 = Tree(data1, reward, avail_actions, state_storer, sim_select_action,
                     sim_get_actions_available, comm_n=comm_n, robot_id=1)  # Robot ID is 1
        tree2 = Tree(data2, reward, avail_actions, state_storer, sim_select_action,
                     sim_get_actions_available, comm_n=comm_n, robot_id=2)  # Robot ID is 2

        # Alg1. SELECT SET OF SEQUENCES?
        # 10 paths resampled every 10 iterations (per robot)

        for _ in range(iterations):
            # Alg1. GROW TREE & UPDATE DISTRIBUTION
            tree1.grow()
            tree2.grow()
            # Alg1. COMMS TRANSMIT & COMMS RECEIVE
            tree1.receive_comms(tree2.send_comms(), 2)
            tree2.receive_comms(tree2.send_comms(), 1)
            # Alg1. Cool?

        # Alg1. Return highest-prob action sequence (best)
        best_act1 = tree1.my_act_dist.best_action().action_seq
        print("best1:", best_act1)
        best_act2 = tree2.my_act_dist.best_action().action_seq
        print("best2:", best_act2)

        # Execute first action in sequence
        # update budgets
        print("move r1 to", best_act1[1])
        print("move r2 to", best_act2[1])
        data1["budget"] -= data1["graph"].get_stoch_cost(
            (best_act1[0], best_act1[1]))
        data2["budget"] -= data2["graph"].get_stoch_cost(
            (best_act2[0], best_act2[1]))
        print("budget1 reduced to:", data1["budget"])
        print("budget2 reduced to:", data2["budget"])
        # update rewards
        rew1.append(rew1[-1] + data1["graph"].rewards[best_act1[0]])
        rew2.append(rew2[-1] + data2["graph"].rewards[best_act2[0]])
        # update locations
        data1["start"] = best_act1[1]
        data2["start"] = best_act2[1]
        # update graphs
        data1["graph"].vertices.remove(best_act1[0])
        data2["graph"].vertices.remove(best_act2[0])
        print("Remaining g1 verts:", data1["graph"].vertices)
        print("Remaining g2 verts:", data2["graph"].vertices)
        # TODO store other robots' action distributions for new tree (so comms doesn't have to start from scratch)

    # TODO determine which results are important to display
    print("Tree 1 best:")
    for x in tree1.my_act_dist.X:
        actions = x.action_seq
        cost = 0
        for i in range(len(actions)-1):
            edge = (actions[i], actions[i+1])
            cost += graph.get_mean_cost(edge)
        print(actions, " | Cost: ", cost)

    print("Tree 2 best:")
    for x in tree1.my_act_dist.X:
        actions = x.action_seq
        cost = 0
        for i in range(len(actions)-1):
            edge = (actions[i], actions[i+1])
            cost += graph.get_mean_cost(edge)
        print(actions, " | Cost: ", cost)

    fig, ax = plt.subplots(2)
    ax[0].plot(rew1[1:])
    ax[1].plot(rew2[1:])
    fig.suptitle("Cum. Reward Obtained Over Time")
    plt.show()
