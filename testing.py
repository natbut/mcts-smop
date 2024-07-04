

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from DecMCTS import Tree
from mcts_sop import Graph, mcts_with_sopcc


# === PROBLEM SETUP ===
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


def sample_cost(distribution):
    # Function to sample costs from distributions
    return distribution.rvs()


# === SETUP FOR DEC-MCTS ===


class State:
    """
    Object stored at each node (defines state of solution, not environment)
    """

    def __init__(self, act_seq, budget):
        self.action_seq = act_seq  # ordered list of tasks to visit
        # self.tasks = tasks
        self.remaining_budget = budget

    def __str__(self):
        return str(self.action_seq)


def state_storer(data: dict, parent_state: State, action, id):
    """
    This calculates the object stored at a given node given parent node and action
    """
    # Root Node edge case
    if parent_state == None:
        # This state is also used Null action when calculating local reward
        return State([data["start"]], data["budget"])

    state = deepcopy(parent_state)  # Make sure to deepcopy if editing parent
    state.action_seq.append(action)
    # state.cumulative_sum = state.cumulative_sum + action
    return state


def avail_actions(data, state: State, robot_id):
    """
    Create an available actions function (FOR EXPANSION)

    This returns a list of possible actions to take from a given state (from state_storer)
    """
    # TODO SET UP THIS FUNCTION

    # This example is simply getting max sum,
    # options are same regardless of state
    return [1, 2, 3, 4, 5]


def reward(data: dict, states: dict[State]):
    """
    Create a reward function. This is the global reward given a list of  actions taken by the current robot, and every other robot states is a dictionary with keys being robot IDs, and values are the object returned by the state_storer function you provide
    """
    # Return sum of reward for each unique task visited (duplicates don't reward)
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
    # print("Finding sim avail actions given state:", state)

    # Return list of unallocated tasks. As the graph is complete, all tasks can be returned. In the future, will only return a subset of tasks that neighbor the final task in rob_id's action sequence

    # NOTE: Currently no consideration for other tasks allocated to other robots, which allows for duplicates to be scheduled

    # Restrict available actions to those that do not exceed travel budget (is how Dec-MCTS paper formulated it). However, the paper does not enforce returning to a goal location. So we want to restrict to actions where cost to travel to action + cost from action to goal does not exceed remaining budget.
    # NOTE: This may be handled differently once stochastic edges are introduced
    unallocated_tasks = []
    graph = data["graph"]
    robot_state = states[rob_id]

    # TODO WHY ISN'T THIS WORKING???
    allocated_tasks = []
    for robot in states:
        print(states[robot].action_seq)
        allocated_tasks.append(states[robot].action_seq)

    print("Allocated tasks:", allocated_tasks)

    for task in graph.vertices:
        if task not in allocated_tasks:
            last_action = robot_state.action_seq[-1]
            cost = graph.get_mean_cost((last_action, task))
            if task != data["end"]:  # handle moving to end
                cost += graph.get_mean_cost((task, data["end"]))
            if cost <= data["budget"]:
                unallocated_tasks.append(task)

    return unallocated_tasks


def sim_select_action(data: dict, options: list, state: State):
    """
    Choose an available option during simulation (can be random)
    FOR ROLLOUT
    """
    # TODO More intelligent selection choice (See SOPCC)
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

    # Parameters for MCTS-SOPCC
    start = "vs"
    end = "vg"
    budget = 10
    iterations = 100
    failure_probability = 0.1

    # result = mcts_with_sopcc(graph_small, start, goal, budget,
    #  iterations, failure_probability)

    # Parameters for Dec-MCTS

    # data can be anything required to calculate your global reward and available actions
    # It can be in any format (only used by your reward and avail_actions functions) ... AND state storer??
    data = {"graph": graph_small, "start": start, "end": end,
            "budget": budget, "fail_prob": failure_probability}

    # Number of Action Sequences to communicate
    comm_n = 5

    # Create tree instances for each robot
    tree1 = Tree(data, reward, avail_actions, state_storer, sim_select_action,
                 sim_get_actions_available, comm_n=comm_n, robot_id=1)  # Robot ID is 1
    tree2 = Tree(data, reward, avail_actions, state_storer, sim_select_action,
                 sim_get_actions_available, comm_n=comm_n, robot_id=2)  # Robot ID is 2
    rew1 = []
    rew2 = []
    for i in range(10):
        if i % 25 == 0:
            print("Step", i)
        # print("Step", i)
        rew1.append(tree1.grow())
        rew2.append(tree2.grow())
        # send comms message doesn't have ID in it
        tree1.receive_comms(tree2.send_comms(), 2)
        tree2.receive_comms(tree2.send_comms(), 1)
        # time.sleep(0.1)

    # TODO display final solution
    print("Tree 1 best:", tree1.my_act_dist)
    print("Tree 2 best:", tree2.my_act_dist)

    fig, ax = plt.subplots(2)
    ax[0].plot(rew1)
    ax[1].plot(rew2)
    plt.show()
