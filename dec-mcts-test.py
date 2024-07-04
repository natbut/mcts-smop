# Access python implementation of Dec-MCTS from University of Technology Sydney here: https://code.research.uts.edu.au/bigprint/pydecmcts
# Install Dec-MCTS with: pip install git+https://code.research.uts.edu.au/bigprint/pydecmcts.git

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from my_DecMCTS import Tree


class State:
    """
    Object stored at each node (defines state of solution, not environment)
    """

    def __init__(self, act_seq, cum_sum):
        self.action_seq = act_seq
        self.cumulative_sum = cum_sum

    def __str__(self):
        return str(self.action_seq)


def state_storer(data, parent_state: State, action, id):
    """
    This calculates the object stored at a given node given parent node and action
    """

    # Root Node edge case
    if parent_state == None:
        # This state is also used Null action when calculating local reward
        return State([], 0)

    state = deepcopy(parent_state)  # Make sure to deepcopy if editing parent
    state.action_seq.append(action)
    state.cumulative_sum = state.cumulative_sum + action
    return state


# data can be anything required to calculate your global reward and available actions
# It can be in any format (only used by your reward and avail_actions functions) ... AND state storer??
data = {}


def avail_actions(data, state: State, robot_id):
    """
    Create an available actions function

    This returns a list of possible actions to take from a given state (from state_storer)
    """

    # This example is simply getting max sum,
    # options are same regardless of state
    return [1, 2, 3, 4, 5]


def reward(data, states: dict):
    """
    Create a reward function. This is the global reward given a list of  actions taken by the current robot, and every other robot states is a dictionary with keys being robot IDs, and values are the object returned by the state_storer function you provide
    """
    # TODO reward is cumulative sum for this specific max sum problem
    each_robot_sum = [states[robot].cumulative_sum for robot in states]
    return sum(each_robot_sum)


def sim_get_actions_available(data, state: State, rob_id: int):
    """
    Return available actions during simulation
    """
    # print("Finding sim avail actions given state:", state)
    return [1, 2, 3, 4, 5]


def sim_select_action(data, options: list, state: State):
    """
    Choose an available option during simulation (can be random)
    """
    return np.random.choice(options)


# Number of Action Sequences to communicate
comm_n = 5

# Create instances for each robot
tree1 = Tree(data, reward, avail_actions, state_storer, sim_select_action,
             sim_get_actions_available, comm_n=comm_n, robot_id=1)  # Robot ID is 1
tree2 = Tree(data, reward, avail_actions, state_storer, sim_select_action,
             sim_get_actions_available, comm_n=comm_n, robot_id=2)  # Robot ID is 2

rew1 = []
rew2 = []
for i in range(100):
    # print("Step", i)
    rew1.append(tree1.grow())
    rew2.append(tree2.grow())
    # send comms message doesn't have ID in it
    tree1.receive_comms(tree2.send_comms(), 2)
    tree2.receive_comms(tree2.send_comms(), 1)
    # time.sleep(0.1)

fig, ax = plt.subplots(2)
ax[0].plot(rew1)
ax[1].plot(rew2)
plt.show()
