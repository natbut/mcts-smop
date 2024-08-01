from __future__ import print_function

from copy import copy
from math import log

import networkx as nx
import numpy as np

from solvers.masop_solver_config import *


def _UCT(mu_j, c_p, n_p, n_j):
    if n_j == 0:
        return float("Inf")

    return mu_j + 2*c_p * (2*log(n_p)/n_j)**0.5


class ActionDistribution:
    """
    Action Distribution
    Working with action sequences and their respective probability

    To initialise, Inputs:
    - X: list of action sequences
        - NOTE: X is is simply a state object returned by state_store.
            You are expected to store action sequence in this object
    - q: probability of each action sequence (normalised in intialisation)

    """

    def __init__(self, X, q):

        # Action sequence as provided
        self.X = X

        # Normalise
        if sum(q) == 0:
            self.q = [1/float(len(q))] * len(q)
        else:
            self.q = (np.array(q).astype(float)/sum(q)).tolist()

    def __str__(self):
        s = ""
        for i in range(len(self.X)):
            s += "x: " + str(self.X[i]) + " | q: " + str(self.q[i]) + "\n"
        return s

    def best_action(self):
        """
        Most likely action sequence (state)
        """
        return self.X[np.argmax(self.q)]

    def random_action(self):
        """
        Weighted random out of possible action sequences (states)
        """
        return self.X[np.random.choice(len(self.q), p=self.q)]


class Tree:
    """
    DecMCTS tree
    To Initiate, Inputs:
    - data
        - data required to calculate reward, available options
    - reward
        - This is a function which has inputs (data, state) and
            returns the GLOBAL reward to be maximised
        - MUST RETURN POSITIVE VALUE
    - available_actions
        - This is a function which has inputs (data, state) and
            returns the possible actions which can be taken
    - state_store
        - This is a function which has inputs 
            (data, parent_state, action) and returns an object to
            store in the node. 
        - Root Node has parent state None and action None.
    - sim_selection_func
        - This is a function which chooses an available of option
            during simulation (can be random or more advanced)
    - c_p
        - exploration multiplier (number between 0 and 1)

    Usage:
    - grow
        - grow MCTS tree by 1 node
    - send_comms
        - get state of this tree to communicate to others
    - receive_comms
        - Input the state of other trees for use in calculating
            reward/available actions in coordination with others
    """

    def __init__(self,
                 data,
                 comm_n,
                 robot_id,
                 c_p=1):

        self.data = data
        self.graph = nx.DiGraph()
        self.reward = local_util_reward
        self.available_actions = avail_actions
        self.sim_available_actions = sim_get_actions_available
        self.state_store = state_storer
        self.sim_selection_func = sim_select_action
        self.c_p = c_p
        self.id = robot_id
        self.comms = {}  # Plan with no robots initially
        self.comm_n = comm_n  # number of action dists to communicate

        # Graph add root node of tree
        self.graph.add_node(1,
                            avg_reward=0,
                            N=0,
                            best_reward=0,
                            state=self.state_store(
                                self.data, None, None, self.id)
                            )  # TODO store failure prob as well

        # Set Action sequence as nothing for now
        self.my_act_dist = ActionDistribution(
            [self.graph.nodes[1]["state"]], [1])

        self._expansion(1)

    def _parent(self, node_id):
        """
        wrapper for code readability
        """
        return list(self.graph.predecessors(node_id))[0]

    def _select(self, children):
        """
        Select Child node which maximises UCT
        """

        # N for parent
        n_p = self.graph.nodes[self._parent(children[0])]["N"]

        # UCT values for children # TODO: use UCTF
        uct = [_UCT(node["avg_reward"], self.c_p, n_p, node["N"])
               for node in map(self.graph.nodes.__getitem__, children)]

        # Return Child with highest UCT
        return children[np.argmax(uct)]

    def _childNodes(self, node_id):
        """
        wrapper for code readability
        """

        return list(self.graph.successors(node_id))

    def _update_distribution(self):
        """
        Get the top n Action sequences and their "probabilities"
            and store them for communication
        """

        # TODO For now, just using q = mu**2
        temp = nx.get_node_attributes(self.graph, "avg_reward")
        temp.pop(1, None)  # remove root to leave children of temp

        if len(temp) == 0:  # not enough children to create distribution
            return False

        top_n_nodes = sorted(temp, key=temp.get, reverse=True)[:self.comm_n]
        X = [self.graph.nodes[n]["best_rollout"]
             for n in top_n_nodes if self.graph.nodes[n]["N"] > 0]
        q = [self.graph.nodes[n]["avg_reward"] **
             2 for n in top_n_nodes if self.graph.nodes[n]["N"] > 0]
        self.my_act_dist = ActionDistribution(X, q)
        return True

    def _get_system_state(self, node_id):
        """
        Randomly select 1 path taken by every other robot & path taken by this robot to get to this node

        Returns dict with sampled state of other robots and this robot's state represented at graph node node_id
        """

        system_state = {k: self.comms[k].random_action() for k in self.comms}
        system_state[self.id] = self.graph.nodes[node_id]["state"]
        return system_state

    def _null_state(self, state):
        temp = copy(state)
        # Null state is if robot still at root node
        temp[self.id] = self.graph.nodes[1]["state"]
        return temp

    def _expansion(self, start_node):
        """
        Does the Expansion step for tree growing.
        Separated into its own function because also done in Init step.
        """

        options = self.available_actions(
            self.data,
            self.graph.nodes[start_node]["state"],
            self.id
        )

        if len(options) == 0:
            return False

        # create empty nodes underneath the node being expanded
        for o in options:
            self.graph.add_node(len(self.graph)+1,
                                avg_reward=0,
                                best_reward=0,
                                N=0,
                                state=self.state_store(
                                    self.data, self.graph.nodes[start_node]["state"], o, self.id)
                                )

            self.graph.add_edge(start_node, len(self.graph))
        return True

    def grow(self, nsims=10, gamma=0.9, depth=10):  # TODO update grow to reflect SOPCC
        """
        Grow Tree by one node

        - nsims is number of sim rollouts to run
        - gamma for D-UCT values
        - depth is how many rollout steps to run
        """

        # SELECTION
        start_node = 1

        # Sample actions of other robots (also this robot's action seq at start_node)
        # NOTE: Sampling done at the begining for dependency graph reasons
        system_state = self._get_system_state(start_node)

        # Propagate down the tree
        # TODO? check how _select handles mu, N = 0
        while len(self._childNodes(start_node)) > 0:
            start_node = self._select(self._childNodes(start_node))

        # EXPANSION
        # TODO? check if _expansion changes start_node to the node after jumping
        self._expansion(start_node)

        # SIMULATION # TODO Sampling-based sims for stochasticity
        avg_reward = 0
        best_reward = float("-Inf")
        best_rollout = None
        for i in range(nsims):
            robot_temp_state = self.graph.nodes[start_node]["state"]
            system_state[self.id] = robot_temp_state
            cum_reward = 0
            d = 0  # depth
            while d < depth:  # also breaks at no available options
                d += 1

                # Get the available actions
                options = self.sim_available_actions(
                    self.data,
                    system_state,
                    self.id
                )

                # If no actions possible, simulation complete
                if len(options) == 0:
                    break

                # NOTE introduced some randomness here using stoch. edge sampling for selecting next task. Should produce randomness over range nsims and lead to different rollouts
                sim_action = self.sim_selection_func(
                    self.data, options, robot_temp_state)

                # add that to the actions of the current robot
                # NOTE introduced some randomness here using stoch. edge sampling for setting remaining_budget. May instead want to take an average of samples?
                robot_temp_state = self.state_store(
                    self.data, robot_temp_state, sim_action, self.id)
                system_state[self.id] = robot_temp_state

            # calculate the reward at the end of simulation using "local utiliity" reward as described in paper
            # TODO incorporate failure here for sopcc
            rew = self.reward(self.data, system_state, self.id)
            cum_reward += rew
            # if best reward so far, store the rollout in the new node
            if rew > best_reward:
                best_reward = rew
                best_rollout = copy(robot_temp_state)

        avg_reward = cum_reward / nsims

        # TODO backprop logic from SOPCC
        self.graph.nodes[start_node]["avg_reward"] = avg_reward
        self.graph.nodes[start_node]["best_reward"] = best_reward
        self.graph.nodes[start_node]["N"] = 1
        self.graph.nodes[start_node]["best_rollout"] = copy(best_rollout)

        # BACKPROPOGATION # TODO integrate backprop logic from SOPCC
        while start_node != 1:  # while not root node

            start_node = self._parent(start_node)

            self.graph.nodes[start_node]["avg_reward"] = \
                (gamma * self.graph.nodes[start_node]["avg_reward"] *
                 self.graph.nodes[start_node]["N"] + avg_reward) \
                / (self.graph.nodes[start_node]["N"] + 1)

            self.graph.nodes[start_node]["N"] = \
                gamma * self.graph.nodes[start_node]["N"] + 1

            if best_reward > self.graph.nodes[start_node]["best_reward"]:
                self.graph.nodes[start_node]["best_reward"] = best_reward
                self.graph.nodes[start_node]["best_rollout"] = copy(
                    best_rollout)

        self._update_distribution()

        return avg_reward

    def send_comms(self):
        # print("Robot", self.id, "Sending:\n", self.my_act_dist)
        return self.my_act_dist

    def receive_comms(self, comms_in, robot_id):
        """
        Save data which has been communicated to this tree
        Only receives from one robot at a time, call once
        for each robot

        Inputs:
        - comms_in
            - An Action distribution object
        - robot_id
            - Robot number/id - used as key for comms
        """
        # print("Robot", self.id, "Received:\n",
        #   comms_in, "from robot", robot_id)
        self.comms[robot_id] = comms_in
        return True

    # TODO Currently the probability distribution of the Action sequences being communicated is not being calculated as described in the paper. The probability distribution is simply being set as proportional to the local reward function
