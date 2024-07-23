from copy import deepcopy

import numpy as np


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


def state_storer(data: dict, parent_state, action, id):
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
        cost += data["graph"].get_stoch_cost_edgeWork(edge)
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
            cost = graph.get_mean_cost_edgeWork(
                (last_action, task)) + graph.get_mean_cost_edgeWork((task, data["end"]))
            if cost < state.remaining_budget:
                choices.append(task)
    # print("Given actions", state.action_seq, " we have choices: ", choices)
    return choices


def local_util_reward(data: dict, states: dict[State], rob_id):
    """
    Create a reward function. This is the global reward given a list of  actions taken by the current robot, and every other robot states is a dictionary with keys being robot IDs, and values are the object returned by the state_storer function you provide
    """
    # Return sum of reward for each unique task visited (duplicates don't reward)
    # TODO improve reward function to prioritize reliability alongside risk
    # TODO Check against local utility function defined in paper
    # NOTE Don't reward tours that exceed budget
    all_tasks_visited = []
    tasks_without_robot_i = []
    for robot in states:
        if states[robot].remaining_budget > 0:
            if robot != rob_id:
                tasks_without_robot_i += states[robot].action_seq
            all_tasks_visited += states[robot].action_seq
    # print("all tasks visited:", all_tasks_visited)
    unique_tasks_visited = set(all_tasks_visited)
    unique_tasks_visited_without = set(tasks_without_robot_i)
    # print("unique tasks visited:", unique_tasks_visited)
    graph = data["graph"]
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited) - sum(graph.rewards[task_id] for task_id in unique_tasks_visited_without)


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
            cost = graph.get_mean_cost_edgeWork(
                (last_action, task)) + graph.get_mean_cost_edgeWork((task, data["end"]))
            if cost < robot_state.remaining_budget:
                reachable_unallocated_tasks.append(task)

    return reachable_unallocated_tasks


def sim_select_action(data: dict, options: list, state: State):
    """
    Choose an available option during simulation (can be random)
    FOR ROLLOUT
    """
    # NOTE Use more intelligent selection choice (See SOPCC)
    # from DecMCTS: Select next edge that does not exceed travel budget and maximizes ratio of increased reward to edge cost
    idx_max = np.argmax([data["graph"].rewards[o] /
                        data["graph"].get_stoch_cost_edgeWork((state.action_seq[-2], o)) for o in options])

    return options[idx_max]


def global_reward_from_solution(solution, graph, num_robots, rem_budget):
    # NOTE unused??
    all_visits = []
    for r in range(num_robots):
        # NOTE: We are only receiving reward for robots that survived their trips
        if rem_budget >= 0:
            all_visits += solution[r]
    # print("All Visits:", all_visits)
    unique_visits = set(all_visits)
    # print("Unique Visits:", unique_visits,
    #       " | Total:", len(unique_visits)-2, " of", problem_size)  # subt 2 for vs and vg

    return sum(graph.rewards[v] for v in unique_visits)
