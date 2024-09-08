from copy import deepcopy

import numpy as np

from solvers.graphing import Graph


class State:
    """
    Object stored at each node (defines state of solution, not environment)
    """

    def __init__(self, act_seq, budget):
        self.action_seq = act_seq  # ordered list of tasks to visit
        # Expected budget remaining once act_seq is executed (USE ONLY FOR DEC-MCTS)
        # self.remaining_budget = budget
        self.age = 1

    def __str__(self):
        # + " | exp_rem_budget:" + str(self.remaining_budget)
        return "schedule:" + str(self.action_seq)


# === GENERAL HELPER FUNCTIONS ===

def routes_det_reward(solution, graph: Graph, budget):
    """
    Evaluate cost of each route (list of vertices) in solution using graph. If cost is within budget, add rewards from route to rewards sum. Return sum.
    """
    all_tasks_visited = []
    for route in solution:
        if route_det_cost(route, graph) < budget:
            all_tasks_visited += route
    unique_tasks_visited = set(all_tasks_visited)
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited)


def route_det_cost(route, graph: Graph):
    """
    Return sum of expected mean time to traverse edges and execute tasks in route
    """
    return sum(graph.get_mean_cost_edgeWork((route[i], route[i+1])) for i in range(len(route)-1))


def routes_stoch_reward(solution, graph: Graph, budget):
    """
    Evaluate cost of each route (list of vertices) in solution using graph. If cost is within budget, add rewards from route to rewards sum. Return sum.
    """
    all_tasks_successfully_visited = []
    fail = 0
    for route in solution:
        # Only apply tasks if route is a success
        if route_stoch_cost(route, graph) < budget:
            all_tasks_successfully_visited += route
        else:
            fail = 1
    unique_tasks_visited = set(all_tasks_successfully_visited)
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited), fail


def route_stoch_cost(route, graph: Graph):
    """
    Return sum of sampled time to traverse edges and execute tasks in route
    """
    return sum(graph.get_stoch_cost_edgeWork((route[i], route[i+1])) for i in range(len(route)-1))


def fast_simulation(solution, graph: Graph, budget, iterations):
    """
    Solution here is list of routes [[vs, v1, v2, vg], [vs, v3, v4, vg], ...]
    Get reward through MCS approach. Return average reward and reliability (percent success)
    """
    rewards = []
    fails = 0
    for _ in range(iterations):
        rew, fail = routes_stoch_reward(solution, graph, budget)
        rewards.append(rew)
        fails += fail
    return sum(rew for rew in rewards) / iterations, (iterations - fails) / iterations


def intensive_simulation(elite_solutions, graph, budget, iterations):
    """
    Solution here is list of routes [[vs, v1, v2, vg], [vs, v3, v4, vg], ...]
    Get reward through MCS approach. Return average reward and reliability (percent success)
    """
    # NOTE Same as fast simulation for now
    return fast_simulation(elite_solutions, graph, budget, iterations)


def calculate_final_potential_reward(task_dict, agent_list):
    # Return sum of reward for each unique task visited (duplicates don't reward)
    all_tasks_visited = []
    for a in agent_list:
        all_tasks_visited += a.glob_completed_tasks
    unique_tasks_visited = set(all_tasks_visited)
    return sum(task_dict[task_id].reward for task_id in unique_tasks_visited) / sum(task_dict[task_id].reward for task_id in task_dict)


def calculate_final_reward(task_dict, agent_list):
    """
    Return sum of reward for each unique task visited (duplicates don't reward)
    """
    all_tasks_visited = []
    for a in agent_list:
        if not a.dead:
            all_tasks_visited += a.glob_completed_tasks
    unique_tasks_visited = set(all_tasks_visited)
    return sum(task_dict[task_id].reward for task_id in unique_tasks_visited) / sum(task_dict[task_id].reward for task_id in task_dict)


def get_state_reward(test_state, task_dict):

    all_tasks_visited = test_state.action_seq[:]

    unique_visited = set(all_tasks_visited)

    return sum(task_dict[id].reward for id in unique_visited)


def sim_util_reward(test_state: State,
                    act_dists,
                    rob_id,
                    task_dict,
                    sim_iters,
                    ):
    """
    Given a state to test and a dictionary of action distributions, perform 
    sampling cycles to evaluate the local utility of the test state.

    Consider also whether tasks have been completed
    """
    rews = []
    for _ in range(sim_iters):

        # Collect visited tasks with and without robot_i
        tasks_without_robot_i = []
        # Don't count starting tasks
        all_tasks_visited = test_state.action_seq[:]
        for id in act_dists.keys():
            scheduled = act_dists[id].random_action().action_seq[:]
            if id != rob_id:
                tasks_without_robot_i += scheduled
            all_tasks_visited += scheduled

        # Also include other completed tasks to reward schedules that don't double-visit tasks
        for id, task in task_dict.items():
            if task.complete:
                if id not in all_tasks_visited:
                    tasks_without_robot_i.append(id)
                    all_tasks_visited.append(id)

        unique_visited = set(all_tasks_visited)
        unique_visited_without = set(tasks_without_robot_i)

        util_rew = sum(task_dict[id].reward for id in unique_visited) - sum(
            task_dict[id].reward for id in unique_visited_without)
        rews.append(util_rew)

    return max(1.0, np.average(rews))


def local_util_reward(data: dict, states: dict[State], rob_id):
    """
    Returns "utility" of rob_id tour, calculated as difference in global reward with and without tour.
    """
    # Return sum of reward for each unique task visited (duplicates don't reward)
    # NOTE Don't reward tours that exceed budget
    task_dict = data["task_dict"]

    reward_with = []
    reward_without = []
    for _ in range(data["sim_iters"]):

        all_tasks_visited = []
        tasks_without_robot_i = []
        for robot in states:
            # if states[robot].remaining_budget > 0:
            if robot != rob_id:
                tasks_without_robot_i += states[robot].action_seq[:]
            all_tasks_visited += states[robot].action_seq[:]
        unique_tasks_visited = set(all_tasks_visited)
        unique_tasks_visited_without = set(tasks_without_robot_i)

        reward_with.append(
            sum(task_dict[task_id].reward for task_id in unique_tasks_visited))
        reward_without.append(
            sum(task_dict[task_id].reward for task_id in unique_tasks_visited_without))

    return np.linalg.norm(np.array(reward_with) - np.array(reward_without))


# === DEC-MCTS HELPER FUNCTIONS ===


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
    # state.remaining_budget = data["budget"] - cost
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
            # last_action = state.action_seq[-2]
            # cost = graph.get_mean_cost_edgeWork(
            #     (last_action, task)) + graph.get_mean_cost_edgeWork((task, data["end"]))
            # if cost < state.remaining_budget:
            #     choices.append(task)

            test_route = state.action_seq[:]
            test_route.insert(-1, task)

            # print("Avail task", task, " in route", test_route)

            # last_action = robot_state.action_seq[-2]
            # cost = graph.get_mean_cost_edgeWork(
            #     (last_action, task)) + graph.get_mean_cost_edgeWork((task, data["end"]))
            cost = route_det_cost(test_route, graph)
            if cost < data["budget"]:
                choices.append(task)
    # print("Given actions", state.action_seq, " we have choices: ", choices)
    return choices


def sim_get_actions_available(data: dict, states: dict[State], rob_id: int):
    """
    Return available actions during simulation (FOR ROLLOUT)
    """
    # Return list of unallocated tasks. As the graph is complete, all tasks can be returned. In the future, may only return a subset of tasks that neighbor the final task in rob_id's action sequence

    # Restrict available actions to those that do not exceed travel budget (is how Dec-MCTS paper formulated it). However, the paper does not enforce returning to a goal location. So we want to restrict to actions where cost to travel to action + cost from action to goal does not exceed remaining budget.
    # NOTE: This may be handled differently once stochastic edges are introduced
    graph = data["graph"]
    robot_state = states[rob_id]

    all_tasks_allocated = []
    for robot in states:
        all_tasks_allocated += states[robot].action_seq
    # print("all tasks visited:", all_tasks_visited)
    unique_tasks_allocated = set(all_tasks_allocated)

    # print("Tasks allocated:", unique_tasks_allocated)
    reachable_unallocated_tasks = []
    for task in graph.vertices:
        if task not in unique_tasks_allocated:
            test_route = robot_state.action_seq[:]
            test_route.insert(-1, task)

            # print("Testing task", task, " in route", test_route)

            # last_action = robot_state.action_seq[-2]
            # cost = graph.get_mean_cost_edgeWork(
            #     (last_action, task)) + graph.get_mean_cost_edgeWork((task, data["end"]))
            cost = route_det_cost(test_route, graph)
            if cost < data["budget"]:
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
