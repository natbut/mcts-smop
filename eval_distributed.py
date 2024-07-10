from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from my_DecMCTS import Tree
from utils.graphing import create_sop_instance


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
    # NOTE More intelligent selection choice (See SOPCC)
    # from DecMCTS: Select next edge that does not exceed travel budget and maximizes ratio of increased reward to edge cost
    idx_max = np.argmax([data["graph"].rewards[o] /
                        data["graph"].get_mean_cost((state.action_seq[-2], o)) for o in options])

    return options[idx_max]


if __name__ == "__main__":

    # === PROBLEM PARAMS ===
    problem_sizes = [20]
    edges_mean_range = (3, 10)
    c = 0.75  # variance modifier (0.05=low, 0.25=med, 0.75=high)
    trials = 5
    total_tests = 6
    num_robots = 8

    start = "vs"
    end = "vg"
    budget = 25
    planning_iters = 10
    comm_n = 5  # Number of Action Sequences to communicate
    failure_probability = 0.1  # for stoch

    # === EVALUATION ===
    det_results = {}  # store results from det env
    stoch_results = {}  # store results from stoch env
    for size in problem_sizes:
        det_results[size] = []
        stoch_results[size] = []
        stoch = False
        for test in range(total_tests):
            # Generate problem instance
            seed = np.random.randint(100)
            graph = create_sop_instance(size,
                                        edges_mean_range,
                                        c,
                                        rand_seed=seed)
            # Divide into det and stoch tests
            if test > total_tests/2:
                stoch = True

            print("\n", size, "Tasks: Test", test, " | Stoch:", stoch)
            for trial in range(trials):
                print("Trial", trial)
                # Generate robots (trees)
                data = {}
                paths = {}
                stored_comms = {}
                for r in range(num_robots):
                    # data required to calculate global reward and available actions
                    data[r] = {"graph": deepcopy(
                        graph), "start": start, "end": end, "budget": budget, "fail_prob": failure_probability}

                    paths[r] = [start]

                    stored_comms[r] = {}

                # === ALGORITHM 1 ===
                operating = True
                # while still traveling and energy remains
                while operating:
                    # Initialize MCTS tree instances for each robot
                    trees = {}
                    for r in range(num_robots):
                        trees[r] = Tree(data[r],
                                        reward,
                                        avail_actions,
                                        state_storer,
                                        sim_select_action,
                                        sim_get_actions_available,
                                        comm_n=comm_n,
                                        robot_id=r)

                        trees[r].comms = stored_comms[r]

                    # Alg1. SELECT SET OF SEQUENCES?
                    # 10 paths resampled every 10 iterations (per robot)

                    for _ in range(planning_iters):
                        # Alg1. GROW TREE & UPDATE DISTRIBUTION
                        for r in range(num_robots):
                            trees[r].grow()
                        # Alg1. COMMS TRANSMIT & COMMS RECEIVE
                        for i in range(num_robots):
                            for j in range(num_robots):
                                if i != j:
                                    trees[i].receive_comms(
                                        trees[j].send_comms(), j)
                        # Alg1. Cool?

                    # Alg1. Return highest-prob action sequence (best)
                    for r in range(num_robots):
                        best_act_seq = trees[r].my_act_dist.best_action(
                        ).action_seq

                        # End Alg. 1 - Execute first action in sequence
                        if best_act_seq[0] == data[r]["end"]:  # cont if at goal
                            continue
                        # update budgets
                        if stoch:
                            data[r]["budget"] -= data[r]["graph"].get_stoch_cost(
                                (best_act_seq[0], best_act_seq[1]))
                        else:
                            data[r]["budget"] -= data[r]["graph"].get_mean_cost(
                                (best_act_seq[0], best_act_seq[1]))
                        # update rewards
                        # rew1.append(rew1[-1] + data1["graph"].rewards[best_act1[0]])
                        # update locations
                        data[r]["start"] = best_act_seq[1]
                        paths[r].append(best_act_seq[1])
                        # update graphs
                        data[r]["graph"].vertices.remove(best_act_seq[0])

                        # store other robots' action distributions for new tree
                        stored_comms[r] = trees[r].comms

                    # Check if still operating (end if all robots are at vg)
                    operating = False
                    for r in range(num_robots):
                        if data[r]["start"] != data[r]["end"]:
                            operating = True

                # Compute global reward upon completion
                all_visits = []
                for r in range(num_robots):
                    # NOTE: We are only receiving reward for robots that survived their trips
                    if data[r]["budget"] >= 0:
                        all_visits += paths[r]
                # print("All Visits:", all_visits)
                unique_visits = set(all_visits)
                print("Unique Visits:", unique_visits,
                      " | Total:", len(unique_visits)-2, " of", size)  # subt 2 for vs and vg

                test_reward = sum(graph.rewards[v] for v in unique_visits)
                # print("Global Reward:", test_reward)

                # Add to reward averages for results NOTE: Currently tracking unique visits
                if stoch:
                    # stoch_results[size].append(test_reward)
                    stoch_results[size].append(len(unique_visits))
                else:
                    # det_results[size].append(test_reward)
                    det_results[size].append(len(unique_visits))

    # Plot results
    fig, ax = plt.subplots()
    x = ["Det Sol in Det Env", "Det Sol in Stoch Env"]
    avg_det = 0
    print("Det results:", det_results)
    print("Stoch results:", stoch_results)
    for size in det_results:
        avg_det = sum(
            res for res in det_results[size]) / len(det_results[size])
    avg_stoch = 0
    for size in stoch_results:
        avg_stoch = sum(
            res for res in stoch_results[size]) / len(stoch_results[size])

    avg_tasks = [avg_det, avg_stoch]

    ax.bar(x, avg_tasks)

    plt.show()
