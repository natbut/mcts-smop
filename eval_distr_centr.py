
import matplotlib.pyplot as plt
import numpy as np

from solvers.decMCTS_config import *
from solvers.my_DecMCTS import Tree
from solvers.sim_brvns import sim_brvns
from utils.graphing import create_sop_instance

if __name__ == "__main__":

    # === PROBLEM PARAMS ===
    problem_size = 25
    edges_mean_range = (3, 10)
    reward_range = (1, 1)
    c = 0.75  # variance modifier (0.05=low, 0.25=med, 0.75=high)
    trials = 10
    num_robots = 4
    budget = 25
    start = "vs"
    end = "vg"
    planning_iters = 10
    comm_n = 5  # Number of Action Sequences to communicate in distr
    failure_probability = 0.1  # for stoch

    # Sim-BRVNS prams
    alpha = 0.5  # (0,1), depending on reward heterogeneity
    # (0,1), controls relative level of greediness present in randomized constructive heuristic
    beta = 0.3
    k_initial = 1
    k_max = 10
    t_max = 200  # Maximum execution time
    exploratory_mcs_iters = 10
    intensive_mcs_iters = 100

    # === EVALUATION ===
    distr_det_results = []  # 0
    distr_stoch_results = []  # 1
    centr_det_results = []  # 2
    centr_stoch_results = []  # 3

    test_case = 0
    for trial in range(trials):
        print("Trial", trial)
        seed = np.random.randint(100)
        graph = create_sop_instance(problem_size,
                                    edges_mean_range,
                                    c,
                                    reward_range,
                                    rand_seed=seed)

        # === CENTRALIZED TEST (1 det, 1 stoch) ===
        print("Solving centralized...")
        solution, reliability = sim_brvns(graph,
                                          budget,
                                          num_robots,
                                          end,
                                          alpha,
                                          beta,
                                          k_initial,
                                          k_max,
                                          t_max,
                                          exploratory_mcs_iters,
                                          intensive_mcs_iters
                                          )
        # Eval solution in det and stoch
        # For each path in solution, sample travel times. If sum of sampled times is within budget, then add path vertices to "all_visited" list
        print("Evaluating centralized solutions...")
        all_det_visits = []
        all_stoch_visits = []
        for path in solution:
            det_cost = sum(graph.get_mean_cost_edgeWork(
                (path[i], path[i+1])) for i in range(len(path)-1))
            stoch_cost = sum(graph.get_stoch_cost_edgeWork(
                (path[i], path[i+1])) for i in range(len(path)-1))
            if det_cost <= budget:
                all_det_visits += path
            if stoch_cost <= budget:
                all_stoch_visits += path

        # Compute reward from unique visits for each list and store
        centr_det_results.append(
            sum(graph.rewards[v] for v in set(all_det_visits)) / problem_size)
        centr_stoch_results.append(
            sum(graph.rewards[v] for v in set(all_stoch_visits)) / problem_size)

        # == DISTRIBUTED TEST (1 det, 1 stoch) ===
        stoch = False
        for st in range(2):
            if st == 1:
                stoch = True
                print("Solving distributed with stoch execution ...")
            else:
                print("Solving distributed with det execution ...")
            # Generate robots (trees)
            data = {}  # store data for planning with each robot
            paths = {}  # store paths executed by each robot
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
                                    local_util_reward,
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
            unique_visits = global_reward_from_solution(
                paths, graph, num_robots, data[r]["budget"])
            # print("Global Reward:", test_reward)

            # Add to reward averages for results NOTE: Currently rewarding unique visits
            if stoch:
                # stoch_results[size].append(test_reward)
                distr_stoch_results.append(unique_visits / problem_size)
            else:
                # det_results[size].append(test_reward)
                distr_det_results.append(unique_visits / problem_size)

    # Plot results
    fig, ax = plt.subplots()
    x = ["DecMCTS in Det", "DecMCTS in Stoch",
         "Sim-BRVNS in Det", "Sim-BRVNS in Stoch"]
    # Mean
    distr_avg_det = np.mean(distr_det_results)
    distr_avg_stoch = np.mean(distr_stoch_results)
    centr_avg_det = np.mean(centr_det_results)
    centr_avg_stoch = np.mean(centr_stoch_results)
    # StdErr
    distr_se_det = np.std(distr_det_results) / np.sqrt(len(distr_det_results))
    distr_se_stoch = np.std(distr_stoch_results) / \
        np.sqrt(len(distr_stoch_results))
    centr_se_det = np.std(centr_det_results) / np.sqrt(len(centr_det_results))
    centr_se_stoch = np.std(centr_stoch_results) / \
        np.sqrt(len(centr_stoch_results))

    avg_tasks = [distr_avg_det, distr_avg_stoch,
                 centr_avg_det, centr_avg_stoch]
    error_bars = [distr_se_det, distr_se_stoch, centr_se_det, centr_se_stoch]

    ax.bar(x, avg_tasks, yerr=error_bars, capsize=5,
           color=['blue', 'lightblue', 'green', 'lightgreen'])

    ax.set_ylabel('Percent Task Completion')
    ax.set_title('Comparison of Task Completion between Algorithms')

    plt.show()
