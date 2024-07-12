import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from control.agent import Agent, generate_agents_with_data
from control.mothership import Mothership
from sim.comms_manager import CommsManager_Basic
from solvers.sim_brvns import sim_brvns
from utils.graphing import Graph, create_sop_instance


def calculate_reward(graph: Graph, agent_list: list[Agent]):
    # Return sum of reward for each unique task visited (duplicates don't reward)
    # print("REWARD CALC")
    all_tasks_visited = []
    for a in agent_list:
        if not a.dead:
            all_tasks_visited += a.completed_tasks
    # print("all tasks visited:", all_tasks_visited)
    unique_tasks_visited = set(all_tasks_visited)
    # print("unique tasks visited:", unique_tasks_visited)
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited) / len(graph.vertices)


 # Define framework for doing DecMCTS with multiple agents on generic graph 
if __name__ == "__main__":

    print("Initialize Simulation")

    # === DEFINE EVALUATION PARAMETERS ===
    problem_size = 20
    edges_mean_range = (3, 10)
    reward_range = (1, 1)
    c = 0.75  # variance modifier (0.05=low, 0.25=med, 0.75=high)
    num_robots = 8
    budget = 20
    trials = 5
    # == Define each type of test that will run on each test case (trial)
    sim_configs = ["Cent | Det",
                   "Cent | Stoch",
                   "Dist | Det",
                   "Dist | Stoch",
                   ]

    # === DEFINE SOLVER PARAMETERS ===
    # == Dec-MCTS Params
    planning_iters = 10  # TODO
    comm_n = 5  # Number of Action Sequences to communicate in distr
    failure_probability = 0.1  # for stoch
    # == Sim-BRVNS Params
    # (0,1), depending on reward heterogeneity (0 for high heter, 1 for low)
    alpha = 0.85
    beta = 0.3  # (0,1), controls greediness in constructive heuristic
    k_initial = 1
    k_max = 10
    t_max = 200  # Maximum execution time
    exploratory_mcs_iters = 10
    intensive_mcs_iters = 100

    # === TRACKING FOR EVALUATION ===
    distr_det_results = []
    distr_det_percentDead = []
    distr_stoch_results = []
    distr_stoch_percentDead = []
    centr_det_results = []
    centr_det_percentDead = []
    centr_stoch_results = []
    centr_stoch_percentDead = []

    for tr in range(trials):
        print("\n==== Trial", tr, " ====")
        # === INITIALIZE SIMULATION ENVIRONMENT ===
        # Create problem instance
        graph = create_sop_instance(problem_size,
                                    edges_mean_range,
                                    c,
                                    reward_range,
                                    rand_seed=np.random.randint(100))
        tasks_work = {}
        for v in graph.vertices:
            tasks_work[v] = 0  # TODO roll work into graph cost in future
        start = "vs"
        end = "vg"

        # == Bundle Data
        dec_mcts_data = {"graph": deepcopy(graph),
                         "start": start,
                         "end": end,
                         "budget": budget,
                         "num_robots": num_robots,
                         "fail_prob": failure_probability,
                         "comm_n": comm_n,
                         "plan_iters": planning_iters
                         }
        sim_brvns_data = {"graph": deepcopy(graph),
                          "start": start,
                          "end": end,
                          "budget": budget,
                          "num_robots": num_robots,
                          "alpha": alpha,
                          "beta": beta,
                          "k_initial": k_initial,
                          "k_max": k_max,
                          "t_max": t_max,
                          "explore_iters": exploratory_mcs_iters,
                          "intense_iters": intensive_mcs_iters
                          }

        for sim_config in sim_configs:
            print("Running", sim_config, "...")
            # Run simulation (planning, etc.)

            # Create agents
            agent_list = generate_agents_with_data(dec_mcts_data, tasks_work)
            mothership = Mothership(-1, sim_brvns_data, tasks_work)

            # Create comms framework (TBD - may be ROS)
            # TODO set up comms failures
            comms_mgr = CommsManager_Basic(agent_list)

            # == If Centralized, Generate Plan & Load on Agents
            if "Cent" in sim_config:
                # Plan on mothership
                print("Solving schedules...")
                initial_solution = mothership.solve_schedules()
                print("Executing schedules...")

                # Communicate plan to agents
                for target in agent_list:
                    mothership.send_message(
                        comms_mgr, target.id, initial_solution[target.id])
                schedules_shared = False
                while not schedules_shared:
                    comms_mgr.step()
                    schedules_shared = True
                    for a in agent_list:
                        if a.schedule == None:
                            schedules_shared = False

            # Run Sim
            running = True
            i = 0
            while running:
                print("TIME STEP:", i)
                for a in agent_list:
                    # Each time action is complete, do rescheduling (if distr)
                    if a.event and "Dist" in sim_config:
                        # Solve for new action distro
                        a.optimize_schedule()
                        # Send new action dist to other agents
                        for target in agent_list:
                            if target.id != a.id:
                                a.send_message(comms_mgr, target.id,
                                               a.action_dist)

                    # Update current action
                    a.action_update(sim_config)
                    a.reduce_energy_basic()

                # Update message passing, also update comms graph
                comms_mgr.step()

                # Check stopping criteria
                running = False
                for a in agent_list:
                    if not (a.dead or a.finished):
                        running = True

                i += 1

            print("Done")

            # Store results
            reward = calculate_reward(graph, agent_list)
            if sim_config == "Dist | Det":
                distr_det_results.append(reward)
                distr_det_percentDead.append(
                    sum(a.dead for a in agent_list) / num_robots)
            elif sim_config == "Dist | Stoch":
                distr_stoch_results.append(reward)
                distr_stoch_percentDead.append(
                    sum(a.dead for a in agent_list) / num_robots)
            elif sim_config == "Cent | Det":
                centr_det_results.append(reward)
                centr_det_percentDead.append(
                    sum(a.dead for a in agent_list) / num_robots)
            elif sim_config == "Cent | Stoch":
                centr_stoch_results.append(reward)
                centr_stoch_percentDead.append(
                    sum(a.dead for a in agent_list) / num_robots)

    print("DEAD Distr DET:", np.mean(distr_det_percentDead),
          "\nDEAD Distr STOCH:", np.mean(distr_stoch_percentDead),
          "\nDEAD Centr DET:", np.mean(centr_det_percentDead),
          "\nDEAD Centr STOCH:", np.mean(centr_stoch_percentDead))

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
    ax.set_ybound(0.0, 1.0)

    plt.show()
