import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from control.agent import Agent, generate_agents_with_data
from control.mothership import generate_mothership_with_data
from sim.comms_manager import CommsManager_Basic
from utils.graphing import Graph, create_dummy_graph, create_sop_instance


def calculate_global_potential_reward(graph: Graph, agent_list: list[Agent]):
    # Return sum of reward for each unique task visited (duplicates don't reward)
    all_tasks_visited = []
    for a in agent_list:
        all_tasks_visited += a.completed_tasks
    unique_tasks_visited = set(all_tasks_visited)
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited) / len(graph.vertices)


def calculate_global_reward(graph: Graph, agent_list: list[Agent]):
    # Return sum of reward for each unique task visited (duplicates don't reward)
    all_tasks_visited = []
    for a in agent_list:
        if not a.dead:
            all_tasks_visited += a.completed_tasks
    unique_tasks_visited = set(all_tasks_visited)
    return sum(graph.rewards[task_id] for task_id in unique_tasks_visited) / len(graph.vertices)

 # Define framework for doing DecMCTS with multiple agents on generic graph
if __name__ == "__main__":

    print("Initialize Simulation")

    # === DEFINE EVALUATION PARAMETERS ===
    trials = 5
    problem_size = 20
    edges_mean_range = (3, 10)
    reward_range = (1, 1)
    c = 0.75  # variance modifier (0.05=low, 0.25=med, 0.75=high)
    num_robots = 6
    budget = 20
    velocity = 1
    start = "vs"
    end = "vg"

    # === DEFINE SOLVER PARAMETERS ===
    # == Dec-MCTS Params
    planning_iters = 100  # TODO try increasing this
    comm_n = 5  # Number of Action Sequences to communicate in distr
    failure_probability = 0.1  # for stoch
    # == Sim-BRVNS Params
    # (0,1), depending on reward heterogeneity (0 for high heter, 1 for low)
    alpha = 0.85
    beta = 0.3  # (0,1), controls greediness in constructive heuristic
    k_initial = 1
    k_max = 10
    t_max = 100  # Maximum execution time
    exploratory_mcs_iters = 10
    intensive_mcs_iters = 100

    # === TRACKING FOR EVALUATION ===
    # distr_det_results = []
    # distr_det_percentDead = []
    best_results = []
    best_percentDead = []
    centr_stoch_results = []
    centr_stoch_potential = []
    centr_stoch_percentDead = []
    distr_stoch_results = []
    distr_stoch_potential = []
    distr_stoch_percentDead = []
    hybrid1_stoch_results = []
    hybrid1_stoch_potential = []
    hybrid1_stoch_percentDead = []
    hybrid2_stoch_results = []
    hybrid2_stoch_potential = []
    hybrid2_stoch_percentDead = []
    # hybrid1_det_results = []
    # hybrid2_det_percentDead = []

    # == Define each type of test that will run on each test case (trial)
    sim_configs = [  # "Hybrid | Det",
        "Cent | Best",
        "Cent | Stoch",
        "Dist | Stoch",
        "Hybrid1 | Stoch",
        "Hybrid2 | Stoch",
        #    "Dist | Det",
    ]

    for tr in range(trials):
        print("\n==== Trial", tr, " ====")
        # === INITIALIZE SIMULATION ENVIRONMENT ===
        # Create problem instance
        true_graph = create_sop_instance(problem_size,
                                         edges_mean_range,
                                         c,
                                         reward_range,
                                         rand_seed=np.random.randint(100))

        # Create planning graph from true graph
        planning_graph = create_dummy_graph(true_graph, c)

        # Data for best solution with Sim-BRVNS on true graph

        # == Bundle Data
        best_sim_data = {"graph": deepcopy(true_graph),
                         "start": start,
                         "end": end,
                         "budget": budget,
                         "velocity": velocity,
                         "basic": True
                         }

        sim_data = {"graph": deepcopy(planning_graph),
                    "start": start,
                    "end": end,
                    "budget": budget,
                    "velocity": velocity,
                    "basic": True
                    }
        dec_mcts_data = {"num_robots": num_robots,
                         "fail_prob": failure_probability,
                         "comm_n": comm_n,
                         "plan_iters": planning_iters
                         }
        sim_brvns_data = {"num_robots": num_robots,
                          "alpha": alpha,
                          "beta": beta,
                          "k_initial": k_initial,
                          "k_max": k_max,
                          "t_max": t_max,
                          "explore_iters": exploratory_mcs_iters,
                          "intense_iters": intensive_mcs_iters
                          }

        for sim_config in sim_configs:
            print("\n === Running", sim_config, "...")
            # Run simulation (planning, etc.)

            # Create agents
            agent_list = generate_agents_with_data(dec_mcts_data, sim_data)
            mothership = generate_mothership_with_data(
                sim_brvns_data, sim_data)
            best_mothership = generate_mothership_with_data(
                sim_brvns_data, best_sim_data)

            # Create comms framework (TBD - may be ROS)
            # TODO set up comms failures
            comms_mgr = CommsManager_Basic(agent_list)

            # == If Centralized, Generate Plan & Load on Agents
            if "Cent" in sim_config or "Hybrid" in sim_config:
                # Plan on mothership
                print("Solving schedules...")
                if "Best" in sim_config:
                    # For baseline, plan over true graph
                    best_mothership.solve_STOP_schedules(comms_mgr, agent_list)
                else:
                    # For other testing, plan over planning graph
                    mothership.solve_STOP_schedules(comms_mgr, agent_list)
                print("Executing schedules...")

            # Run Sim
            running = True
            i = 0
            while running:
                print("== TIME STEP:", i)
                for a in agent_list:
                    print("= Agent", a.id)
                    # Each time action is complete, do rescheduling (if distr)
                    if a.event and ("Dist" in sim_config or "Hybrid" in sim_config) and not a.finished:
                        if "Hybrid2" in sim_config:
                            # Solve for centralized action update
                            print("Solving hybrid schedule...")
                            mothership.solve_new_tour_single(comms_mgr,
                                                             a,
                                                             agent_list
                                                             )
                        # Solve for new action distro
                        a.optimize_schedule(comms_mgr, agent_list)

                    # Update current action
                    a.action_update(true_graph, sim_config)
                    a.reduce_energy_basic()

                # Update message passing, also update comms graph
                comms_mgr.step()

                # Check stopping criteria
                running = False
                for a in agent_list:
                    if not (a.dead or a.finished):
                        running = True

                i += 1
                # time.sleep(0.1)

            print("Done")

            # Store results
            reward = calculate_global_reward(true_graph, agent_list)
            potent = calculate_global_potential_reward(true_graph, agent_list)
            percentDead = sum(a.dead for a in agent_list) / num_robots
            # if sim_config == "Dist | Det":
            #     distr_det_results.append(reward)
            #     distr_det_percentDead.append(
            #         sum(a.dead for a in agent_list) / num_robots)
            if sim_config == "Cent | Best":
                best_results.append(reward)
                best_percentDead.append(percentDead)
            elif sim_config == "Cent | Stoch":
                centr_stoch_results.append(reward)
                centr_stoch_potential.append(potent)
                centr_stoch_percentDead.append(percentDead)
            elif sim_config == "Dist | Stoch":
                distr_stoch_results.append(reward)
                distr_stoch_potential.append(potent)
                distr_stoch_percentDead.append(percentDead)
            # elif sim_config == "Hybrid | Det":
            #     hybrid_det_results.append(reward)
            #     hybrid_det_percentDead.append(
            #         sum(a.dead for a in agent_list) / num_robots)
            elif sim_config == "Hybrid1 | Stoch":
                hybrid1_stoch_results.append(reward)
                hybrid1_stoch_potential.append(potent)
                hybrid1_stoch_percentDead.append(percentDead)
            elif sim_config == "Hybrid2 | Stoch":
                hybrid2_stoch_results.append(reward)
                hybrid2_stoch_potential.append(potent)
                hybrid2_stoch_percentDead.append(percentDead)

    print(  # "DEAD Distr DET:", np.mean(distr_det_percentDead),
        "\nDEAD Centr BEST:", np.mean(best_percentDead),
        "\nDEAD Centr STOCH:", np.mean(centr_stoch_percentDead),
        "\nDEAD Distr STOCH:", np.mean(distr_stoch_percentDead),
        #   "\nDEAD Hybrid DET:", np.mean(hybrid_det_percentDead),
        "\nDEAD Hybrid1-DecMCTS STOCH:", np.mean(hybrid1_stoch_percentDead),
        "\nDEAD Hybrid2-Hybrid STOCH:", np.mean(hybrid2_stoch_percentDead))

    # Mean
    # distr_avg_det = np.mean(distr_det_results)
    best_avg = round(np.mean(best_results), 2)
    centr_avg_stoch = round(np.mean(centr_stoch_results), 2)
    centr_avg_potent = round(np.mean(centr_stoch_potential), 2)
    distr_avg_stoch = round(np.mean(distr_stoch_results), 2)
    distr_avg_potent = round(np.mean(distr_stoch_potential), 2)
    # hybrid_avg_det = np.mean(hybrid_det_results)
    hybrid1_avg_stoch = round(np.mean(hybrid1_stoch_results), 2)
    hybrid1_avg_potent = round(np.mean(hybrid1_stoch_potential), 2)
    hybrid2_avg_stoch = round(np.mean(hybrid2_stoch_results), 2)
    hybrid2_avg_potent = round(np.mean(hybrid2_stoch_potential), 2)
    # StdErr
    # distr_se_det = np.std(distr_det_results) / np.sqrt(len(distr_det_results))

    best_se = np.std(best_results) / np.sqrt(len(best_results))
    centr_se_stoch = np.std(centr_stoch_results) / \
        np.sqrt(len(centr_stoch_results))
    centr_se_stoch_potential = np.std(centr_stoch_potential) / \
        np.sqrt(len(centr_stoch_potential))

    distr_se_stoch = np.std(distr_stoch_results) / \
        np.sqrt(len(distr_stoch_results))
    distr_se_stoch_potential = np.std(distr_stoch_potential) / \
        np.sqrt(len(distr_stoch_potential))

    hybrid1_se_stoch = np.std(hybrid1_stoch_results) / \
        np.sqrt(len(hybrid1_stoch_results))
    hybrid1_se_stoch_potential = np.std(hybrid1_stoch_potential) / \
        np.sqrt(len(hybrid1_stoch_potential))
    hybrid2_se_stoch = np.std(hybrid2_stoch_results) / \
        np.sqrt(len(hybrid2_stoch_results))
    hybrid2_se_stoch_potential = np.std(hybrid2_stoch_potential) / \
        np.sqrt(len(hybrid2_stoch_potential))

    avg_rew = [centr_avg_stoch,
               distr_avg_stoch, hybrid1_avg_stoch, hybrid2_avg_stoch]

    avg_pot = [centr_avg_potent, distr_avg_potent,
               hybrid1_avg_potent, hybrid2_avg_potent]

    error_rew = [centr_se_stoch,
                 distr_se_stoch,  hybrid1_se_stoch, hybrid2_se_stoch]

    error_pot = [centr_se_stoch_potential, distr_se_stoch_potential,
                 hybrid1_se_stoch_potential, hybrid2_se_stoch_potential]

    rew_content = {
        "Tasks Visited": (avg_pot, error_pot),
        "Tasks Returned": (avg_rew, error_rew),
    }

    labels = ["Best", "Front-End Only",
              "Dist Replan Only", "Front End\n+ Dist Replan",
              "Front End\n+ Hybrid Replan"]

    # Plot results
    fig, ax = plt.subplots()

    x = np.arange(len(labels))
    width = 0.3
    multiplier = 0
    rects = ax.bar(
        x[0]+(width/2), best_avg, width, yerr=best_se,  label="Best")
    ax.bar_label(rects, padding=3)

    for attribute, measurements in rew_content.items():
        offset = width * multiplier
        x_temp = x[1:] + offset
        rects = ax.bar(
            x_temp, measurements[0], width, yerr=measurements[1],  label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # ax.bar(x, avg_tasks, yerr=error_bars, capsize=5,
    #        color=['blue', 'lightblue', 'red', 'green', 'darkviolet'])

    ax.set_xticks(x+width/2, labels)
    ax.set_ylabel('Percent Task Completion')
    title = "Task Completion for " + \
        str(problem_size) + " Tasks, " + str(num_robots) + \
        " Robots over " + str(trials) + " Trials"
    ax.set_title(title)
    ax.set_ybound(0.0, 1.0)
    ax.legend(loc='upper right', ncols=1)

    plt.show()