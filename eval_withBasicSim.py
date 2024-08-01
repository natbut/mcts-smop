from argparse import ArgumentParser, Namespace
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import yaml

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from sim.comms_manager import CommsManager_Basic
from solvers.graphing import *
from solvers.masop_solver_config import (calculate_final_potential_reward,
                                         calculate_final_reward)
from utils.logger import FileLogger, init_logger


def get_args() -> Namespace:
    """Parse the script arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "problem_config",
        type=str,
        help="Full path to the problem configuration file",
    )
    parser.add_argument(
        "solver_config",
        type=str,
        help="Full path to the solvers configuration file",
    )
    parser.add_argument(
        "test_config",
        type=str,
        help="Full path to the test configuration file",
    )

    return parser.parse_args()


# Define framework for doing DecMCTS with multiple agents on generic graph
if __name__ == "__main__":
    args = get_args()

    # Load testing params
    with open(args.test_config, "r") as f:
        test_config = yaml.safe_load(f)
        trials = test_config["trials"]
        sim_configs = test_config["sim_configs"]
        comms_succ_prob = test_config["comms_succ_prob"]
        edge_discovery_prob = test_config["edge_discovery_prob"]

    print("Initialize Simulation")

    # Initialize results logger here
    logger = init_logger("Simulations")  # TODO print status updates
    file_logger = FileLogger("LOGGING_TEST")

    # === TRACKING FOR EVALUATION ===
    # TODO move to logging
    best_reward = []
    best_percentDead = []
    centr_stoch_reward = []
    centr_stoch_potential = []
    centr_stoch_percentDead = []
    distr_stoch_reward = []
    distr_stoch_potential = []
    distr_stoch_percentDead = []
    hybrid1_stoch_reward = []
    hybrid1_stoch_potential = []
    hybrid1_stoch_percentDead = []
    hybrid2_stoch_reward = []
    hybrid2_stoch_potential = []
    hybrid2_stoch_percentDead = []

    for tr in range(trials):
        print("\n==== Trial", tr, " ====")
        # === INITIALIZE SIMULATION ENVIRONMENT ===
        # Create problem instance
        planning_graph = create_sop_inst_from_config(args.problem_config)

        # Create planning graph from true graph
        true_graph = create_true_graph(planning_graph)

        # Data for best solution with Sim-BRVNS on true graph

        # == Bundle Data
        # best_sim_data = {"graph": deepcopy(true_graph),
        #                  "start": start,
        #                  "end": end,
        #                  "budget": budget,
        #                  "velocity": velocity,
        #                  "basic": True,
        #                  "m_id": m_id
        #                  }

        # sim_data = {"graph": deepcopy(planning_graph),
        #             "start": start,
        #             "end": end,
        #             "budget": budget,
        #             "velocity": velocity,
        #             "basic": True,
        #             "m_id": m_id
        #             }
        # dec_mcts_data = {"num_robots": num_robots,
        #                  "fail_prob": failure_probability,
        #                  "comm_n": comm_n,
        #                  "plan_iters": planning_iters
        #                  }
        # sim_brvns_data = {"num_robots": num_robots,
        #                   "alpha": alpha,
        #                   "beta": beta,
        #                   "k_initial": k_initial,
        #                   "k_max": k_max,
        #                   "t_max": t_max,
        #                   "explore_iters": exploratory_mcs_iters,
        #                   "intense_iters": intensive_mcs_iters
        #                   }

        for sim_config in sim_configs:
            print("\n ===", " Tr", tr, ": Running", sim_config, "...")
            # Run simulation (planning, etc.)

            # Create agents
            pssngr_list = generate_passengers_from_config(args.solver_config,
                                                          args.problem_config,
                                                          deepcopy(planning_graph))
            # pssngr_list = generate_passengers_with_data(
            #     dec_mcts_data, sim_data)

            # Create comms framework (TBD - may be ROS)
            # TODO set up comms failures
            comms_mgr = CommsManager_Basic(pssngr_list, comms_succ_prob)

            # == If Centralized, Generate Plan & Load on Passengers
            if "Cent" in sim_config or "Hybrid" in sim_config:
                # Plan on mothership
                print("Solving schedules...")
                comms_mgr.success_prob = 1.0
                if "Best" in sim_config:
                    # For baseline, plan over true graph
                    best_mother = gen_mother_from_config(args.solver_config,
                                                         args.problem_config,
                                                         deepcopy(true_graph))
                    comms_mgr.agent_dict[-1] = best_mother
                    best_mother.solve_team_schedules(comms_mgr, pssngr_list)
                else:
                    # For other testing, plan over planning graph
                    mothership = gen_mother_from_config(args.solver_config,
                                                        args.problem_config,
                                                        deepcopy(planning_graph))
                    comms_mgr.agent_dict[-1] = mothership
                    mothership.solve_team_schedules(comms_mgr, pssngr_list)
                comms_mgr.success_prob = comms_succ_prob

            # Run Sim
            print("Executing schedules...")
            running = True
            i = 0
            while running:
                print("== TIME STEP:", i)
                for p in pssngr_list:
                    print("= Passenger", p.id)
                    # Each time action is complete, do rescheduling (if distr)
                    if p.event and ("Dist" in sim_config or "Hybrid" in sim_config) and not p.finished:
                        if "Hybrid2" in sim_config:
                            # Solve for centralized action update
                            print("Solving hybrid schedule...")
                            mothership.solve_new_single_schedule(comms_mgr,
                                                                 p,
                                                                 pssngr_list,
                                                                 act_samples=1,
                                                                 t_limit=30
                                                                 )
                        # Solve for new action distro
                        # If hybrid, will consider mothership-provided solutions
                        p.optimize_schedule_distr(
                            comms_mgr, pssngr_list, sim_config)

                    # Update current action
                    p.action_update(true_graph, sim_config)
                    p.reduce_energy_basic()  # only if not idle
                    # only if not idle
                    p.edge_discover(true_graph, comms_mgr, pssngr_list,
                                    sim_config, edge_discovery_prob)

                # Update message passing, also update comms graph
                comms_mgr.step()

                # Check stopping criteria
                running = False
                for p in pssngr_list:
                    if not (p.dead or p.finished):
                        running = True

                i += 1

            print("Done")

            # Store results
            # TODO logging
            reward = calculate_final_reward(true_graph, pssngr_list)
            potent = calculate_final_potential_reward(true_graph, pssngr_list)
            percentDead = sum(p.dead for p in pssngr_list) / len(pssngr_list)

            if sim_config == "Cent | Best":
                best_reward.append(reward)
                best_percentDead.append(percentDead)
            elif sim_config == "Cent | Stoch":
                centr_stoch_reward.append(reward)
                centr_stoch_potential.append(potent)
                centr_stoch_percentDead.append(percentDead)
            elif sim_config == "Dist | Stoch":
                distr_stoch_reward.append(reward)
                distr_stoch_potential.append(potent)
                distr_stoch_percentDead.append(percentDead)
            elif sim_config == "Hybrid1 | Stoch":
                hybrid1_stoch_reward.append(reward)
                hybrid1_stoch_potential.append(potent)
                hybrid1_stoch_percentDead.append(percentDead)
            elif sim_config == "Hybrid2 | Stoch":
                hybrid2_stoch_reward.append(reward)
                hybrid2_stoch_potential.append(potent)
                hybrid2_stoch_percentDead.append(percentDead)

    file_logger(trials_arr,
                best_results,
                frontEnd_results,
                distrOnly_results,
                twoPart_results,
                hybrid_results
                )

    print(  # "DEAD Distr DET:", np.mean(distr_det_percentDead),
        "\nDEAD Centr BEST:", np.mean(best_percentDead),
        "\nDEAD Centr STOCH:", np.mean(centr_stoch_percentDead),
        "\nDEAD Distr STOCH:", np.mean(distr_stoch_percentDead),
        #   "\nDEAD Hybrid DET:", np.mean(hybrid_det_percentDead),
        "\nDEAD Hybrid1-DecMCTS STOCH:", np.mean(hybrid1_stoch_percentDead),
        "\nDEAD Hybrid2-Hybrid STOCH:", np.mean(hybrid2_stoch_percentDead))

    # Mean
    # distr_avg_det = np.mean(distr_det_results)
    best_avg = round(np.mean(best_reward), 2)
    centr_avg_stoch = round(np.mean(centr_stoch_reward), 2)
    centr_avg_potent = round(np.mean(centr_stoch_potential), 2)
    distr_avg_stoch = round(np.mean(distr_stoch_reward), 2)
    distr_avg_potent = round(np.mean(distr_stoch_potential), 2)
    # hybrid_avg_det = np.mean(hybrid_det_results)
    hybrid1_avg_stoch = round(np.mean(hybrid1_stoch_reward), 2)
    hybrid1_avg_potent = round(np.mean(hybrid1_stoch_potential), 2)
    hybrid2_avg_stoch = round(np.mean(hybrid2_stoch_reward), 2)
    hybrid2_avg_potent = round(np.mean(hybrid2_stoch_potential), 2)
    # StdErr
    # distr_se_det = np.std(distr_det_results) / np.sqrt(len(distr_det_results))

    best_se = np.std(best_reward) / np.sqrt(len(best_reward))
    centr_se_stoch = np.std(centr_stoch_reward) / \
        np.sqrt(len(centr_stoch_reward))
    centr_se_stoch_potential = np.std(centr_stoch_potential) / \
        np.sqrt(len(centr_stoch_potential))

    distr_se_stoch = np.std(distr_stoch_reward) / \
        np.sqrt(len(distr_stoch_reward))
    distr_se_stoch_potential = np.std(distr_stoch_potential) / \
        np.sqrt(len(distr_stoch_potential))

    hybrid1_se_stoch = np.std(hybrid1_stoch_reward) / \
        np.sqrt(len(hybrid1_stoch_reward))
    hybrid1_se_stoch_potential = np.std(hybrid1_stoch_potential) / \
        np.sqrt(len(hybrid1_stoch_potential))
    hybrid2_se_stoch = np.std(hybrid2_stoch_reward) / \
        np.sqrt(len(hybrid2_stoch_reward))
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
              "Distr. Only", "Front End\n+ Dist Replan",
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
    ax.legend(loc='lower right', ncols=1)

    plt.show()
