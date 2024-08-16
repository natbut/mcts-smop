import time
from argparse import ArgumentParser, Namespace

import yaml

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from control.task import add_tasks_to_dict, generate_tasks_from_config
from sim.comms_manager import CommsManager
from sim.environment import make_environment_from_config
from solvers.graphing import *
from solvers.masop_solver_config import (calculate_final_potential_reward,
                                         calculate_final_reward)
from utils.logger import FileLogger, init_logger
from utils.plotter import plot_results_from_log
from utils.visualizer import set_up_visualizer


def get_args() -> Namespace:
    """Parse the script arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "test_config",
        type=str,
        help="Full path to the test configuration file",
    )
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
        "topo_file",
        type=str,
        help="Full path to the topography data file",
    )
    parser.add_argument(
        "tide_folder",
        type=str,
        help="Full path to the tidal data folder",
    )

    return parser.parse_args()


def chance_task_add(prob_config_fp,
                    environment,
                    task_dict,
                    agent_list,
                    num_tasks=1,
                    new_task_prob=0.05):
    sample = np.random.random()
    if sample <= new_task_prob:
        add_tasks_to_dict(prob_config_fp,
                          environment,
                          task_dict,
                          num_tasks,
                          high_rew=True
                          )
        print("!!! NEW TASK")
        for a in agent_list:
            a.load_tasks_on_agent(task_dict)
            a.expected_event = False


if __name__ == "__main__":
    args = get_args()

    # Load testing params
    # TODO maybe load this more elegantly
    with open(args.test_config, "r") as f:
        test_config = yaml.safe_load(f)
        trials = test_config["trials"]
        sim_configs = test_config["sim_configs"]
        comms_max_range = test_config["comms_max_range"]
        comms_decay_range = test_config["comms_decay_range"]
        comms_decay_range_m = test_config["comms_decay_range_m"]
        comms_max_succ_prob = test_config["comms_max_succ_prob"]
        comms_decay_rate = test_config["comms_decay_rate"]
        edge_discovery_prob = test_config["edge_discovery_prob"]
        new_task_prob = test_config["new_task_prob"]
        num_new_tasks = test_config["num_new_tasks"]
        robot_fail_prob = test_config["robot_fail_prob"]
        percent_fail = test_config["percent_fail"]
        fail_timeStep = test_config["fail_timeStep"]

    print("Initialize Simulation")

    # Initialize results logger here
    logger = init_logger("Simulations")  # TODO replace print status updates
    file_logger = FileLogger()

    for tr in range(trials):
        print("\n==== Trial", tr, " ====")

        # Items to track
        best_results = []
        frontEnd_results = []
        distrOnly_results = []
        twoPart_results = []
        hybrid_results = []

        # === INITIALIZE SIMULATION ENVIRONMENT ===
        # Create environment
        env = make_environment_from_config(
            args.problem_config, args.topo_file, args.tide_folder)

        # Create task list from config
        task_dict = generate_tasks_from_config(args.problem_config, env)

        for sim_config in sim_configs:
            print("\n ===", " Tr", tr, ": Running", sim_config, "...")

            # Temp new tasks
            temp_task_dict = deepcopy(task_dict)

            # Create agents
            pssngr_list = generate_passengers_from_config(args.solver_config,
                                                          args.problem_config
                                                          )

            # Set up initial env state
            for p in pssngr_list:
                p.set_up_dim_ranges(env)
                p.sense_location_from_env(env)
                p.load_tasks_on_agent(temp_task_dict)

            # Create comms framework (TBD - may be ROS)
            comms_mgr = CommsManager(env,
                                     pssngr_list,
                                     comms_max_range,
                                     comms_decay_range,
                                     comms_decay_range_m,
                                     comms_max_succ_prob,
                                     comms_decay_rate,
                                     pssngr_list[0].sim_data["m_id"])

            # For other testing, plan over planning graph
            mothership = gen_mother_from_config(args.solver_config,
                                                args.problem_config,
                                                pssngr_list)
            comms_mgr.agent_dict[pssngr_list[0].sim_data["m_id"]
                                 ] = mothership
            mothership.update_reachable_neighbors(comms_mgr)
            mothership.set_up_dim_ranges(env)
            mothership.load_tasks_on_agent(temp_task_dict)

            # == If Centralized, Generate Plan & Load on Passengers
            if "Cent" in sim_config or "Hybrid" in sim_config:
                # Plan on mothership
                print("Solving team schedules...")
                mothership.solve_team_schedules(comms_mgr, pssngr_list)

            # Tracking failed robots for test cases
            fails = 0

            # Run Sim
            print("Executing schedules...")
            running = True
            i = 0
            b = 0
            viz = set_up_visualizer(env, temp_task_dict)
            # viz.display_env(pssngr_list, static=False)
            while running:
                print("\n== TIME STEP:", i)
                if "Hybrid" in sim_config:
                    mothership.update_reachable_neighbors(comms_mgr)

                for p in pssngr_list:
                    print("= Passenger", p.id, " Schedule:", p.schedule, " Rem Energy:",
                          p.sim_data["budget"])
                    # print("Schedule:", p.schedule,
                    #       " Act Dist:", p.my_action_dist)

                    # Update observations
                    p.sense_location_from_env(env)
                    p.sense_flow_from_env(env)
                    p.apply_observations_to_model()
                    p.update_reachable_neighbors(comms_mgr)

                    # if i > len(pssngr_list) and i % (50 + p.id) == 0:
                    #     p.expected_event = False
                    #     p.event = True

                    # Each time action is complete, do rescheduling (if distr)
                    if "Dist" in sim_config or "Hybrid" in sim_config:

                        # TODO At some point, remove need to pass sim_config into agent functions. Keep sim_config usage only at sim file level.

                        # Solve for new action distro
                        p.optimize_schedule_distr(comms_mgr, sim_config)

                    # Update current action, reduce energy
                    p.action_update(None, comms_mgr,
                                    pssngr_list, sim_config)

                    # Potential passenger failure
                    if "Best" not in sim_config:
                        # Check for random failures
                        p.random_failure(
                            comms_mgr, pssngr_list, sim_config, robot_fail_prob)
                        # Execute designed failures
                        if i == fail_timeStep:
                            if fails / len(pssngr_list) < percent_fail:
                                p.have_failure(
                                    comms_mgr, pssngr_list, sim_config)
                                fails += 1
                        # else:
                            # Continue broadcasting failure (up to b times)
                            # if b < 5:
                            #     p.failure_update(
                            #         comms_mgr, pssngr_list, sim_config)
                            #     b += 1

                            # Update comms graph
                    comms_mgr.update_connections()

                # Probability that new task is added to problem
                if "Hybrid" in sim_config:
                    all_agents = pssngr_list[:] + [mothership]
                else:
                    all_agents = pssngr_list
                chance_task_add(args.problem_config,
                                env,
                                temp_task_dict,
                                all_agents,
                                num_new_tasks,
                                new_task_prob)

                # Update environment
                env.step(pssngr_list)

                # Update message passing, also update comms graph
                comms_mgr.step()

                # Update visual
                viz.display_env(pssngr_list, static=False)
                time.sleep(0.01)

                # Check stopping criteria
                running = False
                for p in pssngr_list:
                    if not (p.dead or p.finished):
                        running = True

                i += 1

            print("Done")
            viz.close_viz()
            env.reset()

            # Store results
            reward = calculate_final_reward(temp_task_dict, pssngr_list)
            potent = calculate_final_potential_reward(
                temp_task_dict, pssngr_list)
            percentDead = sum(p.dead for p in pssngr_list) / len(pssngr_list)

            if sim_config == "Cent | Best":
                best_results.append(reward)
            elif sim_config == "Cent | Stoch":
                frontEnd_results.append(reward)
                frontEnd_results.append(potent)
                frontEnd_results.append(percentDead)
            elif sim_config == "Dist | Stoch":
                distrOnly_results.append(reward)
                distrOnly_results.append(potent)
                distrOnly_results.append(percentDead)
            elif sim_config == "Hybrid1 | Stoch":
                twoPart_results.append(reward)
                twoPart_results.append(potent)
                twoPart_results.append(percentDead)
            elif sim_config == "Hybrid2 | Stoch":
                hybrid_results.append(reward)
                hybrid_results.append(potent)
                hybrid_results.append(percentDead)

        file_logger(tr,
                    frontEnd_results,
                    distrOnly_results,
                    twoPart_results,
                    hybrid_results
                    )

    print("filename:", file_logger.log_filename)
    plot_results_from_log(file_logger.log_filename)

    # Run simulation (planning, etc.)
    # agent_list[0].send_message(comms_mgr, agent_list[1].id, "Hello!")
    # i = 0
    # viz = set_up_visualizer_from_config(env, args.config)
    # while True:
    #     print("TIME STEP:", i)

    #     # Update agents (scheduling & control happens here)
    #     for a in agent_list:  # TODO consider a.step() for all this
    #         # Consider condensing this all into an a.step()
    #         # get updated observations
    #         a.sense_location_from_env(env)
    #         a.sense_flow_from_env(env)

    #         # If new flow found (EVENT), do rescheduling
    #         # TODO update to only reschedule after action is completed
    #         if a.event:
    #             a.apply_observations_to_model()
    #             # a.optimize_schedule() # TODO

    #             # TODO Send necessary comms

    #         # Update current action
    #         a.action_update()

    #     # Apply actions to get new environment state (update agent positions & observations, comms update)
    #     env.step(agent_list)

    #     # Update message passing, also update comms graph
    #     comms_mgr.step()

    #     # Update visual
    #     viz.display_env(agent_list, static=False)

    #     i += 1
    #     time.sleep(0.1)
