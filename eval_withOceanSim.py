import time
from argparse import ArgumentParser, Namespace

import yaml

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from control.support import generate_supports_from_config
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


def chance_robot_fail(agent_list,
                      robot_fail_prob,
                      comms_mgr,
                      timeStep):
    # Do fail updates
    if timeStep % 10 == 0:
        for a in agent_list:
            # Potential passenger failure
            if a.dead and a.type != a.MOTHERSHIP:
                # Continue attempting to send failure update (occasionally)
                a.have_failure(comms_mgr)

    # Do random failure
    sample = np.random.random()
    # print("Sampled fail prob:", sample)
    if sample <= robot_fail_prob:
        rob = np.random.choice(agent_list)
        print("!!! ROBOT FAILURE")
        if rob.type != rob.MOTHERSHIP:
            fail_samp = np.random.random()
            if fail_samp <= 0.5:
                # 50% chance total failure
                rob.have_failure(comms_mgr)
            else:
                # 50% chance reduced battery
                rob.sim_data["budget"] = rob.sim_data["budget"] // 2


def chance_task_add(prob_config_fp,
                    environment,
                    task_dict,
                    mothership,
                    group_list,
                    num_tasks_to_add=1,
                    new_task_prob=0.05):

    sample = np.random.random()
    # print("Sampled task prob:", sample)
    if sample <= new_task_prob:
        add_tasks_to_dict(prob_config_fp,
                          environment,
                          task_dict,
                          num_tasks_to_add,
                          high_rew=True
                          )
        print("!!! NEW TASK")
        new_task = list(task_dict.items())[-1][1]
        mothership.added_tasks.append(new_task)
        mothership.load_task(new_task.id,
                             new_task.location,
                             new_task.work,
                             new_task.reward)
        # Distribute new tasks
        for p in group_list:
            content = (mothership.id, p.id, "New Task",
                       mothership.added_tasks)
            mothership.send_msg_down_chain(comms_mgr, content)


if __name__ == "__main__":
    args = get_args()

    # Load testing params
    with open(args.test_config, "r") as f:
        test_config = yaml.safe_load(f)
        test_name = test_config["test_name"]
        trials = test_config["trials"]
        tests = test_config["tests"]
        show_viz = test_config["viz"]
        sim_configs = test_config["sim_configs"]
        comms_max_range = test_config["comms_max_range"]
        comms_decay_range = test_config["comms_decay_range"]
        comms_max_succ_prob = test_config["comms_max_succ_prob"]
        comms_decay_rate = test_config["comms_decay_rate"]
        edge_discovery_prob = test_config["edge_discovery_prob"]
        new_task_prob = test_config["new_task_prob"]
        num_new_tasks = test_config["num_new_tasks"]
        robot_fail_prob = test_config["robot_fail_prob"]
        percent_fail = test_config["percent_fail"]
        fail_timeStep = test_config["fail_timeStep"]

        replan_freq = test_config["replan_freq"]

    print("Initialize Simulation")

    # Initialize results logger here
    logger = init_logger("Simulations")  # TODO replace print status updates
    file_logger = FileLogger(filename=test_name)

    for tr in range(trials):
        print("\n==== Trial", tr, " ====")

        # === INITIALIZE SIMULATION ENVIRONMENT ===
        # Create environment
        env = make_environment_from_config(
            args.problem_config, args.topo_file, args.tide_folder)

        random_base = env.setup_random_base_loc()
        print("Random base location:", random_base)

        # Create task list from config
        task_dict = generate_tasks_from_config(
            args.problem_config, env, random_base)

        for ts in range(tests):

            # Items to track
            tr_arr = []
            ts_arr = []
            best_results = []
            frontEnd_results = []
            distrOnly_results = []
            twoPart_results = []
            hybrid_results = []

            for sim_config in sim_configs:
                print("\n ===", " Tr", tr, ": Running", sim_config, "...")

                # Temp new tasks
                temp_task_dict = deepcopy(task_dict)

                # Create agents
                group_list = generate_passengers_from_config(
                    args.solver_config, args.problem_config, random_base)

                supp_list = generate_supports_from_config(args.solver_config,
                                                          args.problem_config,
                                                          random_base)

                mothership = gen_mother_from_config(args.solver_config,
                                                    args.problem_config,
                                                    group_list,
                                                    random_base)

                # Create comms framework (TBD - may be ROS)
                all_agents = group_list + supp_list + [mothership]
                comms_mgr = CommsManager(env,
                                         all_agents,
                                         comms_max_range,
                                         comms_decay_range,
                                         comms_max_succ_prob,
                                         comms_decay_rate,
                                         group_list[0].sim_data["m_id"])

                # Set up initial env state
                mothership.agent_list = all_agents
                mothership.group_list = group_list
                mothership.update_reachable_neighbors(comms_mgr)
                mothership.set_up_dim_ranges(env)
                mothership.load_tasks_on_agent(temp_task_dict)
                for p in group_list:
                    p.agent_list = all_agents
                    p.group_list = group_list
                    p.mothership = mothership
                    p.update_reachable_neighbors(comms_mgr)
                    p.set_up_dim_ranges(env)
                    p.sense_location_from_env(env)
                    p.load_tasks_on_agent(temp_task_dict)
                for s in supp_list:
                    s.agent_list = all_agents
                    s.group_list = group_list
                    s.mothership = mothership
                    s.update_reachable_neighbors(comms_mgr)
                    s.set_up_dim_ranges(env)
                    s.sense_location_from_env(env)

                # == If Centralized, Generate Plan & Load on Passengers
                if "Cent" in sim_config or "Hybrid" in sim_config:
                    # Plan on mothership
                    print("Solving team schedules...")
                    mothership.solve_team_schedules(comms_mgr)

                # Tracking failed robots for test cases
                fails = 0

                # Run Sim
                print("Executing schedules...")
                running = True
                i = 0
                b = 0
                title = "Tr " + str(tr) + ": " + sim_config
                if show_viz:
                    viz = set_up_visualizer(env, temp_task_dict, title)
                while running:
                    print("\n== TIME STEP:", i)
                    for p in all_agents:
                        # Update observations
                        p.sense_location_from_env(env)
                        p.sense_flow_from_env(env)
                        p.apply_observations_to_model()
                        p.update_reachable_neighbors(comms_mgr)

                        if p.type == p.MOTHERSHIP:
                            print("M completion record:", [
                                t for t in p.task_dict.keys() if p.task_dict[t].complete])

                        if p.type == p.PASSENGER:
                            print("= Passenger", p.id, " Tour:", [p.action[1]] + p.schedule, " Rem Energy:",
                                  p.sim_data["budget"], " | Finished:", p.finished)
                            print(p.id, "completion record:", [
                                t for t in p.task_dict.keys() if p.task_dict[t].complete])

                            # Each time action is complete, do rescheduling (if distr)
                            if "Dist" in sim_config or "Hybrid" in sim_config:
                                # Forces periodic rescheduling
                                if i > len(group_list) and i % (replan_freq + (2*p.id)) == 0:
                                    p.action[0] = p.IDLE
                                    p.event = True
                                # Solve for new action distro
                                p.optimize_schedule_distr(
                                    comms_mgr, sim_config)
                            # Update current action, reduce energy
                            p.action_update(comms_mgr)
                            # p.share_location(comms_mgr)

                        if p.type == p.SUPPORT:
                            p.action_update()

                    # EVENTS
                    chance_robot_fail(group_list,
                                      robot_fail_prob,
                                      comms_mgr,
                                      i)
                    chance_task_add(args.problem_config,
                                    env,
                                    temp_task_dict,
                                    mothership,
                                    group_list,
                                    num_new_tasks,
                                    new_task_prob)

                    # Update environment
                    env.step(all_agents)

                    # Update message passing, also update comms graph
                    comms_mgr.step()

                    # Update visual
                    if show_viz:
                        viz.display_env(group_list, supp_list, static=False)
                        # time.sleep(0.01)

                    # Check stopping criteria
                    running = False
                    for p in group_list:
                        if not (p.dead or p.finished):
                            running = True

                    i += 1

                print("Done")
                if show_viz:
                    viz.close_viz()
                env.reset()

                # Store results
                reward = calculate_final_reward(temp_task_dict, group_list)
                potent = calculate_final_potential_reward(
                    temp_task_dict, group_list)
                percentDead = sum(p.dead for p in group_list) / len(group_list)

                print("Final Potential:", potent, " | Actual:", reward)

                if sim_config == "Cent | Stoch":
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

            tr_arr.append(tr)
            ts_arr.append(ts)

            if len(frontEnd_results) == 0:
                frontEnd_results.append(0)
                frontEnd_results.append(0)
                frontEnd_results.append(0)
            if len(distrOnly_results) == 0:
                distrOnly_results.append(0)
                distrOnly_results.append(0)
                distrOnly_results.append(0)
            if len(twoPart_results) == 0:
                twoPart_results.append(0)
                twoPart_results.append(0)
                twoPart_results.append(0)
            if len(hybrid_results) == 0:
                hybrid_results.append(0)
                hybrid_results.append(0)
                hybrid_results.append(0)

            file_logger(tr_arr,
                        ts_arr,
                        frontEnd_results,
                        distrOnly_results,
                        twoPart_results,
                        hybrid_results
                        )

    plot_results_from_log(file_logger.log_filename)
