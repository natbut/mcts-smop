from argparse import ArgumentParser, Namespace
from copy import deepcopy

import yaml

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from sim.comms_manager import CommsManager_Basic
from solvers.graphing import *
from solvers.masop_solver_config import (calculate_final_potential_reward,
                                         calculate_final_reward)
from utils.logger import FileLogger, init_logger
from utils.plotter import plot_results_from_log


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


def chance_task_add(agent_list,
                    problem_config,
                    true_graph,
                    new_task_prob,
                    true_c=0.05):

    sample = np.random.random()
    if sample <= new_task_prob:
        with open(problem_config, "r") as f:
            config = yaml.safe_load(f)

            new_vert = "v" + str(len(true_graph.vertices))
            print("! Adding edge", new_vert)

            # Add HIGH-REWARD Tasks only
            new_rew = config["reward_range"][1]+1

            if config["work_range"][0] == config["work_range"][1]:
                new_work = config["work_range"][0]
            else:
                new_work = np.random.randint(
                    config["work_range"][0],
                    config["work_range"][1])

            new_edges = []
            new_dists = {}
            for v in true_graph.vertices:
                new_edges.append((v, new_vert))
                mean = np.random.uniform(
                    config["edges_mean_range"][0],
                    config["edges_mean_range"][1])
                stddev = (config["c"] * mean)**0.5
                new_dists[(v, new_vert)] = norm(loc=mean, scale=stddev)

                new_edges.append((new_vert, v))
                mean = np.random.uniform(
                    config["edges_mean_range"][0],
                    config["edges_mean_range"][1])
                stddev = (config["c"] * mean)**0.5
                new_dists[(new_vert, v)] = norm(loc=mean, scale=stddev)

            true_graph.vertices.append(new_vert)
            true_graph.rewards[new_vert] = new_rew
            true_graph.works[new_vert] = new_work
            true_graph.edges += new_edges[:]
            true_graph.cost_distributions.update(new_dists)
            for edge in new_edges:
                cost_sample = true_graph.sample_edge_stoch(edge)
                stddev = (true_c * cost_sample)**0.5
                true_graph.cost_distributions[edge] = norm(
                    loc=cost_sample, scale=stddev)

            for p in agent_list:
                p.sim_data["graph"].vertices.append(new_vert)
                p.sim_data["graph"].rewards[new_vert] = new_rew
                p.sim_data["graph"].works[new_vert] = new_work
                p.load_task(new_vert, None, new_work)
                for edge in new_edges:
                    p.sim_data["graph"].edges.append(edge)
                    p.sim_data["graph"].cost_distributions[edge] = new_dists[edge]


# Define framework for doing DecMCTS with multiple agents on generic graph
if __name__ == "__main__":
    args = get_args()

    # Load testing params
    with open(args.test_config, "r") as f:
        test_config = yaml.safe_load(f)
        trials = test_config["trials"]
        sim_configs = test_config["sim_configs"]
        comms_succ_prob_m = test_config["comms_succ_prob_m"]
        comms_succ_prob_p = test_config["comms_succ_prob_p"]
        edge_discovery_prob = test_config["edge_discovery_prob"]
        new_task_prob = test_config["new_task_prob"]
        robot_fail_prob = test_config["robot_fail_prob"]
        percent_fail = test_config["percent_fail"]
        fail_timeStep = test_config["fail_timeStep"]
        rel_mod = test_config["rel_mod"]

    print("Initialize Simulation")

    # Initialize results logger here
    logger = init_logger("Simulations")  # TODO print status updates
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
        # Create problem instance
        planning_graph = create_sop_inst_from_config(args.problem_config)

        # Create planning graph from true graph
        true_graph = create_true_graph(planning_graph)

        for sim_config in sim_configs:
            print("\n ===", " Tr", tr, ": Running", sim_config, "...")

            trial_graph = deepcopy(true_graph)

            # Create agents
            pssngr_list = generate_passengers_from_config(args.solver_config,
                                                          args.problem_config,
                                                          deepcopy(planning_graph))

            # Create comms framework (TBD - may be ROS)
            comms_mgr = CommsManager_Basic(pssngr_list,
                                           comms_succ_prob_m, comms_succ_prob_p,
                                           pssngr_list[0].sim_data["m_id"]
                                           )

            fails = 0

            # == If Centralized, Generate Plan & Load on Passengers
            if "Cent" in sim_config or "Hybrid" in sim_config:
                # Plan on mothership
                print("Solving schedules...")
                comms_mgr.success_prob_m = 1.0
                if "Best" in sim_config:
                    # For baseline, plan over true graph
                    best_mother = gen_mother_from_config(args.solver_config,
                                                         args.problem_config,
                                                         deepcopy(trial_graph),
                                                         pssngr_list)
                    comms_mgr.agent_dict[-1] = best_mother
                    best_mother.solve_team_schedules(comms_mgr, pssngr_list)
                else:
                    # For other testing, plan over planning graph
                    mothership = gen_mother_from_config(args.solver_config,
                                                        args.problem_config,
                                                        deepcopy(
                                                            planning_graph),
                                                        pssngr_list)
                    comms_mgr.agent_dict[-1] = mothership
                    mothership.solve_team_schedules(comms_mgr, pssngr_list)
                comms_mgr.success_prob_m = comms_succ_prob_m

            # Run Sim
            print("Executing schedules...")
            running = True
            i = 0
            while running:
                print("\n== TIME STEP:", i)
                for p in pssngr_list:
                    print("= Passenger", p.id)
                    # Each time action is complete, do rescheduling (if distr)
                    if "Dist" in sim_config or "Hybrid" in sim_config:
                        if "Hybrid2" in sim_config:
                            # Solve for centralized action update
                            mothership.solve_new_single_schedule(comms_mgr,
                                                                 p,
                                                                 pssngr_list,
                                                                 act_samples=1,
                                                                 t_limit=30
                                                                 )  # TODO address params
                        # Solve for new action distro
                        # If hybrid, will consider mothership-provided solutions
                        p.optimize_schedule_distr(
                            comms_mgr, pssngr_list, sim_config, rel_mod)

                    # Update current action
                    p.action_update(trial_graph, comms_mgr,
                                    pssngr_list, sim_config)
                    p.reduce_energy_basic()  # only if not idle

                    # Discover edges
                    p.edge_discover(trial_graph, comms_mgr, pssngr_list,
                                    sim_config, edge_discovery_prob)

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

                # Probability that new task is added to problem
                if "Hybrid2" in sim_config:
                    all_agents = pssngr_list[:] + [mothership]
                else:
                    all_agents = pssngr_list
                chance_task_add(all_agents,
                                args.problem_config, trial_graph, new_task_prob)

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
            reward = calculate_final_reward(trial_graph, pssngr_list)
            potent = calculate_final_potential_reward(trial_graph, pssngr_list)
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
                    best_results,
                    frontEnd_results,
                    distrOnly_results,
                    twoPart_results,
                    hybrid_results
                    )

    print("filename:", file_logger.log_filename)
    plot_results_from_log(file_logger.log_filename)
