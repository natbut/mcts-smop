import math
import random
import time as tm
from copy import deepcopy

import numpy as np
from scipy.stats import geom

from solvers.graphing import Graph, create_sop_instance
from solvers.masop_solver_config import *

# === FUNCTIONS FOR BR CONSTRUCTIVE HEURISTIC ===


def generateDummySolution(start, end, locations):
    # Generate an initial dummy solution where each location is connected to the origin and destination
    dummy_solution = [[(start, loc), (loc, end)]
                      for loc in locations if loc != start and loc != end]
    return dummy_solution


def computeSortedSavingsList(graph: Graph, alpha=0.5):
    savings_list = []
    locations = graph.vertices
    for i in locations:
        for j in locations:
            if i != j and i != "vs" and i != "vg" and j != "vs" and j != "vg":
                t_i_vg = graph.get_mean_cost_edgeWork((i, "vg"))
                t_vs_j = graph.get_mean_cost_edgeWork(("vs", j))
                t_i_j = graph.get_mean_cost_edgeWork((i, j))
                sij = t_i_vg + t_vs_j - t_i_j
                ui = graph.rewards[i]
                uj = graph.rewards[j]
                eij = alpha * sij + (1 - alpha) * (ui + uj)
                savings_list.append((eij, (i, j)))
    # Sort by efficiency in descending order
    savings_list.sort(reverse=True, key=lambda x: x[0])
    return savings_list


def getStartingRoute(arc, solution):
    for route in solution:
        if arc[0] == route[-2][1]:  # check if route ends where arc starts
            return route
    return None


def getClosingRoute(arc, solution):
    for route in solution:
        if arc[1] == route[1][0]:  # check if route starts where arc ends
            return route
    return None


def mergeRoutes(iRoute, jRoute, arc):
    if iRoute and jRoute:
        # Merging iRoute and jRoute
        return iRoute[:-1] + [arc] + jRoute[1:]  # Merging at common point
    return None


def calcRouteTravelTime(route, graph: Graph):
    if route == None:
        return None
    travel_time = 0
    for edge in route:
        travel_time += graph.get_mean_cost_edgeWork(edge)
    return travel_time


def validateMergeDrivingConstraints(travelTimeNewRoute, budget):
    if travelTimeNewRoute == None:
        return False
    return travelTimeNewRoute <= budget


def updateSolution(newRoute, iRoute, jRoute, solution):
    solution.remove(iRoute)
    try:  # TODO address this failure
        solution.remove(jRoute)
    except:
        print("Unable to remove start", iRoute,
              " and end", jRoute, " from", solution)
    solution.append(newRoute)
    return solution


def sortRoutesByProfit(solution, rewards):
    solution.sort(key=lambda route: sum(rewards.get(
        edge[1], 0) for edge in route), reverse=True)


def deleteRoutesByProfit(solution, num_robots):
    while len(solution) > num_robots:
        solution.pop()


def geometric_distribution_selection(list, beta):
    """
    Select an arc from the savings_list based on a geometric probability distribution.

    Parameters:
    savings_list (list): List of arcs sorted by their savings scores in descending order.
    beta (float): The parameter for the geometric distribution (0 < beta < 1).

    Returns:
    tuple: The selected arc.
    """
    # Calculate the cumulative probabilities using the geometric distribution
    probabilities = [(1 - beta)**i * beta for i in range(len(list))]
    cumulative_probabilities = np.cumsum(probabilities)

    # Normalize cumulative probabilities to sum to 1
    cumulative_probabilities /= cumulative_probabilities[-1]

    # Randomly select an arc based on the cumulative probabilities
    random_value = np.random.rand()
    selected_index = np.searchsorted(cumulative_probabilities, random_value)

    return selected_index


def constructive_heuristic(graph: Graph,
                           budget,
                           num_routes,
                           start,
                           end,
                           alpha=0.5,
                           beta=0.3):

    # print("=== CONSTR HEURISTIC ===")
    locations = graph.vertices
    rewards = graph.rewards
    # Step 1: Generate initial dummy solution
    solution = generateDummySolution(start, end, locations)
    # print("Dummy solution:", solution)

    # Step 2: Compute sorted savings list
    savings_list = computeSortedSavingsList(graph, alpha)
    # print("Savings list:", savings_list)

    # Step 3-14: Merge routes based on savings list
    while savings_list and len(solution) > 1:
        idx = geometric_distribution_selection(savings_list, beta)
        arc = savings_list.pop(idx)[1]
        # savings_list.pop(0)[1]
        # print("=== Evaluating arc:", arc)
        iRoute = getStartingRoute(arc, solution)
        # print("Route start:", iRoute)
        jRoute = getClosingRoute(arc, solution)
        # print("Route end:", jRoute)
        if iRoute == jRoute:
            continue
        newRoute = mergeRoutes(iRoute, jRoute, arc)
        # print("New route:", newRoute)
        travelTimeNewRoute = calcRouteTravelTime(newRoute, graph)
        # print("Travel time:", travelTimeNewRoute)
        isMergeValid = validateMergeDrivingConstraints(
            travelTimeNewRoute, budget)  # NOTE: May need to constrain valid to limit inclusion of nodes (s.t. each visited once max)
        if isMergeValid:
            solution = updateSolution(newRoute, iRoute, jRoute, solution)
            # print("Valid route")
        # print("Solution progress:", solution)

    # Step 15: Sort routes by profit
    sortRoutesByProfit(solution, rewards)
    # print("Routes sorted by profit:", solution)

    # Step 16: Delete routes by profit
    deleteRoutesByProfit(solution, num_routes)
    # print("Reduced list:", solution)

    # Step 17: Return the final solution
    for i, route in enumerate(solution):
        node_route = [edge[0] for edge in route] + [end]
        solution[i] = node_route
    return solution


# === FUNCTIONS FOR VNS ===


def BRVNS(initial_solution,
          graph: Graph,
          budget: int,
          start,
          end,
          alpha=0.5,
          beta=0.3,
          k_initial=1,
          k_max=100,
          t_max=1000,
          exploratory_mcs_iters=200,
          intensive_mcs_iters=1000
          ):

    # print("Initial Solution:", initial_solution)
    baseSol = initial_solution
    bestSol = initial_solution
    det_reward_base = routes_det_reward(baseSol, graph, budget)
    stoch_reward, reliability = fast_simulation(baseSol,
                                                graph,
                                                budget,
                                                exploratory_mcs_iters
                                                )
    stoch_reward_base = stoch_reward
    stoch_reward_best = stoch_reward

    elite_solutions = [initial_solution]
    k = k_initial
    T = 1000
    lamb = 0.99  # 0.999
    start_time = tm.time()
    elapsed_time = 0
    # Phase 1 processing
    while elapsed_time <= t_max:  # TODO introduce timeout
        k = k_initial  # degree of shaking destruction (1%-100%)
        while k <= k_max:
            # print("== TIME STEP", time, " ==")
            newSol = shake(baseSol, k, k_max, graph,
                           budget, start, end, alpha, beta)
            # Apply 2-opt procedure to each route until no further improvement
            newSol = two_opt(newSol, graph)
            # Remove a subset of nodes from each route
            newSol = remove_subset(newSol, graph)
            # Re-insert nodes with biased insertion procedure
            newSol = biased_insertion(newSol, graph, budget, beta2=0.3)
            # print("Checking sol.:", newSol, " | Current k:", k)

            det_reward_new = routes_det_reward(newSol, graph, budget)
            # print("New det:", det_reward_new, " Base det:", det_reward_base)
            if det_reward_new > det_reward_base:
                # print("Better than base det, check stoch")
                stoch_reward, _ = fast_simulation(newSol,
                                                  graph,
                                                  budget,
                                                  intensive_mcs_iters
                                                  )
                if stoch_reward > stoch_reward_base:
                    # print("Better than base stoch; update base")
                    baseSol = newSol
                    stoch_reward_base = stoch_reward
                    if stoch_reward > stoch_reward_best:
                        # print("Better than best stoch; update best")
                        bestSol = newSol
                        stoch_reward_best = stoch_reward
                        elite_solutions.append(newSol)
                    k = k_initial
            else:
                update_prob = prob_of_updating(det_reward_new,
                                               det_reward_base,
                                               T)
                # print("Update prob:", update_prob, " | T:", T)
                if update_prob > np.random.random():  # TODO Does this work right?
                    # print("Update to excape minima")
                    baseSol = newSol
                    det_reward_base = det_reward_new
                    k = k_initial
                else:
                    k += 1
            # Update prob parameter (cooling)
            T = T*lamb

            elapsed_time = tm.time() - start_time
            if elapsed_time > t_max:
                break

    # Phase 2 processing
    best_reliability = 0
    for sol in elite_solutions:
        stoch_reward_sol, reliability_sol = intensive_simulation(sol,
                                                                 graph, budget, intensive_mcs_iters)
        if stoch_reward_sol > stoch_reward_best:
            bestSol = sol
            best_reliability = reliability_sol
            stoch_reward_best = stoch_reward_sol

    # NOTE create a set of states for routes in bestSol
    sol_as_states = []
    for route in bestSol:
        rem_budget = budget - route_det_cost(route, graph)
        sol_as_states.append(State(route, rem_budget))

    return sol_as_states, stoch_reward_best, best_reliability


def shake(solution, k, k_max, graph: Graph, budget, start, end, alpha, beta):
    # Destruction-reconstruction procedure
    num_to_remove = math.ceil(len(solution) * (k/k_max))
    # print("Removing", num_to_remove, " routes from", solution)
    new_solution = solution[:]
    # Randomly delete k% of routes from solution
    for _ in range(num_to_remove):
        new_solution.pop(random.randint(0, len(new_solution) - 1))
    # Biased-randomized constructive heuristic to generate new routes
    # print("Shake1 New Sol:", new_solution)
    shake_graph = deepcopy(graph)
    # Remove nodes/edges currently in new_solution from shake graph
    for route in new_solution:
        for i, vert in enumerate(route):
            if vert != route[0] and vert != route[-1]:  # TODO do this right
                shake_graph.vertices.remove(vert)
            # elif i != len(route)-1:
            #     shake_graph.edges.remove((vert, route[i+1]))
    # Merge solutions
    new_solution = constructive_heuristic(
        shake_graph, budget, num_to_remove, start, end, alpha, beta) + new_solution
    # print("Shake2 New Sol:", new_solution)
    return new_solution


def two_opt(solution, graph: Graph):
    """
    Perform a 2-opt local search on a given route.
    """
    for r, route in enumerate(solution):
        best_route = route[:]
        # Goal is to untangle routes, thus reducing cost
        best_cost = route_det_cost(best_route, graph)
        improved = True
        # 2-opt route until no improvement
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1]
                    # print("check in 2-opt 2")
                    try:
                        cost = route_det_cost(new_route, graph)
                    except:
                        print("Failed switching", route, " to", new_route)
                        exit()
                    if cost < best_cost:
                        best_route = new_route
                        best_cost = cost
                        improved = True
        solution[r] = best_route
    return solution


def remove_subset(solution, graph: Graph):
    """
    Remove a subset of nodes from the routes based on one of three mechanisms.
    """
    mechanisms = ['random', 'highest_reward', 'lowest_reward']
    for i, route in enumerate(solution):
        if len(route) == 2:
            continue
        route = solution[i]
        num_nodes_to_remove = max(
            1, int(len(route) * random.uniform(0.05, 0.10)))
        mechanism = random.choice(mechanisms)
        if mechanism == 'random':
            nodes_to_remove = random.sample(route[1:-1], num_nodes_to_remove)
        elif mechanism == 'highest_reward':
            nodes_to_remove = sorted(
                route[1:-1], key=lambda x: graph.rewards[x], reverse=True)[:num_nodes_to_remove]
        else:  # lowest reward
            nodes_to_remove = sorted(
                route[1:-1], key=lambda x: graph.rewards[x])[:num_nodes_to_remove]

        solution[i] = [node for node in route if node not in nodes_to_remove]
    return solution


def biased_insertion(solution, graph: Graph, budget, beta2=0.3):
    """
    Perform a biased insertion of nodes into the route.
    """
    # print("\nSolution before biased insert:", solution)
    for n, route in enumerate(solution):
        # Find unserved nodes from solution & graph
        served_nodes = []
        for r in solution:
            served_nodes += [v for v in r if v not in served_nodes]
        non_served_nodes = [v for v in graph.vertices if v not in served_nodes]
        # print("Available nodes:", non_served_nodes)
        # Attempt to add unserved nodes to route
        while non_served_nodes:
            # Get candidate nodes with values
            candidate_nodes = []
            for i in range(1, len(route)):
                for node in non_served_nodes:
                    test_route = route[:]
                    test_route.insert(i, node)
                    # Evaluate node if addition yields valid tour
                    # print("check in BIASED INSERT")
                    if route_det_cost(test_route, graph) < budget:
                        cost_increase = graph.get_mean_cost_edgeWork(
                            (route[i-1], node)) + graph.get_mean_cost_edgeWork((node, route[i])) - graph.get_mean_cost_edgeWork((route[i-1], route[i]))
                        reward = graph.rewards[node]
                        candidate_nodes.append(
                            (node, cost_increase / reward, i))
            # print("Candidate nodes:", candidate_nodes)
            if not candidate_nodes:
                break

            candidate_nodes.sort(key=lambda x: x[1])  # sort min to max vals
            # BR select node to insert
            selected_index = geometric_distribution_selection(
                candidate_nodes, beta2)
            selected_node, _, insert_position = candidate_nodes[selected_index]
            # print("Inserting", selected_node, " into route =", route)
            route.insert(insert_position, selected_node)
            non_served_nodes.remove(selected_node)

        solution[n] = route
        # print("Updated solution:", solution)
    # print("Solution after biased insert:", solution)
    return solution


def prob_of_updating(newSol_profit, baseSol_profit, T):
    return math.exp((newSol_profit-baseSol_profit)/T)


def sim_brvns(graph: Graph,
              budget: int,
              num_routes: int,
              start,
              end,
              alpha=0.5,
              beta=0.3,
              k_initial=1,
              k_max=100,
              t_max=1000,
              exploratory_mcs_iters=200,
              intensive_mcs_iters=1000
              ):

    initial_solution = constructive_heuristic(graph,
                                              budget,
                                              num_routes,
                                              start,
                                              end,
                                              alpha,
                                              beta=0.05  # det initial solution
                                              )

    final_solution, reward, reliability = BRVNS(initial_solution,
                                                graph,
                                                budget,
                                                start,
                                                end,
                                                alpha,
                                                beta,
                                                k_initial,
                                                k_max,
                                                t_max,
                                                exploratory_mcs_iters,
                                                intensive_mcs_iters
                                                )
    return final_solution, reward, reliability


if __name__ == "__main__":
    # Define solver parameters
    alpha = 0.5  # (0,1), depending on reward heterogeneity
    # (0,1), controls relative level of greediness present in randomized constructive heuristic
    beta = 0.3
    k_initial = 1
    k_max = 10
    t_max = 250  # Maximum execution time
    exploratory_mcs_iters = 50
    intensive_mcs_iters = 250

    # Example problem-specific parameters
    start = "vs"
    end = "vg"
    size = 20
    edges_mean_range = (3, 10)
    c = 0.05

    graph = create_sop_instance(size,
                                edges_mean_range,
                                c,
                                )

    num_robots = 3
    budget = 20

    sim_brvns(graph,
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
