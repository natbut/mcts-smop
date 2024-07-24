import random
from copy import deepcopy

import numpy as np

from control.agent import Agent
from sim.comms_manager import CommsManager_Basic
from solvers.my_DecMCTS import ActionDistribution
from solvers.sim_brvns import sim_brvns


class Mothership(Agent):

    # Scheduling functions
    def solve_new_single_schedule(self, comms_mgr: CommsManager_Basic, agent: Agent, agent_list: list[Agent], act_samples=5, t_limit=100):
        if len(agent.schedule) <= 1:
            return

        # Set up solver params
        data = self.solver_params
        data["end"] = self.sim_data["end"]
        data["budget"] = agent.sim_data["budget"]
        data["start"] = agent.schedule[0]
        data["num_robots"] = 1

        # Reduce vertices to only available tasks
        # This encompasses tasks that M has been told are complete PLUS those that M believes to be scheduled by other agents
        self.completed_tasks = list(set(
            self.completed_tasks + agent.completed_tasks))

        # Planning with multiple samples from stored act dists to get an action_dist of best solutions
        sols = []
        rews = []
        rels = []
        for _ in range(act_samples):
            alloc_tasks = []
            for robot in self.stored_act_dists:
                if [robot] != agent.id:
                    alloc_tasks += self.stored_act_dists[robot].random_action(
                    ).action_seq

            # print("Allocated tasks full:", alloc_tasks)
            # print("Completed tasks full:", self.completed_tasks)

            alloc_tasks = set(alloc_tasks + self.completed_tasks)

            # print("Reduced set:", alloc_tasks)

            data["graph"] = deepcopy(self.sim_data["graph"])
            for v in alloc_tasks:
                data["graph"].vertices.remove(v)

            # print("Planning with:", data["graph"].vertices)

            if len(data["graph"].vertices) == 0:
                return

            # TODO try shorter solver time during runtime b/c multiple samples
            solution, rew, rel = sim_brvns(data["graph"],
                                           data["budget"],
                                           data["num_robots"],
                                           data["start"],
                                           data["end"],
                                           data["alpha"],
                                           data["beta"],
                                           data["k_initial"],
                                           data["k_max"],
                                           data["t_max"],
                                           data["explore_iters"],
                                           data["intense_iters"],
                                           )
            sols.append(solution[0])
            rews.append(rew)
            rels.append(rel)

        # Update action distribution
        # NOTE we are now using rew+rel as score for determining q vals in hybrid approach - these are also re-evaluated when this message is received by agent using agent's local graph
        rews = np.array(self.normalize(rews))
        rels = np.array(rels)
        scores = rews + rels
        action_dist = ActionDistribution(sols, scores)

        # Send action_dist to all robots
        for target in agent_list:
            self.send_message(comms_mgr,
                              target.id,
                              (agent.id, action_dist))

        # NOTE may need to enforce message sending here while not doing threading
        # NOTE though we don't have confirmation that an action_dist was delivered, we save results anyways here to influence scheduling for other agents
        self.stored_act_dists[agent.id] = action_dist  # ActionDistribution(
        # [solution[0]], [1])

    def solve_team_schedules(self, comms_mgr: CommsManager_Basic, agent_list: list[Agent]):
        # Get an initial solution
        data = self.solver_params
        data["graph"] = self.sim_data["graph"]
        data["budget"] = self.sim_data["budget"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]

        # returns list of States
        solution, _, _ = sim_brvns(data["graph"],
                                   data["budget"],
                                   data["num_robots"],
                                   data["start"],
                                   data["end"],
                                   data["alpha"],
                                   data["beta"],
                                   data["k_initial"],
                                   data["k_max"],
                                   data["t_max"],
                                   data["explore_iters"],
                                   data["intense_iters"]
                                   )

        # Pair solution with robot ids
        # paired_solution = {}
        for i, state in enumerate(solution):
            # paired_solution[i] = deepcopy(state)
            self.stored_act_dists[i] = ActionDistribution(
                [deepcopy(state)], [1])
        for i in range(len(solution), self.solver_params["num_robots"]):
            state = random.choice(solution)
            # paired_solution[i] = deepcopy(state)
            self.stored_act_dists[i] = ActionDistribution(
                [deepcopy(state)], [1])

        # Send agent their own plans AND other agents' plans
        for target in agent_list:
            for a in agent_list:
                content = self.stored_act_dists[a.id]
                self.send_message(comms_mgr,
                                  target.id,
                                  (a.id, content))


def generate_mothership_with_data(id, solver_params, sim_data) -> Mothership:
    """
    Create new Mothership
    """
    m = Mothership(id,
                   solver_params=deepcopy(solver_params),
                   sim_data=deepcopy(sim_data)
                   )
    return m
