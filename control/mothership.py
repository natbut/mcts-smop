import random
from copy import deepcopy

import numpy as np

from control.agent import Agent
from sim.comms_manager import CommsManager_Basic, Message
from solvers.decMCTS_config import State
from solvers.my_DecMCTS import ActionDistribution
from solvers.sim_brvns import sim_brvns


class Mothership:

    def __init__(self, id, solver_params, sim_data) -> None:
        self.id = id
        self.solver_params = solver_params
        self.sim_data = sim_data
        self.stored_act_dists = {}
        self.completed_tasks = []
        self.queued_robots = []

    # Scheduling functions
    def solve_new_tour_single(self, comms_mgr: CommsManager_Basic, agent: Agent, agent_list: list[Agent], act_samples=5, t_limit=100):
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

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero if all values are the same
        if max_val == min_val:
            # Arbitrary choice to set all values to 0.5
            return [0.5 for _ in values]

        return [(val - min_val) / (max_val - min_val) for val in values]

    def solve_STOP_schedules(self, comms_mgr: CommsManager_Basic, agent_list: list[Agent]):
        # Get an initial solution
        data = self.solver_params
        data["graph"] = self.sim_data["graph"]
        data["budget"] = self.sim_data["budget"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]

        # returns list of States
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

    # === COMMS FUNCTIONS ===

    def update_reachable_neighbors(self, comms_mgr: CommsManager_Basic):
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    def send_message(self, comms_mgr: CommsManager_Basic, target_id: int, content=None):
        # create a message & send to neighbor via comms manager
        # TODO add consideration for available neighbors
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    def receive_message(self, comms_mgr, msg: Message):
        # receive a message
        if msg.content[0] == "Update":
            self.stored_act_dists[msg.sender_id] = msg.content[1]
        if msg.content[0] == "Edge":
            self.sim_data["graph"].cost_distributions[msg.content[1]
                                                      ] = msg.content[2]


def generate_mothership_with_data(solver_params, sim_data) -> Mothership:
    """
    Create new Mothership
    """
    m = Mothership(-1,
                   solver_params=deepcopy(solver_params),
                   sim_data=deepcopy(sim_data)
                   )
    return m
