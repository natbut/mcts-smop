import random
from copy import deepcopy

import numpy as np

from control.agent import Agent, load_data_from_config
from control.passenger import Passenger
from control.task import Task
from sim.comms_manager import CommsManager, Message
from solvers.masop_solver_config import State
from solvers.my_DecMCTS import ActionDistribution
from solvers.sim_brvns import sim_brvns


class Mothership(Agent):

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict, pssngr_list) -> None:
        super().__init__(id, solver_params, sim_data, merger_params)

        self.pssngr_list = pssngr_list

    def solve_team_schedules(self, comms_mgr: CommsManager, agent_list: list[Passenger]):
        # Get an initial solution
        data = self.solver_params
        self.sim_data["graph"] = self.generate_graph(self.task_dict,
                                                     self.sim_data["start"], self.sim_data["end"],
                                                     filter=False)
        data["graph"] = self.sim_data["graph"]
        data["graph"].vertices.remove(self.sim_data["rob_task"])
        data["budget"] = self.sim_data["budget"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]

        print("M Planning with:", data["graph"].vertices)
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

        # self.sim_data["graph"].vertices.remove(self.sim_data["start"])
        self.task_dict[self.sim_data["start"]].complete = True

        print("Schedules solved:", [s.action_seq for s in solution])

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

# Scheduling functions
    def solve_new_single_schedule(self,
                                  comms_mgr: CommsManager,
                                  agent_id,
                                  budget,
                                  current_schedule,
                                  starting_location,
                                  act_samples=1):

        # No rescheduling for agents that don't need it
        # if (not agent.event) or agent.dead or agent.finished:
        #     return

        if len(current_schedule) <= 1:
            return

        print("Solving centralized schedule...")

        # Set up solver params
        data = self.solver_params
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], starting_location, 0, 1)
        # self.sim_data["start"] = "vr"
        print("Starting Task Location:",
              planning_task_dict[self.sim_data["rob_task"]].location, " Robot location:", starting_location)

        # planning_task_dict[starting_task_id].location = starting_location
        self.sim_data["planning_graph"] = self.generate_graph(
            planning_task_dict,
            self.sim_data["rob_task"],
            self.sim_data["end"],
            filter=True
        )
        data["end"] = self.sim_data["end"]
        data["budget"] = budget
        data["start"] = self.sim_data["rob_task"]
        data["num_robots"] = 1

        # Reduce vertices to only available tasks
        # # This encompasses tasks that M has been told are complete PLUS those that M believes to be scheduled by other agents
        # self.completed_tasks = list(set(
        #     self.completed_tasks + agent.completed_tasks))

        # Planning with multiple samples from stored act dists to get an action_dist of best solutions
        sols = []
        rews = []
        rels = []
        for _ in range(act_samples):
            alloc_tasks = []
            for rob_id, act_dist in self.stored_act_dists.items():
                if rob_id != agent_id:
                    alloc_tasks += act_dist.best_action().action_seq[:]
                    # TODO: Do random_action here if doing many samples

            alloc_tasks = set(alloc_tasks)  # + self.completed_tasks)

            # print("Reduced set:", alloc_tasks)
            # print("Tasks to remove:", alloc_tasks)

            data["planning_graph"] = deepcopy(self.sim_data["planning_graph"])
            for v in alloc_tasks:
                if v in data["planning_graph"].vertices:
                    if v != data["end"] and v != data["start"]:
                        data["planning_graph"].vertices.remove(v)

            print("M Planning with:", data["planning_graph"].vertices)

            if len(data["planning_graph"].vertices) == 0:
                continue

            solution, rew, rel = sim_brvns(data["planning_graph"],
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

            if len(solution) == 0:
                continue

            sols.append(solution[0])
            rews.append(rew)
            rels.append(rel)

        print("Schedules solved:", [s.action_seq for s in sols])
        for s in sols:
            if s.action_seq[0] == self.sim_data["rob_task"]:
                s.action_seq.pop(0)

        if len(sols) == 0:
            return

        # Update action distribution
        # NOTE we are now using rew+rel as score for determining q vals in hybrid approach - these are also re-evaluated when this message is received by agent using agent's local graph
        rews = np.array(self.normalize(rews))
        rels = np.array(self.normalize(rels))
        scores = rews + (rels * self.merger_params["rel_mod"])
        action_dist = ActionDistribution(sols, scores)

        # self.send_message(comms_mgr,
        #                   agent.id,
        #                   (agent.id, action_dist))

        # Send action_dist to all robots
        # TODO I think this is causing confusion in low-comms hybrid replanning because it is not the actual action_dist that an agent is using. We may want to only broadcast actual action_dists received from an agent in "Update" step.

        # for target in self.pssngr_list:
        #     self.send_message(comms_mgr,
        #                       target.id,
        #                       (agent_id, action_dist))
        self.send_message(comms_mgr, agent_id, (agent_id, action_dist))

        # NOTE may need to enforce message sending here while not doing threading
        # NOTE though we don't have confirmation that an action_dist was delivered, we save results anyways here to influence scheduling for other agents
        self.stored_act_dists[agent_id] = action_dist  # ActionDistribution(
        # [solution[0]], [1])

    def receive_message(self, comms_mgr: CommsManager, msg: Message):
        super().receive_message(comms_mgr, msg)

        # Broadcast actual action dists received from passengers to other passengers
        if msg.content[0] == "Update":
            for target in self.pssngr_list:
                if target.id != msg.sender_id:
                    self.send_message(comms_mgr,
                                      target.id,
                                      (msg.sender_id, msg.content[1]))
        # Generate new schedules
        if msg.content[0] == "Schedule Request":
            self.solve_new_single_schedule(comms_mgr,
                                           msg.sender_id,
                                           msg.content[1],
                                           msg.content[2],
                                           msg.content[3],
                                           )

        if msg.content == "Dead":
            for target in self.pssngr_list:
                if target.id != msg.sender_id:
                    content = (msg.sender_id, msg.content)
                    self.send_message(comms_mgr, target.id, content)


def generate_mothership_with_data(id, solver_params, sim_data, merger_params, pssngr_list) -> Mothership:
    """
    Create new Mothership
    """
    m = Mothership(id,
                   deepcopy(solver_params),
                   deepcopy(sim_data),
                   deepcopy(merger_params),
                   pssngr_list
                   )
    return m


def gen_mother_from_config(solver_config_fp,
                           problem_config_fp,
                           #    planning_graph,
                           pssngr_list) -> Mothership:

    sim_data, _, sim_brvns_data, merger_params = load_data_from_config(
        solver_config_fp, problem_config_fp)

    return generate_mothership_with_data(sim_data["m_id"],
                                         sim_brvns_data,
                                         sim_data,
                                         merger_params,
                                         pssngr_list)
