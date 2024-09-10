import random
from copy import deepcopy

import numpy as np

from control.agent import Agent, load_data_from_config
from control.passenger import Passenger
from control.task import Task
from sim.comms_manager import CommsManager
from solvers.masop_solver_config import State
from solvers.my_DecMCTS import ActionDistribution
from solvers.sim_brvns import sim_brvns


class Mothership(Agent):

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict, pssngr_list) -> None:
        super().__init__(id, solver_params, sim_data, merger_params)
        self.type = self.MOTHERSHIP
        self.added_tasks = []
        self.sched_cooldown_dict = {}
        for rob in pssngr_list:
            self.sched_cooldown_dict[rob.id] = 0  # budget at last request
        self.agent_info_dict = {}

    def solve_team_schedules(self, comms_mgr: CommsManager):
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
                                   data["t_max_init"],
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
        for target in self.group_list:
            for a in self.group_list:
                content = (self.id, target.id, "Update",
                           (a.id, self.stored_act_dists[a.id]))
                self.send_msg_down_chain(comms_mgr, content)

    def solve_new_single_schedule(self,
                                  comms_mgr: CommsManager,
                                  agent_id,
                                  budget,
                                  current_schedule,
                                  starting_location,
                                  act_samples=1):

        print("Solving centralized schedule...")

        # Set up solver params
        data = self.solver_params
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], starting_location, 0, 1)

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

        reduced_sched_dists = deepcopy(self.stored_act_dists)
        if reduced_sched_dists and agent_id in reduced_sched_dists.keys():
            del reduced_sched_dists[agent_id]
        data["reduced_sched_dists"] = reduced_sched_dists

        # Reduce vertices to only available tasks
        # This encompasses tasks that M has been told are complete PLUS those that M believes to be scheduled by other agents

        # Planning with multiple samples from stored act dists to get an action_dist of best solutions
        sols = []
        rews = []
        rels = []
        for _ in range(self.solver_params["act_samples"]):
            # alloc_tasks = []
            # for rob_id, act_dist in self.stored_act_dists.items():
            #     if rob_id != agent_id:
            #         alloc_tasks += act_dist.best_action().action_seq[:]
            # TODO: Do random_action here if doing many samples

            # alloc_tasks = set(alloc_tasks)  # + self.completed_tasks)

            # print("Reduced set:", alloc_tasks)
            # print("Tasks to remove:", alloc_tasks)

            data["planning_graph"] = deepcopy(self.sim_data["planning_graph"])
            # for v in alloc_tasks:
            #     if v in data["planning_graph"].vertices:
            #         if v != data["end"] and v != data["start"]:
            #             data["planning_graph"].vertices.remove(v)

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
                                           data["reduced_sched_dists"]
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
        scores = rews * rels
        action_dist = ActionDistribution(sols, scores)

        # Send action_dist back to requesting agent
        content = (self.id, agent_id, "Update", (agent_id, action_dist))
        self.send_msg_down_chain(comms_mgr, content)

        # NOTE though we don't have confirmation that an action_dist was delivered, we save results anyways here to influence scheduling for other agents
        # TODO TRY REMOVING THIS, as it is perhaps a source of confusion for mothership solver
        # self.stored_act_dists[agent_id] = action_dist  # ActionDistribution(
        # [solution[0]], [1])

    def solve_new_schedules_subset(self,
                                   comms_mgr: CommsManager,
                                   agent_ids,
                                   budgets,
                                   starting_locations
                                   ):

        print("Solving centralized schedule...")

        # Set up solver params
        data = self.solver_params
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], starting_location, 0, 1)

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
        # This encompasses tasks that M has been told are complete PLUS those that M believes to be scheduled by other agents

        # Planning with multiple samples from stored act dists to get an action_dist of best solutions
        sols = []
        rews = []
        rels = []
        for _ in range(self.solver_params["act_samples"]):
            alloc_tasks = []
            for rob_id, act_dist in self.stored_act_dists.items():
                if rob_id != agent_id:
                    alloc_tasks += act_dist.random_action().action_seq[:]
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
        scores = rews * rels
        action_dist = ActionDistribution(sols, scores)

        # Send action_dist to all robots
        content = (self.id, agent_id, "Update", (agent_id, action_dist))
        self.send_msg_down_chain(comms_mgr, content)

        # NOTE though we don't have confirmation that an action_dist was delivered, we save results anyways here to influence scheduling for other agents
        self.stored_act_dists[agent_id] = action_dist  # ActionDistribution(
        # [solution[0]], [1])

    def process_msg_content(self, comms_mgr: CommsManager, origin, tag, data):
        super().process_msg_content(comms_mgr, origin, tag, data)

        # Processing messages received from origin other groups
        if tag == "Update":
            # print(self.id, "!!! Received act dist update for",
            #       data[0], ":", data[1])
            self.stored_act_dists[data[0]] = data[1]

            # self.broadcast_message(comms_mgr, self.pssngr_list, content)

            # TODO Removed this as each agent now broadcasts act dist updates to neighboring robots
            # for target in self.group_list:
            #     if target.id != origin:
            #         content = (self.id, target.id, "Update", data)
            #         self.send_msg_down_chain(comms_mgr, content)

        # Return copy of current act dis if sending agent has initiated comms
        elif tag == "Initiate":
            # self.stored_act_dists[data[0]] = data[1]
            # if self.my_action_dist != None:
            for target in self.group_list:
                if target.id != origin:
                    content = (self.id, target.id, "Initiate", data)
                    self.send_msg_down_chain(comms_mgr, content)

        elif tag == "Dead":
            if data[0] == self.id:
                return
            if len(self.stored_act_dists[data[0]].best_action().action_seq) > 0:
                self.stored_act_dists[data[0]] = ActionDistribution(
                    [State([], -1)], [1])
                # self.event = True
                # self.expected_event = False

            for target in self.group_list:
                if target.id != origin:
                    content = (self.id, target.id, "Dead", data)
                    self.send_msg_down_chain(comms_mgr, content)

        elif tag == "Complete Task":
            for task in data:
                if task not in self.glob_completed_tasks:
                    # print("!!! M Received complete task:", task)
                    self.glob_completed_tasks.append(task)
                    self.task_dict[task].complete = True
            # if self.sim_data["basic"]:
            #     if msg.content[1] in self.sim_data["graph"].vertices:
            #         self.sim_data["graph"].vertices.remove(msg.content[1])
            # TODO Removed data sharing here to emphasize how info is aggragated on M
            # for target in self.group_list:
            #     if target.id != origin:
            #         content = (self.id, target.id, "Complete Task", data)
            #         self.send_msg_down_chain(comms_mgr, content)

        elif tag == "Schedule Request":
            # print(self.id, " received Schedule Request")
            # Generate new schedules
            if self.sched_cooldown_dict[origin] != data[0]:
                self.sched_cooldown_dict[origin] == data[0]
                self.solve_new_single_schedule(comms_mgr,
                                               origin,
                                               data[0],
                                               data[1],
                                               data[2],
                                               )


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
                           pssngr_list,
                           rand_base=None) -> Mothership:
    print("Load mothership...")
    sim_data, _, sim_brvns_data, merger_params = load_data_from_config(
        solver_config_fp, problem_config_fp, rand_base)

    return generate_mothership_with_data(sim_data["m_id"],
                                         sim_brvns_data,
                                         sim_data,
                                         merger_params,
                                         pssngr_list)
