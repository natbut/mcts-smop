import math
import random
from copy import deepcopy

import numpy as np

from control.agent import Agent, load_data_from_config
from control.task import Task
from sim.comms_manager import CommsManager
from solvers.masop_solver_config import State, fast_simulation
from solvers.my_DecMCTS import ActionDistribution, Tree

# TODO: Think about how to share completed tasks between agents


class Passenger(Agent):
    IDLE = 0
    TRAVELING = 1
    WORKING = 2

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict) -> None:

        super().__init__(id, solver_params, sim_data, merger_params)

        # Passenger scheduling & control variables
        self.schedule = []
        self.initial_alloc_reward = 0

        # Passenger action variables
        self.dead = False
        self.finished = False
        self.work_remaining = 0

        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, "Init"]

    # === SCHEDULING FUNCTIONS ===

    def _update_my_best_action_dist(self,
                                    new_act_dist: ActionDistribution
                                    ):
        # Automatically use schedule if currently have no schedule
        if self.action[1] == "Init":
            # TODO? this might struggle in purely distributed as it always uses 1st state only
            self.my_action_dist = new_act_dist
            self.schedule = self.my_action_dist.best_action().action_seq[:]
            self.event = False
            # initial reward
            # self.initial_alloc_reward = sum(
            #     self.sim_data["full_graph"].rewards[v] for v in self.schedule)

            print("!! Schedule selected:", self.schedule, "\n")
            return

        # Otherwise, distro
        super()._update_my_best_action_dist(new_act_dist)

    def optimize_schedule_distr(self, comms_mgr: CommsManager, sim_config):

        # No optimization needed if no events or are dead/finished
        if (not self.event) or self.dead or self.finished:
            return

        # No optimization if returning home or at home
        if self.sim_data["start"] == self.sim_data["end"]:
            self.schedule = []
            self.event = False
            return

        # Evaluate remaining schedule. If failure risk is low then do not optimize
        # NOTE I expect that this will improve performance in low-disturbance situations. However, when robots fail or new tasks are added we would like to mandate that an update happens and skip this check.
        # TODO we've removed this chunk for testing
        # if self.expected_event and "Dist" not in sim_config:
        #     # [[(self.schedule[i], self.schedule[i+1])
        #     route_as_edges = [self.schedule]
        #     #   for i in range(len(self.schedule)-1)]]
        #     # print("Route check 1:", route_as_edges)
        #     planning_task_dict = deepcopy(self.task_dict)
        #     planning_task_dict[self.sim_data["start"]].location = self.location
        #     self.sim_data["sim_graph"] = self.generate_graph(planning_task_dict, self.sim_data["start"], self.sim_data["end"],
        #                                                      filter=False)
        #     _, rel = fast_simulation(
        #         route_as_edges, self.sim_data["sim_graph"], self.sim_data["budget"], self.merger_params["mcs_iters"])

        #     if rel > self.merger_params["rel_thresh"]:
        #         self.event = False
        #         print("Optimization not required")
        #         return

        print("\n! Optimizing schedule")

        # Send schedule request
        if "Hybrid2" in sim_config:
            content = ("Schedule Request",
                       self.sim_data["budget"],
                       self.schedule,
                       self.location
                       )

            self.send_message(comms_mgr,
                              self.sim_data["m_id"],
                              content)
        # Load search tree
        data = self.solver_params
        # print("Generating graph...")
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], self.location, 0, 1)
        self.sim_data["start"] = self.sim_data["rob_task"]
        # planning_task_dict[self.sim_data["start"]].location = self.location
        self.sim_data["plan_graph"] = self.generate_graph(planning_task_dict,
                                                          self.sim_data["start"], self.sim_data["end"],
                                                          filter=True)
        data["task_dict"] = planning_task_dict
        data["graph"] = self.sim_data["plan_graph"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]
        data["budget"] = self.sim_data["budget"]

        print("Distr scheduling with", data["graph"].vertices)

        tree = Tree(data,
                    comm_n=self.solver_params["comm_n"],
                    robot_id=self.id)

        tree.comms = self.stored_act_dists
        # print("Solving...")
        # Optimize schedule to get action dist of candidate schedules
        for _ in range(self.solver_params["plan_iters"]):
            # Alg1. GROW TREE & UPDATE DISTRIBUTION
            tree.grow()
            # print("Iter complete, doing comms...")
            # NOTE removed comms stuff here because unrealistic. Handling elsewhere.
            # TODO test with comms exchange removed during planning
            content = ("initiate", tree.my_act_dist)
            for target in self.neighbors_status:
                if self.neighbors_status[target] and target != self.sim_data["m_id"]:
                    self.send_message(comms_mgr,
                                      target,
                                      ("initiate", tree.my_act_dist))
        for sched in tree.my_act_dist.X:
            if sched.action_seq[0] == self.sim_data["rob_task"]:
                sched.action_seq.pop(0)
        # Evaluate candidate solutions against current stored elites, reduce to subset of size n_comms, select best act sequence from elites as new schedule
        # NOTE only need to do this type of solution merge if hybrid
        if "Hybrid" in sim_config:
            self._update_my_best_action_dist(tree.my_act_dist)
            # Send updated distro to M for use in scheduling other robots
            content = ("Update", self.my_action_dist)
            self.send_message(
                comms_mgr, self.sim_data["m_id"], content)
        else:  # For distributed-only
            self.my_action_dist = tree.my_act_dist

        # Send new action dist to other agents
        content = ("Update", self.my_action_dist)
        for target in self.neighbors_status:
            if self.neighbors_status[target] and target != self.sim_data["m_id"]:
                self.send_message(comms_mgr, target, content)

        # Select schedule to be used by agent
        self.schedule = self.my_action_dist.best_action().action_seq[:]
        print("!! Schedule selected:", self.schedule, "\n")

        # Remove event flag once schedule is updated
        self.event = False
        self.expected_event = True

    # === ACTIONS ===

    def action_update(self, true_graph, comms_mgr, agent_list, sim_config):
        """
        Update action according to current agent status and local schedule
        """
        # print("Passenger", self.id, " Current action:",
        #       self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
        # ("IDLE", -1)
        self.task_dict[self.sim_data["rob_task"]].location = self.location

        # === BREAKING CRITERIA ===
        # If out of energy, don't do anything
        if self.sim_data["budget"] < 0 and not self.finished:
            self.action[0] = self.IDLE
            self.dead = True
            return

        # If home and idle with no schedule, do nothing
        if (self.action[0] == self.IDLE
                and self.action[1] == self.sim_data["end"]):
            self.finished = True
            return

        # If waiting for init schedule command from comms
        if len(self.schedule) == 0 and self.action[1] == "Init":
            if self.my_action_dist != None:
                self.schedule = self.my_action_dist.best_action().action_seq[:]
            else:
                return

        if self.action[0] == self.IDLE and len(self.schedule) > 0:
            print("Traveling to first task in sched:", self.schedule)
            # Start tour & remove first element from schedule
            self.action[0] = self.TRAVELING
            # leaving = self.schedule.pop(0)
            # self.task_dict[leaving].complete = True
            self.action[1] = self.schedule.pop(0)

            # NOTE remove first action (step) from each possible seq
            # Handled by optimize_distr
            # for state in self.my_action_dist.X:
            #     state.action_seq.pop(0)

            if self.sim_data["basic"]:
                edge = (leaving, self.action[1])
                # print(self.id, "Traveling to new task on edge", edge)
                self.travel_remaining = true_graph.get_edge_mean(edge)

                if leaving in self.sim_data["graph"].vertices:
                    self.sim_data["graph"].vertices.remove(leaving)
                    if leaving not in self.completed_tasks:
                        self.completed_tasks.append(leaving)
                self.sim_data["graph"].edges.remove(edge)

            # self.task_dict[leaving].complete = True  # Mark task complete
            self.sim_data["start"] = self.action[1]

        task = self.task_dict[self.action[1]]
        arrived = False

        # 0) Update travel progress if traveling, check arrived
        if self.action[0] == self.TRAVELING:
            # print(self.id, "Travel in progress...")
            if self.sim_data["basic"]:
                # Travel progress for basic
                self.travel_remaining -= self.sim_data["velocity"]
                if self.travel_remaining <= 0:
                    arrived = True
            else:
                # Travel progress for ocean sim
                self.update_position_mod_vector()
                arrived = self.env_model.check_location_within_threshold(
                    self.location, task.location, self.ARRIVAL_THRESH
                )

        # 1) If traveling and arrived at task, begin work
        if self.action[0] == self.TRAVELING and arrived:
            # print(self.id, "Arrived at task. starting Work. Work remaining:", task.work)
            self.action[0] = self.WORKING
            self.work_remaining = task.work

        # 2) If working and work is complete, become Idle
        if self.action[0] == self.WORKING and self.work_remaining <= 0:
            # print(self.id, "Work complete, becoming Idle")
            # self.event = True
            self.action[0] = self.IDLE
            # Mark task complete
            self.completed_tasks.append(self.action[1])
            self.task_dict[self.action[1]].complete = True
            self.sim_data["start"] = self.action[1]
            content = ("Complete Task", self.action[1])
            self.broadcast_message(comms_mgr, agent_list, sim_config, content)

        # 3) Otherwise continue doing work
        elif self.action[0] == self.WORKING and self.work_remaining > 0:
            # otherwise, continue working
            # print(self.id, "Work in progress")
            self.work_remaining -= 1

        # print("Agent", self.id, " Updated action is", self.action,
        #       " | Schedule:", self.schedule)

    # === FUNCTIONS FOR EXPLORING SIMULATION ===

    def broadcast_message(self, comms_mgr, agent_list, sim_config, content):
        for target in agent_list:
            if target.id != self.id:
                self.send_message(comms_mgr,
                                  target.id,
                                  content)
        if "Hybrid" in sim_config:
            self.send_message(
                comms_mgr, self.sim_data["m_id"], content)

    def have_failure(self, comms_mgr, agent_list, sim_config, ):
        self.action[0] = self.IDLE
        self.dead = True
        self.sim_data["budget"] = -1
        print("!! ROBOT", self.id, " FAILURE")

        # NOTE communicates that failure has occurred
        # Send out an empty action distro to all agents and mothership
        self.my_action_dist = ActionDistribution([State([], -1)], [1])
        content = "Dead"
        self.broadcast_message(comms_mgr, agent_list, sim_config, content)

    def failure_update(self, comms_mgr, agent_list, sim_config):
        if self.dead:
            content = "Dead"

            self.broadcast_message(comms_mgr, agent_list, sim_config, content)

    def random_failure(self, comms_mgr, agent_list, sim_config, robot_fail_prob=0.0):
        """
        Active robots may incur random failures, with varying effects on performance
        """
        if self.action[0] != self.IDLE:
            sample = np.random.random()
            if sample > robot_fail_prob:
                # No failure
                return
            self.have_failure(comms_mgr, agent_list, sim_config)

    def edge_discover(self, true_graph, comms_mgr, agent_list, sim_config, discovery_prob):
        """
        Active agents may discover random edges with probability discovery_prob
        """
        if self.action[0] != self.IDLE:
            sample = np.random.random()
            # Prob of returning early
            if sample <= discovery_prob:
                return
            # Sample random edge from true graph
            edge = random.choice(true_graph.edges)
            self.sim_data["graph"].cost_distributions[edge] = true_graph.cost_distributions[edge]

            content = ("Edge", edge, true_graph.cost_distributions[edge])

            for target in agent_list:
                if target.id != self.id:
                    self.send_message(comms_mgr, target.id, content)

            if "Hybrid" in sim_config:
                self.send_message(comms_mgr, -1, content)

    def update_position_mod_vector(self):
        if self.sim_data["basic"]:
            print("No pos vector because basic sim")
            return

        dest_loc = self.task_dict[self.action[1]].location
        # Get modification to position assuming travel at constant velocity
        vector = self.env_model.generate_scaled_travel_vector(
            self.location, dest_loc, self.sim_data["velocity"]
        )
        self.position_mod_vector = vector
        # print("Current loc is", self.location, "Destination is", dest_loc)
        # print("Position mod vector is then", vector)

    # === Helpers for detailed simulations ===

    def get_target_location(self):
        if self.sim_data["basic"]:
            print("No target loc because basic sim")
            return
        if self.action[1] == "Init":
            return self.location
        else:
            return self.task_dict[self.action[1]].location

    def get_command_velocity(self):
        """
        Returns velocity command required to reach waypoint given
        local flows
        """
        if self.sim_data["basic"]:
            print("No cmd vel because basic sim")
            return
        # print("Position Mod:", self.position_mod_vector)
        # print("Local flow:", self.flow)
        cmd_vel = tuple(
            self.position_mod_vector[i] - self.local_flow[i] for i in range(len(self.local_flow))
        )
        resultant_cmd_vel = round(
            math.sqrt(cmd_vel[0] ** 2 + cmd_vel[1] ** 2), 2)
        # print("Command Vel", cmd_vel)
        return resultant_cmd_vel


def generate_passengers_with_data(solver_params, sim_data, merger_params) -> list[Passenger]:
    pssngr_list = []
    for i in range(solver_params["num_robots"]):
        p = Passenger(i,
                      solver_params=deepcopy(solver_params),
                      sim_data=deepcopy(sim_data),
                      merger_params=deepcopy(merger_params)
                      )
        # for v in sim_data["graph"].vertices:
        #     p.load_task(v, None, sim_data["graph"].works[v])
        pssngr_list.append(p)
    return pssngr_list


def generate_passengers_from_config(solver_config_fp,
                                    problem_config_fp,
                                    # planning_graph
                                    ) -> list[Passenger]:

    sim_data, dec_mcts_data, _, merger_data = load_data_from_config(
        solver_config_fp, problem_config_fp)

    return generate_passengers_with_data(dec_mcts_data, sim_data, merger_data)


# def generate_passengers_for_OceanSim(config_filepath) -> list[Passenger]:

#     pssngr_list = []

#     with open(config_filepath, "r") as f:
#         config = yaml.safe_load(f)

#         # Environment dimensions for agent models
#         dims = (
#             tuple(config["xCoordRange"]),
#             tuple(config["yCoordRange"]),
#             tuple(config["zCoordRange"]),
#         )

#         # Create agents
#         for i in range(config["num_agents"]):
#             p = Passenger(
#                 i, config["energy"], dims, config["start_loc"], config["velocity"]
#             )
#             # Load tasks
#             for task in config["tasks"]:
#                 for key in task.keys():
#                     p.load_task(key, task[key]["loc"], task[key]["work"])

#             p.load_task(-1, config["start_loc"], 0)  # load "home" task

#             pssngr_list.append(p)

#     return pssngr_list
