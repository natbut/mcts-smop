import math
import random
from copy import deepcopy

import numpy as np
import yaml

from control.agent import Agent
from sim.comms_manager import CommsManager_Basic
from solvers.masop_solver_config import *
from solvers.my_DecMCTS import ActionDistribution, Tree

# TODO: Think about how to share completed tasks between agents


class Passenger(Agent):
    IDLE = 0
    TRAVELING = 1
    WORKING = 2

    def __init__(self, id: int, solver_params: dict, sim_data: dict) -> None:

        super().__init__(id, solver_params, sim_data)

        # Passenger scheduling & control variables
        self.event = True
        self.schedule = None
        self.initial_alloc_reward = 0

        # Passenger action variables
        self.dead = False
        self.finished = False
        self.work_remaining = 0

        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, "Init"]

    # === SCHEDULING FUNCTIONS ===

    def update_my_best_action_dist(self,
                                   new_act_dist: ActionDistribution, rel_thresh=0.99,
                                   perf_thresh=0.9,
                                   sim_iters=10
                                   ):
        # Automatically use schedule if currently have no schedule
        if self.action[1] == "Init":
            # TODO? this might struggle in purely distributed as it always uses 1st state only
            self.my_action_dist = new_act_dist
            self.schedule = self.my_action_dist.best_action().action_seq[:]
            self.event = False
            # initial reward
            self.initial_alloc_reward = sum(
                self.sim_data["graph"].rewards[v] for v in self.schedule)
            return

        # NOTE consider a factor related to self.completed tasks that encourages early return with increasing numbers of completed tasks
        # TODO Naive approach to safe return with perf thresh filter
        # complete_tasks_reward = sum(
        #     self.sim_data["graph"].rewards[v] for v in self.completed_tasks)
        # if complete_tasks_reward/self.initial_alloc_reward > perf_thresh:
        #     return_home_state = State(
        #         [self.schedule[0], self.sim_data["end"]], self.sim_data["budget"])
        #     self.my_action_dist = ActionDistribution([return_home_state], [1])
        #     self.schedule = self.my_action_dist.best_action().action_seq[:]
        #     self.event = False
        #     return

        # Otherwise, distro
        return super().update_my_best_action_dist(new_act_dist,
                                                  rel_thresh,
                                                  perf_thresh,
                                                  sim_iters)

    def optimize_schedule(self, comms_mgr: CommsManager_Basic, agent_list, sim_config):

        # No optimization if returning home or at home
        if self.sim_data["start"] == self.sim_data["end"]:
            self.schedule = []
            self.event = False
            return

        # Load search tree
        data = self.solver_params
        data["graph"] = self.sim_data["graph"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]
        data["budget"] = self.sim_data["budget"]

        tree = Tree(data,
                    local_util_reward,
                    avail_actions,
                    state_storer,
                    sim_select_action,
                    sim_get_actions_available,
                    comm_n=self.solver_params["comm_n"],
                    robot_id=self.id)

        tree.comms = self.stored_act_dists

        # Optimize schedule to get action dist of candidate schedules
        for _ in range(self.solver_params["plan_iters"]):
            # Alg1. GROW TREE & UPDATE DISTRIBUTION
            tree.grow()
            # NOTE removed comms stuff here because unrealistic. Handling elsewhere.
            # TODO do testing with comms exchange added back in...
            for target in agent_list:
                if target.id != self.id:
                    self.send_message(comms_mgr,
                                      target.id,
                                      ("initiate", tree.my_act_dist))

        # Evaluate candidate solutions against current stored elites, reduce to subset of size n_comms, select best act sequence from elites as new schedule
        # NOTE only need to do this type of solution merge if hybrid
        if "Hybrid" in sim_config:
            candidate_states = tree.my_act_dist
            self.update_my_best_action_dist(candidate_states)
            # Send updated distro to M for use in scheduling other robots
            self.send_message(
                comms_mgr, self.sim_data["m_id"], ("Update", self.my_action_dist))
        else:
            self.my_action_dist = tree.my_act_dist

        self.schedule = self.my_action_dist.best_action().action_seq[:]

        # Remove event flag once schedule is updated
        self.event = False

        # Send new action dist to other agents
        for target in agent_list:
            if target.id != self.id:
                self.send_message(comms_mgr,
                                  target.id,
                                  ("Update", self.my_action_dist))

    # === ACTIONS ===

    def action_update(self, true_graph, sim_config):
        """
        Update action according to current agent status and local schedule
        """
        print("Passenger", self.id, " Current action:",
              self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
        # ("IDLE", -1)
        # === BREAKING CRITERIA ===
        # If out of energy, don't do anything
        if self.sim_data["budget"] < 0 and not self.finished:
            self.action[0] = self.IDLE
            self.dead = True
            print("Agent", self.id, " Updated action:",
                  self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
            return

        # If home and idle with no schedule, do nothing
        if (self.action[0] == self.IDLE
                and self.action[1] == self.sim_data["end"]):
            self.finished = True
            print("Agent", self.id, " Updated action:",
                  self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
            return

        # If waiting for init schedule command from comms
        if len(self.schedule) == 0 and self.action[1] == "Init":
            print("Agent", self.id, " Updated action:",
                  self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
            return

        # TODO some tours end with [vg] in schedule. Are these robots dead? Is there a logic error here?
        if self.action[0] == self.IDLE and len(self.schedule) > 0:
            # Start tour & remove first element from schedule
            self.action[0] = self.TRAVELING
            leaving = self.schedule.pop(0)
            self.action[1] = self.schedule[0]
            edge = (leaving, self.action[1])

            # NOTE remove first action (step) from each possible seq
            for state in self.my_action_dist.X:
                state.action_seq.pop(0)

            if self.sim_data["basic"]:
                # print(self.id, "Traveling to new task on edge", edge)
                if "Stoch" in sim_config:
                    self.travel_remaining = true_graph.sample_edge_stoch(
                        edge)
                else:
                    self.travel_remaining = true_graph.get_edge_mean(
                        edge)
                # print("remove", leaving, " from graph verts:",
                #       self.sim_data["graph"].vertices)
                self.sim_data["graph"].vertices.remove(leaving)
                self.sim_data["graph"].edges.remove(edge)
                self.sim_data["start"] = self.action[1]
                # print(self.id, "Traveling to",
                #   self.action[1], " with time", self.travel_remaining, " | Rem. Energy:", self.energy)

        task = self.task_dict[self.action[1]]
        arrived = False

        # 0) Update travel progress if traveling, check arrived
        if self.action[0] == self.TRAVELING:
            # print(self.id, "Travel in progress...")
            if self.sim_data["basic"]:
                self.travel_remaining -= self.sim_data["velocity"]
                if self.travel_remaining <= 0:
                    arrived = True
            else:
                self.update_position_mod_vector()
                arrived = self.env_model.check_location_within_threshold(
                    self.location, task.location, self.THRESHOLD
                )

        # 1) If traveling and arrived at task, begin work
        if self.action[0] == self.TRAVELING and arrived:
            # print(self.id, "Arrived at task. starting Work. Work remaining:", task.work)
            self.action[0] = self.WORKING
            self.work_remaining = task.work

        # 2) If working and work is complete, become Idle
        if self.action[0] == self.WORKING and self.work_remaining <= 0:
            # print(self.id, "Work complete, becoming Idle")
            self.event = True
            self.action[0] = self.IDLE
            self.completed_tasks.append(self.action[1])
        elif self.action[0] == self.WORKING and self.work_remaining > 0:
            # otherwise, continue working
            # print(self.id, "Work in progress")
            self.work_remaining -= 1

        print("Agent", self.id, " Updated action is", self.action,
              " | Schedule:", self.schedule)

    # === FUNCTIONS FOR EXPLORING SIMULATION ===

    def edge_discover(self, true_graph, comms_mgr, agent_list, sim_config, discovery_prob):
        # Active agents may discover random edges with probability discovery_prob
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

    def reduce_energy_basic(self):
        if self.action[0] != self.IDLE:
            self.sim_data["budget"] -= 1

    def reduce_energy_by_vel(self, vel_mag):
        """
        Given a velocity, reduce energy for 1 timestep of holding
        that velocity

        @param vel_mag: commanded velocity m/timestep
        """
        self.sim_data["budget"] -= self.energy_burn_rate * vel_mag

    def update_position_mod_vector(self):
        if self.sim_data["basic"]:
            print("No pos vector because basic sim")
            return

        dest_loc = self.task_dict[self.action[1]].location
        vector = self.env_model.get_scaled_travel_vector(
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
            self.position_mod_vector[i] - self.flow[i] for i in range(len(self.flow))
        )
        resultant_cmd_vel = round(
            math.sqrt(cmd_vel[0] ** 2 + cmd_vel[1] ** 2), 2)
        # print("Command Vel", cmd_vel)
        return resultant_cmd_vel


def generate_passengers_with_data(solver_params, sim_data) -> list[Passenger]:
    pssngr_list = []
    for i in range(solver_params["num_robots"]):
        p = Passenger(i,
                      solver_params=deepcopy(solver_params),
                      sim_data=deepcopy(sim_data)
                      )
        for v in sim_data["graph"].vertices:
            p.load_task(v, None, sim_data["graph"].works[v])
        pssngr_list.append(p)
    return pssngr_list


def generate_passengers_from_config(config_filepath) -> list[Passenger]:

    pssngr_list = []

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

        # Environment dimensions for agent models
        dims = (
            tuple(config["xCoordRange"]),
            tuple(config["yCoordRange"]),
            tuple(config["zCoordRange"]),
        )

        # Create agents
        for i in range(config["num_agents"]):
            p = Passenger(
                i, config["energy"], dims, config["start_loc"], config["velocity"]
            )
            # Load tasks
            for task in config["tasks"]:
                for key in task.keys():
                    p.load_task(key, task[key]["loc"], task[key]["work"])

            p.load_task(-1, config["start_loc"], 0)  # load "home" task

            pssngr_list.append(p)

    return pssngr_list
