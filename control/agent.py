import math
from copy import deepcopy

import numpy as np
import yaml

from control.env_model import EnvironmentModel, TaskNode
from sim.comms_manager import CommsManager_Basic, Message
from sim.environment import Environment
from solvers.decMCTS_config import *
from solvers.my_DecMCTS import ActionDistribution, Tree
from solvers.sim_brvns import fast_simulation

# TODO: Think about how to share completed tasks between agents


class Agent:
    IDLE = 0
    TRAVELING = 1
    WORKING = 2

    def __init__(
        self,
        id: int,
        solver_params: dict,
        sim_params: dict,
    ) -> None:

        # Energy and position
        self.id = id
        self.solver_params = solver_params
        self.sim_data = sim_params
        self.travel_remaining = 0

        if not sim_params["basic"]:
            # Environment status
            self.env_dim_ranges = None
            self.env_model = self.initialize_model(sim_params["env_dims"])
            self.observations = []  # list of observations (grows over time)

            # Env Processing parameters
            self.THRESHOLD = 1000  # TODO
            self.position_mod_vector = None
            self.flow = 0.0  # local flow
            self.energy_burn_rate = 0.001  # Wh / (m/s) # TODO

        # Communication variables
        self.stored_act_dists = {}
        self.my_action_dist = None
        self.msg_queue = ...  # TODO?
        self.neighbors_status = (
            {}
        )  # dictionary of True/False reachable status, indexed by agent ID

        # Scheduling & control variables
        self.event = True
        self.schedule = None
        self.initial_alloc_reward = 0
        self.completed_tasks = [self.sim_data["start"]]

        # Action variables
        self.task_dict = {}
        self.dead = False
        self.finished = False
        self.work_remaining = 0
        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, "Init"]

    # === SCHEDULING FUNCTIONS ===

    def optimize_schedule(self, comms_mgr: CommsManager_Basic, agent_list):
        # No optimization if returning home or at home
        if self.sim_data["start"] == self.sim_data["end"]:
            self.schedule = []
            return

        # Load search tree
        data = self.solver_params
        data["graph"] = self.sim_data["graph"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]
        data["budget"] = self.sim_data["budget"]

        tree = Tree(data,
                    reward,
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

        # Evaluate candidate solutions against current stored elites, reduce to subset of size n_comms, select best act sequence from elites as new schedule
        candidate_states = tree.my_act_dist.X
        self.update_my_best_action_dist(candidate_states)

        # Send new action dist to other agents
        for target in agent_list:
            if target.id != self.id:
                self.send_message(comms_mgr,
                                  target.id,
                                  self.my_action_dist)

    def get_valid_schedule_options(self, new_states: list[State]):
        # Trim down to only valid schedules (new states plus valid stored)
        states = new_states
        # Add stored states
        if self.my_action_dist != None:
            # Add any currently-stored applicable schedules
            for state in self.my_action_dist.X:
                # TODO? consider allowing any states where action[1] appears in the schedule, then trimming the front end of the state's schedule down to start with action[1]
                if self.action[1] == state.action_seq[0]:
                    states.append(state)
        return states

    def update_my_best_action_dist(self,
                                   new_states: list[State],
                                   rel_thresh=0.99,
                                   perf_thresh=0.5,
                                   sim_iters=10):
        # Automatically use schedule if currently have no schedule
        if self.action[1] == "Init":
            # TODO this might struggle with purely distributed as it always uses 1st state only
            self.my_action_dist = ActionDistribution([new_states[0]], [1])
            self.schedule = self.my_action_dist.best_action().action_seq[:]
            self.event = False
            # initial reward
            self.initial_alloc_reward = sum(
                self.sim_data["graph"].rewards[v] for v in self.schedule)  # TODO
            return
        # TODO think about assigning current remaining budget to any/all states being saved in elites

        # TODO Naive approach to safe return
        complete_tasks_reward = sum(
            self.sim_data["graph"].rewards[v] for v in self.completed_tasks)
        if complete_tasks_reward/self.initial_alloc_reward > perf_thresh:
            return_home_state = State(
                [self.schedule[0], self.sim_data["end"]], self.sim_data["budget"])
            self.my_action_dist = ActionDistribution([return_home_state], [1])
            self.schedule = self.my_action_dist.best_action().action_seq[:]
            self.event = False
            return

        states = self.get_valid_schedule_options(new_states)

        # TODO create separate function for this (probably once we incorporate safe returns)
        # Evaluate reward for each reliable state schedule
        rewards = []
        usable_states = []
        for state in states:
            route = state.action_seq
            # Ensure 1 instance per route is evaluated
            # TODO try doing this filter earlier
            new_route = True
            for test in states:
                if test.action_seq == route:
                    new_route = False
            if new_route:
                # Fast MCS to get updated reliability sample
                route_as_edges = [(route[i], route[i+1])
                                  for i in range(len(route)-1)]
                rew, rel = fast_simulation(
                    route_as_edges, self.sim_data["graph"], self.sim_data["budget"], sim_iters)
                # NOTE filter routes to only those above a reliability threshold
                if rel >= rel_thresh:
                    rewards.append(rew)
                    usable_states.append(state)

        print("Agent", self.id, "usable states:", len(usable_states))

        # NOTE attempt to GO HOME if no usable schedules
        # NOTE consider a factor related to self.completed tasks that encourages early return with increasing numbers of completed tasks
        if len(usable_states) == 0:
            usable_states.append(
                State([self.schedule[0], self.sim_data["end"]], self.sim_data["budget"]))
            rewards.append(1.0)

        # Reduce to only top comms_n states (or fewer)
        # TODO normalize rewards (0,1), sort by rew_norm + rel values
        pairs = list(zip(rewards, usable_states))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        rewards, states = zip(*sorted_pairs)
        top_states = states[:self.solver_params["comm_n"]]
        top_rewards = rewards[:self.solver_params["comm_n"]]
        # Update action distribution
        self.my_action_dist = ActionDistribution(top_states, top_rewards)
        self.schedule = self.my_action_dist.best_action().action_seq[:]
        self.event = False

        print("Agent", self.id, ": Updated distro:\n", self.my_action_dist)

    # === COMMS FUNCTIONS ===

    def update_reachable_neighbors(self, comms_mgr: CommsManager_Basic):
        # TODO check that this works
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    def send_message(self, comms_mgr: CommsManager_Basic, target_id: int, content=None):
        """Create a message & send to neighbor via comms manager"""
        # TODO? add consideration for available neighbors
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    def receive_message(self, msg: Message):
        """Receive a message"""
        # print("\nReceived message from", msg.sender_id, " Content:", msg.content)
        if msg.sender_id == -1:  # Message from mothership
            if self.id == msg.content[0]:
                print("!!! Agent", self.id, " Received new schedule from M:",
                      msg.content[1].action_seq)
                print("with current action:", self.action,
                      " | and Schedule:", self.schedule)
                # Receiving own schedule
                # Compare schedule to stored elites, update action distro
                self.update_my_best_action_dist([msg.content[1]])
            else:
                # Receiving other robot's schedule
                # TODO? Add this to stored action distros for other agents (rather than simply replacing them)
                other_act_dist = ActionDistribution([msg.content[1]], [1])
                self.stored_act_dists[msg.content[0]] = other_act_dist
        else:
            self.stored_act_dists[msg.sender_id] = msg.content

    # === REALISTIC SIM COMPONENTS ===

    def set_up_dim_ranges(self, env: Environment):
        self.env_dim_ranges = env.get_dim_ranges()

    def load_task(self, task_id: int, task_loc: tuple, task_work: int):
        """Adds a task to this agent's task list"""
        self.task_dict[task_id] = TaskNode(task_id, (task_loc), task_work)

    # === SENSING and MODEL UPDATES ===

    def initialize_model(self, dims: tuple) -> EnvironmentModel:
        """initialize an environment model, scaled by dimensions"""
        if self.sim_data["basic"]:
            print("No env model because basic sim")
            return

        # Dims are env coord ranges (like (100, 195))
        x_size = abs(dims[0][0] - dims[0][1])
        y_size = abs(dims[1][0] - dims[1][1])
        z_size = abs(dims[2][0] - dims[2][1])

        if z_size == 0:  # 2D environment
            model = EnvironmentModel(y_size, x_size)

        return model

    def sense_location_from_env(self, env: Environment):
        """
        Sense loc from environment (location dict with keys agent id)
        """
        if self.sim_data["basic"]:
            print("No loc sense because basic sim")
            return

        self.location = env.agent_loc_dict[self.id]

    def sense_flow_from_env(self, env: Environment):
        """
        Sense flow from environment, log to observations locally
        """
        if self.sim_data["basic"]:
            print("No flow sense because basic sim")
            return

        # Get flow from actual agent location
        self.flow = env.get_local_flow(self.location)

        # Map observation location to model coordinate
        x_model, y_model = self.env_model.convert_location_to_model_coord(
            self.env_dim_ranges, self.location
        )

        # Add to observation list with model location (if new obs)
        # y,x for row, col
        obs = ((y_model, x_model), (self.flow[0], self.flow[1]))
        if obs not in self.observations:
            self.observations.append(obs)
            self.event = True

    def apply_observations_to_model(self):
        """
        Applies agent's local observations to local copy of model, updates resulting task graph for planning
        """
        if self.sim_data["basic"]:
            print("No obs applied because basic sim")
            return
        # Apply to environment model
        for obs in self.observations:
            self.env_model.apply_observation(obs)

        # Modify Task Graph edge costs
        for task in self.task_dict.values():
            for neighbor in self.task_dict.values():
                if task.id != neighbor.id:
                    dist_vec, mean, variance = (
                        self.env_model.get_travel_cost_distribution(
                            task.location,
                            neighbor.location,
                            self.env_dim_ranges,
                            self.sim_data["velocity"],
                        )
                    )
                    task.set_distance_to_neighbor(id, dist_vec, mean, variance)

    # === ACTIONS ===

    def action_update(self, true_graph, sim_config):
        """
        Update action according to current agent status and local schedule
        """
        print("Agent", self.id, " Current action:",
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

    # def have_energy_remaining(self) -> bool:
    #     return self.energy > 0

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


def generate_agents_with_data(solver_params, sim_data) -> list[Agent]:
    agent_list = []
    for i in range(solver_params["num_robots"]):
        a = Agent(i,
                  solver_params=deepcopy(solver_params),
                  sim_params=deepcopy(sim_data)
                  )
        for v in sim_data["graph"].vertices:
            a.load_task(v, None, sim_data["graph"].works[v])
        agent_list.append(a)
    return agent_list


def generate_agents_from_config(config_filepath) -> list[Agent]:

    agent_list = []

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
            a = Agent(
                i, config["energy"], dims, config["start_loc"], config["velocity"]
            )
            # Load tasks
            for task in config["tasks"]:
                for key in task.keys():
                    a.load_task(key, task[key]["loc"], task[key]["work"])

            a.load_task(-1, config["start_loc"], 0)  # load "home" task

            agent_list.append(a)

    return agent_list
