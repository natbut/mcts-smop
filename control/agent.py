import math
from copy import deepcopy

import numpy as np
import yaml

from control.env_model import EnvironmentModel, TaskNode
from sim.comms_manager import CommsManager_Basic, Message
from sim.environment import Environment
from solvers.decMCTS_config import *
from solvers.my_DecMCTS import Tree

# Set up with local version of env_model and op_solver

# May also want a queue of comms stuff to process


class Agent:

    def __init__(
        self,
        id: int,
        energy: int,
        env_dims: tuple,
        location: list[int],
        prob_data: dict = None,
        velocity: float = 1.0,
    ) -> None:

        # Energy and position
        self.id = id
        if prob_data:
            self.data = prob_data
            self.energy = self.data["budget"]
            self.basic = True
            self.travel_remaining = 0
        else:
            self.energy = energy
            self.velocity = velocity
            self.location = location

        # Environment status
        if env_dims:
            self.env_dim_ranges = None
            self.env_model = self.initialize_model(env_dims)
            self.observations = []  # list of observations (grows over time)

        # Communication variables TODO
        self.received_comms = {}
        self.msg_queue = ...
        self.neighbors_status = (
            {}
        )  # dictionary of True/False reachable status, indexed by agent ID

        # Scheduling & control variables TODO
        self.event = True
        self.action_dist = None
        self.schedule = []
        self.completed_tasks = []

        # Action variables
        self.task_dict = {}
        self.dead = False
        self.finished = False
        self.IDLE = 0
        self.TRAVELING = 1
        self.WORKING = 2
        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, None]

        # Addt. parameters
        self.THRESHOLD = 1000  # TODO
        self.work_remaining = 0
        self.position_mod_vector = None
        self.flow = 0.0  # local flow
        self.energy_burn_rate = 0.001  # Wh / (m/s) # TODO

    # === SCHEDULING ===

    def optimize_schedule(self):
        if self.data["start"] == self.data["end"]:
            self.schedule = []
            return
        # Need all the supporting functions for the tree
        tree = Tree(self.data,
                    reward,
                    avail_actions,
                    state_storer,
                    sim_select_action,
                    sim_get_actions_available,
                    comm_n=self.data["comm_n"],
                    robot_id=self.id)
        tree.comms = self.received_comms

        self.event = False
        # optimize schedule
        for _ in range(self.data["plan_iters"]):
            # Alg1. GROW TREE & UPDATE DISTRIBUTION
            tree.grow()
            # Alg1. TODO? COMMS TRANSMIT & COMMS RECEIVE
            # for i in range(self.data["num_robots"]):
            #     for j in range(self.data["num_robots"]):
            #         if i != j:
            #             self.tree.receive_comms(
            #                 self.tree.send_comms(), j)
            # Alg1. Cool?

        # Alg1. Return action sequence
        self.action_dist = tree.my_act_dist

        # Store best as current schedule
        self.schedule = tree.my_act_dist.best_action().action_seq
        print(self.id, "Schedule:", self.schedule)

    # === COMMS FUNCTIONS ===

    # TODO check that this works

    def update_reachable_neighbors(self, comms_mgr: CommsManager_Basic):
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    # create a message & send to neighbor via comms manager
    # TODO add consideration for available neighbors
    def send_message(self, comms_mgr: CommsManager_Basic, target_id: int, content=None):
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    # receive a message
    def receive_message(self, msg: Message):
        # print("Message received by robot", self.id, ":", msg.content)
        if msg.sender_id == -1:  # FROM MOTHERSHIP
            self.schedule = msg.content
        else:
            self.received_comms[msg.sender_id] = msg.content

    # TODO? function to process a message

        # === REALISTIC SIM COMPONENTS ===
    # Set up to avoid repeated computations

    def set_up_dim_ranges(self, env: Environment):
        self.env_dim_ranges = env.get_dim_ranges()

    def load_task(self, task_id: int, task_loc: tuple, task_work: int):
        """
        Adds a task to this agent's task list
        """
        self.task_dict[task_id] = TaskNode(task_id, (task_loc), task_work)

    # === SENSING and MODEL UPDATES ===

    def initialize_model(self, dims: tuple) -> EnvironmentModel:
        # Dims are env coord ranges (like (100, 195))
        x_size = abs(dims[0][0] - dims[0][1])
        y_size = abs(dims[1][0] - dims[1][1])
        z_size = abs(dims[2][0] - dims[2][1])

        # initialize an environment model, scaled by dimensions
        if z_size == 0:  # 2D environment
            model = EnvironmentModel(y_size, x_size)

        return model

    # Sense loc from environment (location dict with keys agent id)
    def sense_location_from_env(self, env: Environment):
        self.location = env.agent_loc_dict[self.id]

    # Sense flow from environment, log to observations locally
    def sense_flow_from_env(self, env: Environment):
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

    # Apply observations for scheduling
    def apply_observations_to_model(self):
        """
        Applies agent's local observations to local copy of model, updates resulting
        task graph for planning
        """
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
                            self.velocity,
                        )
                    )
                    task.set_distance_to_neighbor(id, dist_vec, mean, variance)

    # === ACTIONS ===

    def action_update(self, sim_config):
        """
        Update action according to current agent status and local schedule
        """
        print("Agent", self.id, " Current agent action is", self.action,
              " | Schedule:", self.schedule)  # ("IDLE", -1)
        # === BREAKING CRITERIA ===
        # If out of energy, don't do anything
        if not self.have_energy_remaining():
            self.action[0] = self.IDLE
            self.dead = True
            print(self.id, "OUT OF ENERGY")
            return

        # If home and idle with no schedule, do nothing
        elif (
            self.action[0] == self.IDLE
            and self.action[1] == self.data["end"]
        ):
            self.finished = True
            # print(self.id, "RETURNED TO BASE")
            return
        # Else awaiting a schedule command from comms
        elif len(self.schedule) == 0 and self.action[1] == None:
            return

        # === UPDATE FROM IDLE ===
        # 0) If Idle, Check if tour is complete
        # if self.action[0] == self.IDLE and len(self.schedule) == 0:
        #     # print(self.id, "Idle, empty schedule")
        #     # If schedule is empty, return home
        #     # NOTE "Home" should now be scheduled as part of tour, so shouldn't see this used
        #     self.action[0] = self.TRAVELING
        #     self.action[1] = -1
        # elif if top is used
        if self.action[0] == self.IDLE and len(self.schedule) > 0:
            # otherwise, start tour & remove first element from schedule
            self.action[0] = self.TRAVELING
            leaving = self.schedule.pop(0)
            self.action[1] = self.schedule[0]

            if self.basic:
                edge = (leaving, self.action[1])
                # print(self.id, "Traveling to new task on edge", edge)
                if "Stoch" in sim_config:
                    self.travel_remaining = self.data["graph"].get_stoch_cost(
                        edge)
                    # print("STOCH EDGE")
                else:
                    self.travel_remaining = self.data["graph"].get_mean_cost(
                        edge)
                    # print("DET EDGE")
                self.data["graph"].vertices.remove(leaving)
                self.data["start"] = self.action[1]
                # print(self.id, "Traveling to",
                #   self.action[1], " with time", self.travel_remaining, " | Rem. Energy:", self.energy)

        # 0) Update travel progress if traveling, check arrived
        task = self.task_dict[self.action[1]]
        arrived = False
        # print(self.id, "Travel in progress...")
        if self.action[0] == self.TRAVELING:
            if self.basic:
                self.travel_remaining -= 1
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

        # NOTE This "instantly" begins work
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

    def have_energy_remaining(self) -> bool:
        return self.energy > 0

    def get_target_location(self):
        return self.task_dict[self.action[1]].location

    def update_position_mod_vector(self):
        dest_loc = self.task_dict[self.action[1]].location
        vector = self.env_model.get_scaled_travel_vector(
            self.location, dest_loc, self.velocity
        )
        self.position_mod_vector = vector
        # print("Current loc is", self.location, "Destination is", dest_loc)
        # print("Position mod vector is then", vector)

    def get_command_velocity(self):
        """
        Returns velocity command required to reach waypoint given
        local flows
        """
        # print("Position Mod:", self.position_mod_vector)
        # print("Local flow:", self.flow)
        cmd_vel = tuple(
            self.position_mod_vector[i] - self.flow[i] for i in range(len(self.flow))
        )
        resultant_cmd_vel = round(
            math.sqrt(cmd_vel[0] ** 2 + cmd_vel[1] ** 2), 2)
        # print("Command Vel", cmd_vel)
        return resultant_cmd_vel

    def reduce_energy_by_Vel(self, vel_mag):
        """
        Given a velocity, reduce energy for 1 timestep of holding
        that velocity

        @param vel_mag: commanded velocity m/timestep
        """

        self.energy -= self.energy_burn_rate * vel_mag
        print("Energy reduced to", self.energy)

    def reduce_energy_basic(self):
        if self.action[0] != self.IDLE:
            # print(self.id, "Reducing energy")
            self.data["budget"] -= 1
            self.energy -= 1


def generate_agents_with_data(data, tasks_work) -> list[Agent]:
    agent_list = []
    for i in range(data["num_robots"]):
        a = Agent(i,
                  energy=None,
                  env_dims=None,
                  location=None,
                  prob_data=deepcopy(data)
                  )
        for key in tasks_work:
            a.load_task(key, None, tasks_work[key])
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
