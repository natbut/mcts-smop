
import numpy as np

from control.env_model import EnvironmentModel, TaskNode
from sim.comms_manager import CommsManager_Basic, Message
from sim.environment import Environment
# TODO using separate scheduling class we should be able to clear out these imports
from solvers.masop_solver_config import State
from solvers.my_DecMCTS import ActionDistribution
from solvers.sim_brvns import fast_simulation


# TODO: Think about sharing completed tasks between agents
class Agent:

    def __init__(self, id: int, solver_params: dict, sim_data: dict) -> None:

        # Energy and position
        self.id = id
        self.solver_params = solver_params
        self.sim_data = sim_data
        self.travel_remaining = 0

        if not sim_data["basic"]:
            # Environment status
            self.env_dim_ranges = None
            self.env_model = self.initialize_model(sim_data["env_dims"])
            self.observations = []  # list of observations (grows over time)

            # Env Processing & Planning parameters
            self.THRESHOLD = 1000  # TODO
            self.position_mod_vector = None
            self.local_flow = 0.0  # local flow
            self.energy_burn_rate = 0.001  # Wh / (m/s) # TODO

        # Communication variables
        self.stored_act_dists = {}
        self.my_action_dist = None
        self.msg_queue = ...  # TODO?
        self.neighbors_status = (
            {}
        )  # dictionary of True/False reachable status, indexed by agent ID

        # Task-related variables
        self.completed_tasks = [self.sim_data["start"]]
        self.task_dict = {}

    def get_valid_schedule_options(self, new_states: list[State]):
        # Trim down to only valid schedules (consider new states and stored)

        states = new_states
        # Add stored states
        if self.my_action_dist != None:
            # Add any currently-stored applicable schedules
            for state in self.my_action_dist.X:
                # Allow any states where action[1] appears in the schedule, then trim the front end of the state's schedule to start with action[1]
                if self.action[1] in state.action_seq:
                    # NOTE trim to self.action[1]
                    v = state.action_seq[0]
                    while v != self.action[1]:
                        state.action_seq.pop(0)
                        v = state.action_seq[0]
                    # NOTE budgets updated here
                    state.remaining_budget = self.sim_data["budget"]
                    states.append(state)
        return states

    def update_my_best_action_dist(self,
                                   new_act_dist: ActionDistribution,
                                   rel_thresh=0.99,
                                   perf_thresh=0.9,
                                   sim_iters=10):
        # TODO Make this more generic to be usable by Mothership

        # print("New states input:", new_act_dist.X)
        states = self.get_valid_schedule_options(new_act_dist.X)

        # TODO create separate function for this (probably once we incorporate safe returns)
        # Evaluate reward for each reliable state schedule
        rewards = []
        rels = []
        usable_states = []
        for state in states:
            route = state.action_seq
            # Ensure 1 instance per route is evaluated
            # TODO try doing this filter earlier
            new_route = True
            for test in usable_states:
                if test.action_seq == route:
                    new_route = False
            if new_route:
                # Fast MCS to get updated reliability sample
                route_as_edges = [(route[i], route[i+1])
                                  for i in range(len(route)-1)]
                rew, rel = fast_simulation(
                    route_as_edges, self.sim_data["graph"], self.sim_data["budget"], sim_iters)
                state.remaining_budget = self.sim_data["budget"]
                # NOTE filter routes to only those above a reliability threshold
                if rel >= rel_thresh:
                    rewards.append(rew)
                    rels.append(rel)
                    usable_states.append(state)

        print("Agent", self.id, "usable states:", len(usable_states))

        # NOTE attempt to GO HOME if no usable schedules
        if len(usable_states) == 0:
            usable_states.append(
                State([self.schedule[0], self.sim_data["end"]], self.sim_data["budget"]))
            rewards.append(1.0)
            rels.append(1.0)

        # Reduce to only top comms_n states (or fewer)
        # normalize rewards (0,1), sort by rew_norm + rel values
        rewards = np.array(self.normalize(rewards))
        rels = np.array(rels)
        scores = rewards + rels
        pairs = list(zip(scores, usable_states))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        rewards, states = zip(*sorted_pairs)
        top_states = states[:self.solver_params["comm_n"]]
        top_scores = scores[:self.solver_params["comm_n"]]
        # Update action distribution
        # NOTE we are now using rew+rel as score for determining q vals in hybrid approach
        self.my_action_dist = ActionDistribution(top_states, top_scores)
        self.event = False

        print("Agent", self.id, ": Updated distro:\n", self.my_action_dist)

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero if all values are the same
        if max_val == min_val:
            # Arbitrary choice to set all values to 0.5
            return [0.5 for _ in values]

        return [(val - min_val) / (max_val - min_val) for val in values]

    # === COMMS FUNCTIONS ===

    def update_reachable_neighbors(self, comms_mgr: CommsManager_Basic):
        # TODO check that this works
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    def send_message(self, comms_mgr: CommsManager_Basic, target_id: int, content=None):
        """Create a message & send to neighbor via comms manager"""
        # TODO? add consideration for available neighbors
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    def receive_message(self, comms_mgr: CommsManager_Basic, msg: Message):
        """Receive a message"""
        # print("\nReceived message from", msg.sender_id, " Content:", msg.content)
        if msg.sender_id == self.sim_data["m_id"]:  # Message from mothership
            if self.id == msg.content[0]:
                # Receiving own schedule
                # Compare schedule to stored elites, update action distro
                self.update_my_best_action_dist(msg.content[1])
            else:
                # Receiving other robot's schedule
                # TODO? Add this to stored action distros for other agents (rather than simply replacing them)
                self.stored_act_dists[msg.content[0]] = msg.content[1]
        else:
            if msg.content[0] == "Update":
                self.stored_act_dists[msg.sender_id] = msg.content[1]
            # Return copy of current act dis if sending agent has initiated comms
            if msg.content[0] == "initiate":
                if self.my_action_dist != None:
                    self.send_message(comms_mgr, msg.sender_id,
                                      ("respond", self.my_action_dist))
            if msg.content[0] == "Edge":
                self.sim_data["graph"].cost_distributions[msg.content[1]
                                                          ] = msg.content[2]

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
        self.local_flow = env.get_local_flow(self.location)

        # Map observation location to model coordinate
        x_model, y_model = self.env_model.convert_location_to_model_coord(
            self.env_dim_ranges, self.location
        )

        # Add to observation list with model location (if new obs)
        # y,x for row, col
        obs = ((y_model, x_model), (self.local_flow[0], self.local_flow[1]))
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
