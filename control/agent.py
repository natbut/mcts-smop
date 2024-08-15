
import numpy as np
import yaml

from control.env_model import EnvironmentModel
from control.task import Task
from sim.comms_manager import CommsManager, Message
from sim.environment import Environment
from solvers.graphing import generate_graph_from_model
from solvers.masop_solver_config import State, fast_simulation
from solvers.my_DecMCTS import ActionDistribution


class Agent:

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict = None) -> None:

        self.id = id
        self.sim_data = sim_data
        self.solver_params = solver_params
        self.merger_params = merger_params
        self.travel_remaining = 0

        if not sim_data["basic"]:
            # Environment status
            self.env_dim_ranges = None
            self.env_model = self._initialize_model(sim_data["env_dims"])
            self.observations = []  # list of observations (grows over time)
            # TODO add some max obs list size

            # Env Processing & Planning parameters
            self.ARRIVAL_THRESH = 1000  # TODO
            self.position_mod_vector = None
            self.local_flow = []  # local flow
            self.energy_burn_rate = sim_data["energy_burn_rate"]

            self.base_loc = sim_data["base_loc"]
            self.location = sim_data["base_loc"]

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

    def _get_valid_schedule_options(self, new_states: list[State]):
        # Trim down to only valid schedules (consider new states and stored)

        states = new_states
        # Add stored states
        if self.my_action_dist != None:
            # Add any currently-stored applicable schedules
            for state in self.my_action_dist.X:
                # Allow any states where action[1] appears in the schedule, then trim the front end of the state's schedule to start with action[1]
                if self.action[1] in state.action_seq:
                    # NOTE trim to self.action[1]
                    # NOTE This also kind of happens in action_update
                    v = state.action_seq[0]
                    while v != self.action[1]:
                        state.action_seq.pop(0)
                        v = state.action_seq[0]
                    # NOTE budgets updated here?
                    state.remaining_budget = self.sim_data["budget"]
                    states.append(state)
        return states

    # TODO this function should be moved to passengers because it uses self.schedule
    def _update_my_best_action_dist(self,
                                    new_act_dist: ActionDistribution,
                                    sim_iters=10):
        # TODO Make this more generic to be usable by Mothership?

        # print("New states input:", new_act_dist.X)
        states = self._get_valid_schedule_options(new_act_dist.X)

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
                # TODO DO LOCAL UTIL REWARDS HERE INSTEAD
                rew, rel = fast_simulation(
                    route_as_edges, self.sim_data["full_graph"], self.sim_data["budget"], sim_iters)
                state.remaining_budget = self.sim_data["budget"]
                # NOTE filter routes to only those above a reliability threshold
                if rel >= self.merger_params["rel_thresh"]:
                    rewards.append(rew)
                    rels.append(rel)
                    usable_states.append(state)

        # print("Agent", self.id, "usable states:", len(usable_states))

        # NOTE attempt to GO HOME if no usable schedules
        if len(usable_states) == 0:
            usable_states.append(
                State([self.schedule[0], self.sim_data["end"]], self.sim_data["budget"]))
            rewards.append(1.0)
            rels.append(1.0)

        # Reduce to only top comms_n states (or fewer)
        # normalize rewards (0,1), sort by rew_norm + rel values
        rewards = np.array(self.normalize(rewards))
        rels = np.array(self.normalize(rels))
        scores = rewards + (rels * self.merger_params["rel_mod"])
        pairs = list(zip(scores, usable_states))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        rewards, states = zip(*sorted_pairs)
        top_states = states[:self.solver_params["comm_n"]]
        top_scores = scores[:self.solver_params["comm_n"]]
        # Update action distribution
        # NOTE we are now using rew+rel as score for determining q vals in hybrid approach
        self.my_action_dist = ActionDistribution(top_states, top_scores)

        print("Agent", self.id, ": Updated distro:\n", self.my_action_dist)

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero if all values are the same
        if max_val == min_val:
            # Arbitrary choice to set all values to 0.5
            return [0.5 for _ in values]

        return [(val - min_val) / (max_val - min_val) for val in values]

    def generate_graph(self, start_task, end_task, filter=True, disp=False):

        planning_dict = {}
        for id, task in self.task_dict.items():
            if filter:
                if (not task.complete) or id == start_task or id == end_task:
                    planning_dict[id] = task
            else:
                planning_dict[id] = task

        return generate_graph_from_model(self.env_model,
                                         planning_dict,
                                         self.env_dim_ranges,
                                         self.sim_data["velocity"],
                                         self.get_energy_cost_from_vel,
                                         self.sim_data["c"],
                                         disp=disp)

        # print("Graph vertices:", self.sim_data["graph"].vertices)
        # print("Graph edges:", self.sim_data["graph"].edges)

    # === COMMS FUNCTIONS ===

    def update_reachable_neighbors(self, comms_mgr: CommsManager):
        # TODO check that this works
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    def send_message(self, comms_mgr: CommsManager, target_id: int, content=None):
        """Create a message & send to neighbor via comms manager"""
        # TODO? add consideration for available neighbors
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    def receive_message(self, comms_mgr: CommsManager, msg: Message):
        """Receive a message"""
        # print("\nReceived message from", msg.sender_id, " Content:", msg.content)
        if msg.sender_id == self.sim_data["m_id"]:  # Message from mothership
            print(self.id, "!!! Received M update")
            if self.id == msg.content[0]:
                # Receiving own schedule
                # Compare schedule to stored elites, update action distro
                self._update_my_best_action_dist(msg.content[1])
            else:
                # Receiving other robot's schedule
                # TODO? Add this to stored action distros for other agents (rather than simply replacing them)
                self.stored_act_dists[msg.content[0]] = msg.content[1]
        else:
            if msg.content[0] == "Update":
                print(self.id, "!!! Received act dist update for",
                      msg.sender_id, ":", msg.content[1])
                self.stored_act_dists[msg.sender_id] = msg.content[1]
            # Return copy of current act dis if sending agent has initiated comms
            if msg.content[0] == "initiate":
                if self.my_action_dist != None:
                    self.send_message(comms_mgr, msg.sender_id,
                                      ("respond", self.my_action_dist))
            if msg.content[0] == "respond":
                self.stored_act_dists[msg.sender_id] = msg.content[1]
            if msg.content[0] == "Edge":
                if self.sim_data["basic"]:
                    self.sim_data["graph"].cost_distributions[msg.content[1]
                                                              ] = msg.content[2]
            if msg.content[0] == "Complete Task":
                self.task_dict[msg.content[1]].complete = True
                print(self.id, "!!! Received task complete:", msg.content[1])
                if self.sim_data["basic"]:
                    if msg.content[1] in self.sim_data["graph"].vertices:
                        self.sim_data["graph"].vertices.remove(msg.content[1])

    # === REALISTIC SIM COMPONENTS ===

    def set_up_dim_ranges(self, env: Environment):
        self.env_dim_ranges = env.get_dim_ranges()

    def load_task(self, id: int, loc: tuple, work: int, reward: int):
        """Adds a task to this agent's task list"""
        if id not in self.task_dict.keys():
            self.task_dict[id] = Task(id, loc, work, reward)

    def load_tasks_on_agent(self, task_dict: dict[Task]):
        for t_id, task in task_dict.items():
            self.load_task(t_id, task.location, task.work, task.reward)

        self.sim_data["full_graph"] = self.generate_graph(
            self.sim_data["start"], self.sim_data["end"], filter=False)

    def reduce_energy(self, velocity=0):
        if self.sim_data["basic"]:
            self.reduce_energy_basic()  # only if not idle
        else:
            self.reduce_energy_by_vel(velocity)

    def reduce_energy_basic(self):
        if self.action[0] != self.IDLE:
            self.sim_data["budget"] -= 1

    def reduce_energy_by_vel(self, vel_mag):
        """
        Given a velocity, reduce energy for 1 timestep of holding
        that velocity

        @param vel_mag: commanded velocity m/timestep
        """
        self.sim_data["budget"] -= self.get_energy_cost_from_vel(vel_mag)

    def get_energy_cost_from_vel(self, vel_mag):
        return self.energy_burn_rate * vel_mag
    # === SENSING and MODEL UPDATES ===

    def _initialize_model(self, dims: tuple) -> EnvironmentModel:
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
            # self.event = True

    def apply_observations_to_model(self):
        """
        Applies agent's local observations to local copy of model, updates resulting task graph for planning
        """
        if self.sim_data["basic"]:
            print("No obs applied because basic sim")
            return

        # Apply to environment model
        # TODO it would be nice to only apply once
        # print("Applying obs", self.observations)
        while len(self.observations) > 0:
            obs = self.observations.pop(0)
            self.env_model.apply_observation(obs)

        # Modify Task Graph edge costs
        # TODO Merge graph and env model w/o Tasks?
        # for task in self.task_dict.values():
        #     for neighbor in self.task_dict.values():
        #         if task.id != neighbor.id:
        #             dist_vec, mean, variance = (
        #                 self.env_model.get_travel_cost_distribution(
        #                     task.location,
        #                     neighbor.location,
        #                     self.env_dim_ranges,
        #                     self.sim_data["velocity"],
        #                 )
        #             )
        #             task.set_distance_to_neighbor(id, dist_vec, mean, variance)


def load_data_from_config(solver_config_fp, problem_config_fp):
    """
    Load in problem and solver data for agents to use during operations
    """
    with open(problem_config_fp, "r") as p_fp:
        prob_config = yaml.safe_load(p_fp)
        with open(solver_config_fp, "r") as s_fp:
            solve_config = yaml.safe_load(s_fp)

            dims = (tuple(prob_config["xCoordRange"]),
                    tuple(prob_config["yCoordRange"]),
                    tuple(prob_config["zCoordRange"]),
                    )

            sim_data = {  # "graph": deepcopy(planning_graph),
                "start": prob_config["start"],
                "end": prob_config["end"],
                "c": prob_config["c"],
                "budget": prob_config["budget"],
                "velocity": prob_config["velocity"],
                "energy_burn_rate": prob_config["energy_burn_rate"],
                "basic": prob_config["basic"],
                "m_id": prob_config["m_id"],
                "env_dims": dims,
                "base_loc": prob_config["base_loc"]
            }

            merger_data = {"rel_mod": solve_config["rel_mod"],
                           "rel_thresh": solve_config["rel_thresh"],
                           "mcs_iters": solve_config["mcs_iters"]
                           }

            dec_mcts_data = {"num_robots": prob_config["num_robots"],
                             "fail_prob": solve_config["failure_probability"],
                             "comm_n": solve_config["comm_n"],
                             "plan_iters": solve_config["planning_iters"],
                             "t_max": solve_config["t_max_decMCTS"],
                             }

            sim_brvns_data = {"num_robots": prob_config["num_robots"],
                              "alpha": solve_config["alpha"],
                              "beta": solve_config["beta"],
                              "k_initial": solve_config["k_initial"],
                              "k_max": solve_config["k_max"],
                              "t_max": solve_config["t_max_simBRVNS"],
                              "explore_iters": solve_config["exploratory_mcs_iters"],
                              "intense_iters": solve_config["intensive_mcs_iters"],
                              }

    return sim_data, dec_mcts_data, sim_brvns_data, merger_data
