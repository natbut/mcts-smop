
import math
from copy import deepcopy

import numpy as np
import yaml

from control.env_model import EnvironmentModel
from control.task import Task
from sim.comms_manager import CommsManager, Message
from sim.environment import Environment
from solvers.graphing import generate_graph_from_model
from solvers.masop_solver_config import (State, fast_simulation,
                                         get_state_reward, sim_util_reward)
from solvers.my_DecMCTS import ActionDistribution


class Agent:

    IDLE = 0
    TRAVELING = 1
    WORKING = 2
    MOTHERSHIP = "Mothership"
    PASSENGER = "Passenger"
    SUPPORT = "Support"

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict = None) -> None:

        self.id = id
        self.sim_data = sim_data
        self.solver_params = solver_params
        self.merger_params = merger_params
        self.travel_remaining = 0

        self.agent_list = []
        self.group_list = []

        self.stored_reward_sum = 1
        self.last_msg_content = None

        self.new_states_to_eval = []

        self.event = True
        self.expected_event = True

        self.dead = False

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
        self.glob_completed_tasks = [self.sim_data["start"]]
        self.task_dict = {}
        self.task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], self.location, 0, 1)

        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, "Init"]

    def _combine_stored_and_new_states(self, new_states: list[State]):
        # Trim down to only valid schedules (consider new states and stored)

        states = new_states
        # Add stored states
        if self.my_action_dist != None:
            # Add any currently-stored applicable schedules
            for old_state in self.my_action_dist.X:
                if old_state not in states:
                    states.append(old_state)
        return states

    def _prune_completed_tasks(self, states: list[State]):
        # Remove any completed tasks from proposed schedules
        for state in states:
            # print("Evaluating seq", state.action_seq)
            for task in self.glob_completed_tasks:  # state.action_seq:
                # print("Task", task, "Complete?", self.task_dict[task].complete)
                if task in state.action_seq and task != self.sim_data["end"]:
                    # print("Removing", task, "from ", state.action_seq)
                    state.action_seq.remove(task)

    # TODO this function should be moved to passengers because it uses self.schedule
    def _update_my_best_action_dist(self,
                                    force_new=False):

        # print("Un-pruned new states:", [x.action_seq for x in new_act_dist.X])
        # print("Un-pruned old states:", [
        #       x.action_seq for x in self.my_action_dist.X])

        self._prune_completed_tasks(self.new_states_to_eval)
        self._prune_completed_tasks(self.my_action_dist.X)

        print("New states to eval:", [
              x.action_seq for x in self.new_states_to_eval])
        print("Old states to eval:", [
              x.action_seq for x in self.my_action_dist.X])

        if force_new:
            print("Forcing new state usage")
            states = self.new_states_to_eval
        else:
            states = self._combine_stored_and_new_states(
                self.new_states_to_eval)

        # Evaluate reward for each reliable state schedule
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], self.location, 0, 1)
        self.sim_data["sim_graph"] = self.generate_graph(planning_task_dict,
                                                         self.sim_data["rob_task"],
                                                         self.sim_data["end"],
                                                         filter=False)
        rewards = []
        usable_states = []

        for state in states:
            route = state.action_seq
            # Ensure 1 instance per route is evaluated
            new_route = True
            for test in usable_states:
                # Don't evaluate duplicate routes
                if test.action_seq == route:  # and test.age < state.age:
                    new_route = False
            for test_task_id in route:
                # Exclude routes that contain tasks that we do not have info for
                if test_task_id not in self.task_dict.keys():
                    new_route = False
            if new_route:
                # age = state.age
                rel_route = [self.sim_data["rob_task"]] + route
                print("Evaluating", rel_route)
                _, rel = fast_simulation(
                    [rel_route], self.sim_data["sim_graph"], self.sim_data["budget"], self.merger_params["mcs_iters"])

                # NOTE DOING LOCAL UTIL REWARDS HERE INSTEAD
                # NOTE MCTS solver already tries to consider local util, as does Sim-BRVNS. BUT we do this check because we also are checking against old schedules, and they may perform differently given new state info.
                # rew = sim_util_reward(state,
                #                       self.stored_act_dists,
                #                       self.id,
                #                       self.task_dict,
                #                       self.merger_params["mcs_iters"])
                rew = get_state_reward(state, self.task_dict)

                alpha = rew * rel  # * max(0.01, (1/age))
                print("Reliability:", rel, " Reward:", rew, "Alpha:", alpha)

                if rel > self.merger_params["rel_thresh"]:
                    # state.remaining_budget = self.sim_data["budget"]
                    rewards.append(alpha)
                    # rels.append(rel)
                    usable_states.append(state)
                elif force_new:
                    rewards.append(alpha)
                    # rels.append(alpha)
                    usable_states.append(state)

        # print("Agent", self.id, "usable states:", len(usable_states))

        # NOTE attempt to GO HOME if no usable schedules
        if len(usable_states) == 0:
            if len(self.schedule) == 0:
                usable_states.append(
                    State([self.sim_data["end"]], self.sim_data["budget"]))
            elif self.schedule[0] == self.sim_data["end"]:
                usable_states.append(
                    State([self.schedule[0]], self.sim_data["budget"]))
            else:
                usable_states.append(
                    State([self.schedule[0], self.sim_data["end"]], self.sim_data["budget"]))
            rewards.append(1.0)

        # Reduce to only top comms_n states (or fewer)
        # normalize rewards (0,1), sort by rew_norm + rel values
        # rewards = np.array(self.normalize(rewards))
        # rels = np.array(self.normalize(rels))
        # scores = rewards + (rels * self.merger_params["rel_mod"])
        # print("Rews:", rewards, " Rels:", rels, " Scores:", scores)
        pairs = list(zip(rewards, usable_states))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        # print("Sorted pairs:", [(pair[0], pair[1].action_seq)
        #   for pair in sorted_pairs])
        scores, states = zip(*sorted_pairs)
        # print("Unzipped pairs:", scores, [st.action_seq for st in states])
        top_states = states[:self.solver_params["comm_n"]]
        top_scores = scores[:self.solver_params["comm_n"]]
        # Update action distribution
        # NOTE we are now using rew+rel as score for determining q vals in hybrid approach
        self.my_action_dist = ActionDistribution(top_states, top_scores)

        self.new_states_to_eval = []

        print("Agent", self.id, ": Updated distro:\n", self.my_action_dist)

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero if all values are the same
        if max_val == min_val:
            # Arbitrary choice to set all values to 0.5
            return [0.5 for _ in values]

        return [(val - min_val) / (max_val - min_val) for val in values]

    def generate_graph(self, task_dict, start_task, end_task, filter=True, disp=False):

        planning_dict = {}
        for id, task in task_dict.items():
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
    def broadcast_msg_to_neighbors(self, comms_mgr, content):
        for target in self.agent_list:
            if target.id == self.id:
                continue
            # print(self.id, "broadcasting. Checking status of", target.id)
            if self.neighbors_status[target.id]:
                # print(self.id, "sending message to", target.id)
                self.send_message(comms_mgr,
                                  target.id,
                                  content)
        # if "Hybrid" in sim_config:
        #     self.send_message(
        #         comms_mgr, self.sim_data["m_id"], content)

    def send_msg_up_chain(self, comms_mgr, content):
        final_dest = content[1]
        # print("Final dest: ", final_dest)
        # Quick sends
        # if content[2] == "Update" or content[2] == "Dead":
        #     # Share dist scheduling info with available neighbors
        #     for target in self.agent_list:
        #         if target.id != self.id and target.type == self.PASSENGER:
        #             if self.neighbors_status[target.id]:
        #                 self.send_message(comms_mgr, target.id, content)

        if self.neighbors_status[final_dest]:
            # Send message up to final dest (mothership)
            # print(self.id, "sending",
            #       content[2], "message up to", final_dest)
            self.send_message(comms_mgr,
                              final_dest,
                              content)
            return

        if self.type == self.PASSENGER:
            # Send message up to first support robot
            id = (self.id, self.sim_data["support_robots"]-1)
            if self.neighbors_status[id]:
                # print(self.id, "sending",
                #       content[2], "message up to", id)
                self.send_message(comms_mgr,
                                  id,
                                  content)
                return

        if self.type == self.SUPPORT:
            # Send message up to next support robot
            id = (self.id[0], self.id[1]-1)
            if id[1] < 0:
                pass
            elif self.neighbors_status[id]:
                # print(self.id, "sending",
                #       content[2], "message up to", id)
                self.send_message(comms_mgr,
                                  id,
                                  content)
                return

        # Broadcast-type sends if above have all failed
        for target in self.agent_list:
            if target.id == self.id or not self.neighbors_status[target.id]:
                continue
            send_to_target = False
            if self.type == self.PASSENGER:
                if target.type == self.SUPPORT and target.id[0] == self.id:
                    send_to_target = True
            if self.type == self.SUPPORT:
                if target.type == self.SUPPORT and target.id[0] == self.id[0] and target.id[1] < self.id[1]:
                    send_to_target = True
            if send_to_target:
                self.send_message(comms_mgr,
                                  target.id,
                                  content)

    def send_msg_down_chain(self, comms_mgr, content):
        final_dest = content[1]
        # print("Final dest: ", final_dest)

        # Quick sends
        # if content[2] == "Update" or content[2] == "Dead":
        #     # Share dist scheduling info with available groups (passengers)
        #     for target in self.agent_list:
        #         if target.id != self.id and target.type == self.PASSENGER:
        #             if self.neighbors_status[target.id]:
        #                 self.send_message(comms_mgr, target.id, content)

        if self.neighbors_status[final_dest]:
            # Send message to final dest (a passenger/group)
            # print(self.id, "sending",
            #       content[2], "message down to", final_dest)
            self.send_message(comms_mgr,
                              final_dest,
                              content)
            return

        if self.type == self.MOTHERSHIP:
            # Send message to first robot in chain to final dest
            id = (final_dest, 0)
            if self.neighbors_status[id]:
                # print(self.id, "sending",
                #       content[2], "message down to", id)
                self.send_message(comms_mgr,
                                  id,
                                  content)
                return

        if self.type == self.SUPPORT:
            # Send message to next robot in chain to final dest
            id = (self.id[0], self.id[1]+1)
            if id[1] > self.sim_data["support_robots"]-1:
                pass
            elif self.neighbors_status[id]:
                # print(self.id, "sending",
                #   content[2], "message down to", id)
                self.send_message(comms_mgr,
                                  id,
                                  content)
                return

        # Broadcast-type sends if above have all failed
        for target in self.agent_list:
            # Send msg down toward group
            if target.id == self.id or not self.neighbors_status[target.id]:
                continue
            send_to_target = False
            if self.type == self.MOTHERSHIP:
                if target.type == self.SUPPORT and target.id[0] == final_dest:
                    send_to_target = True
            if self.type == self.SUPPORT:
                if target.type == self.SUPPORT and target.id[0] == self.id[0] and target.id[1] > self.id[1]:
                    send_to_target = True
            if send_to_target:
                # print(self.id, "sending",
                #       content[2], "message down to", target.id)
                self.send_message(comms_mgr,
                                  target.id,
                                  content)

    def update_reachable_neighbors(self, comms_mgr: CommsManager):
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]

    def send_message(self, comms_mgr: CommsManager, target_id: int, content=None):
        """Create a message & send to neighbor via comms manager"""
        # TODO? add consideration for available neighbors
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)

    def process_msg_content(self, comms_mgr: CommsManager, origin, tag, data):
        return

    def forward_msg(self, comms_mgr: CommsManager, msg: Message):
        origin = msg.content[0]
        final_dest = msg.content[1]
        tag = msg.content[2]
        data = msg.content[3]

        # TODO - probably don't need this mothership bit
        if self.type == self.MOTHERSHIP:
            # If this is the mothership, send down chain
            self.send_msg_down_chain(comms_mgr, msg.content)

        elif self.type == self.SUPPORT and final_dest == self.sim_data["m_id"]:
            # If message come from group leader, send up chain
            self.send_msg_up_chain(comms_mgr, msg.content)

        elif self.type == self.SUPPORT and final_dest == self.id[0]:
            # If this is a support robot, check if final_dest == self.id[0]
            self.send_msg_down_chain(comms_mgr, msg.content)

    def receive_message(self, comms_mgr: CommsManager, msg: Message):
        """Receive a message"""
        origin = msg.content[0]
        final_dest = msg.content[1]
        tag = msg.content[2]
        data = msg.content[3]

        if self.id == final_dest:
            if tag == "Complete Task":
                print(self.id, "Received complete task",
                      data, "from", msg.sender_id)
            self.process_msg_content(comms_mgr, origin, tag, data)
        else:
            self.forward_msg(comms_mgr, msg)

    # === REALISTIC SIM COMPONENTS ===

    def set_up_dim_ranges(self, env: Environment):
        self.env_dim_ranges = env.get_dim_ranges()

    def load_task(self, id: int, loc: tuple, work: int, reward: int):
        """Adds a task to this agent's task list"""
        if id not in self.task_dict.keys():
            self.task_dict[id] = Task(id, loc, work, reward)
        # print("After loading task", id, " new dict for",
        #       self.id, " is: ", self.task_dict.keys())

    def load_tasks_on_agent(self, task_dict: dict[Task]):
        for t_id, task in task_dict.items():
            self.load_task(t_id, task.location, task.work, task.reward)

        self.sim_data["sim_graph"] = self.generate_graph(self.task_dict,
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

    def have_failure(self, comms_mgr):
        self.action[0] = self.IDLE
        self.dead = True
        self.sim_data["budget"] = -1

        # NOTE communicates that failure has occurred
        # Send out an empty action distro to all agents and mothership
        if self.type == self.PASSENGER:
            self.my_action_dist = ActionDistribution([State([], -1)], [1])
            content = (self.id, self.sim_data["m_id"], "Dead", (self.id, None))
            self.send_msg_up_chain(comms_mgr, content)

    def random_failure(self, comms_mgr, robot_fail_prob=0.0):
        """
        Active robots may incur random failures, with varying effects on performance
        """
        if self.action[0] != self.IDLE:
            sample = np.random.random()
            if sample <= robot_fail_prob:
                self.have_failure(comms_mgr)

    def update_position_mod_vector(self, loc=[]):
        if self.sim_data["basic"]:
            print("No pos vector because basic sim")
            return

        if len(loc) == 0:
            dest_loc = self.task_dict[self.action[1]].location
        else:
            dest_loc = loc
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
        # print("Local flow:", self.local_flow)
        cmd_vel = tuple(
            self.position_mod_vector[i] - self.local_flow[i] for i in range(len(self.local_flow))
        )
        resultant_cmd_vel = np.linalg.norm(cmd_vel)
        # print("Command Vel", cmd_vel)
        return resultant_cmd_vel


# Setup functions

def load_data_from_config(solver_config_fp, problem_config_fp, rand_base=None):
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
                "base_loc": prob_config["base_loc"],
                "rob_task": prob_config["rob_task"],
                "support_robots": prob_config["support_robots"]
            }
            if rand_base:
                sim_data["base_loc"] = rand_base

            merger_data = {"rel_mod": solve_config["rel_mod"],
                           "rel_thresh": solve_config["rel_thresh"],
                           "mcs_iters": solve_config["mcs_iters"]
                           }

            dec_mcts_data = {"num_robots": prob_config["num_robots"],
                             "fail_prob": solve_config["failure_probability"],
                             "comm_n": solve_config["comm_n"],
                             "plan_iters": solve_config["planning_iters"],
                             "t_max": solve_config["t_max_decMCTS"],
                             "sim_iters": solve_config["sim_iters"]
                             }

            sim_brvns_data = {"num_robots": prob_config["num_robots"],
                              "alpha": solve_config["alpha"],
                              "beta": solve_config["beta"],
                              "k_initial": solve_config["k_initial"],
                              "k_max": solve_config["k_max"],
                              "t_max": solve_config["t_max_simBRVNS"],
                              "t_max_init": solve_config["t_max_init"],
                              "explore_iters": solve_config["exploratory_mcs_iters"],
                              "intense_iters": solve_config["intensive_mcs_iters"],
                              "act_samples": solve_config["act_samples"],
                              }

    return sim_data, dec_mcts_data, sim_brvns_data, merger_data
