from copy import deepcopy

from control.agent import Agent, load_data_from_config
from control.task import Task
from sim.comms_manager import CommsManager
from solvers.masop_solver_config import State
from solvers.my_DecMCTS import ActionDistribution, Tree


class Passenger(Agent):

    def __init__(self, id: int, solver_params: dict, sim_data: dict, merger_params: dict) -> None:

        super().__init__(id, solver_params, sim_data, merger_params)

        self.type = self.PASSENGER

        self.mothership = None

        # Passenger scheduling & control variables
        self.schedule = []

        # Passenger action variables
        self.finished = False
        self.work_remaining = 0

    # === SCHEDULING FUNCTIONS ===

    def optimize_schedule_distr(self, comms_mgr: CommsManager, sim_config):

        # No optimization needed if no events or are dead/finished
        if (not self.event) or self.dead or self.finished:
            return

        print("\n! Optimizing schedule")

        # Send schedule request
        if "DHyb" in sim_config:
            content = (self.id,
                       self.sim_data["m_id"],
                       "Schedule Request",
                       (self.sim_data["budget"],
                        self.schedule,
                        self.location)
                       )

            self.send_msg_up_chain(comms_mgr, content)

            # If message has been received and processed from M
            if not self.event:
                print("No further schedules should be solved")
                return

        # Load search tree
        data = self.solver_params
        # print("Generating graph...")
        planning_task_dict = deepcopy(self.task_dict)
        planning_task_dict[self.sim_data["rob_task"]] = Task(
            self.sim_data["rob_task"], self.location, 0, 1)
        if self.action[1] != "Init":
            self.sim_data["start"] = self.sim_data["rob_task"]
        # planning_task_dict[self.sim_data["start"]].location = self.location
        self.sim_data["plan_graph"] = self.generate_graph(planning_task_dict,
                                                          self.sim_data["start"], self.sim_data["end"],
                                                          filter=True)

        if self.action[1] == "Init":
            self.sim_data["plan_graph"].vertices.remove(
                self.sim_data["rob_task"])
            planning_task_dict.pop(self.sim_data["rob_task"])

        data["task_dict"] = planning_task_dict
        data["graph"] = self.sim_data["plan_graph"]
        data["start"] = self.sim_data["start"]
        data["end"] = self.sim_data["end"]
        data["budget"] = self.sim_data["budget"]
        data["sim_iters"] = self.solver_params["sim_iters"]

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
            # content = (self.id, self.sim_data["m_id"], "Initiate",
            #            (self.id, tree.my_act_dist))
            # self.send_msg_up_chain(comms_mgr, content)

        for sched in tree.my_act_dist.X:
            if sched.action_seq[0] == self.sim_data["rob_task"]:
                sched.action_seq.pop(0)
        # Evaluate candidate solutions against current stored elites, reduce to subset of size n_comms, select best act sequence from elites as new schedule
        # NOTE only need to do this type of solution merge if hybrid
        if "Hyb" in sim_config or "2Stg" in sim_config:
            self.new_states_to_eval += tree.my_act_dist.X
            self._update_my_best_action_dist()
            # Send updated distro to M for use in scheduling other robots
        else:  # For distributed-only
            self.my_action_dist = tree.my_act_dist

        # Send up to mothership, which then sends to other agents
        # TODO Changed this to a distributed message sending, rather than only sending messages up to M
        for a in self.agent_list:
            if a.id != self.id:
                if a.type == self.PASSENGER and self.neighbors_status[a.id]:
                    content = (self.id, a.id,
                               "Update", (self.id, self.my_action_dist))
                    self.send_msg_up_chain(comms_mgr, content)
        content = (
            self.id, self.sim_data["m_id"], "Update", (self.id, self.my_action_dist))
        self.send_msg_up_chain(comms_mgr, content)

        # Select schedule to be used by agent
        self.schedule = self.my_action_dist.best_action().action_seq[:]
        print("!! Schedule selected:", self.schedule, "\n")

        # Advance age of stored schedules
        for state in self.my_action_dist.X:
            state.age += 1

        # Remove event flag once schedule is updated
        self.event = False
        self.expected_event = True

    # === ACTIONS ===

    def action_update(self, comms_mgr):
        """
        Update action according to current agent status and local schedule
        """
        # print("Passenger", self.id, " Current action:",
        #       self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
        # ("IDLE", -1)
        self.task_dict[self.sim_data["rob_task"]].location = self.location

        # === BREAKING CRITERIA ===
        if self.finished or self.dead:
            # print(self.id, "Finished/Dead! Finished:",
            #       self.finished, " Dead:", self.dead)
            return

        # If out of energy, don't do anything
        if self.sim_data["budget"] < 0 and not self.finished:
            print(self.id, "Dead!")
            self.action[0] = self.IDLE
            self.dead = True
            return

        # If waiting for init schedule command from comms
        if len(self.schedule) == 0 and self.action[1] == "Init":
            if self.my_action_dist != None:
                self.schedule = self.my_action_dist.best_action().action_seq[:]
            else:
                return

        if self.action[0] == self.IDLE and len(self.schedule) > 0:
            # print("Traveling to first task in sched:", self.schedule)
            # Start tour & remove first element from schedule
            self.action[0] = self.TRAVELING
            # leaving = self.schedule.pop(0)
            # self.task_dict[leaving].complete = True
            self.action[1] = self.schedule.pop(0)

        if self.action[0] == self.IDLE:
            # print(self.id, "Resuming travel to", self.action[1])
            self.action[0] = self.TRAVELING

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
            self.glob_completed_tasks.append(self.action[1])
            self.task_dict[self.action[1]].complete = True
            self.stored_reward_sum += self.task_dict[self.action[1]].reward
            # content = (
            #     self.id, self.sim_data["m_id"], "Complete Task", self.glob_completed_tasks)

            # distributed-type message passing here
            for a in self.agent_list:
                if a.id != self.id:
                    if a.type == self.PASSENGER and self.neighbors_status[a.id]:
                        # print(self.id, "Sending completed task to", a.id)
                        content = (
                            self.id, a.id, "Complete Task", [self.action[1]])
                        self.send_msg_up_chain(comms_mgr, content)
            # print(self.id, "Sending completed task to M")
            content = (
                self.id, self.sim_data["m_id"], "Complete Task", [self.action[1]])
            self.send_msg_up_chain(comms_mgr, content)

        # 3) Otherwise continue doing work
        elif self.action[0] == self.WORKING and self.work_remaining > 0:
            # otherwise, continue working
            # print(self.id, "Work in progress")
            self.work_remaining -= 1

        # If arrived at home, set finished to true
        if (arrived and self.action[0] == self.IDLE
                and self.action[1] == self.sim_data["end"]):
            # if no additional tasks are reachable
            self.finished = True
            return

        # print("Agent", self.id, " Updated action is", self.action,
        #       " | Schedule:", self.schedule)

    # === COMMS FUNCTIONS ===

    def process_msg_content(self, comms_mgr: CommsManager, origin, tag, data):
        super().process_msg_content(comms_mgr, origin, tag, data)

        # Message has arrived at destination
        if origin == self.sim_data["m_id"]:
            # Processing messages received from origin mothership
            # print(self.id, "!!! Received M update of tag:", tag)
            if tag == "Dead":
                if data[0] == self.id:
                    return
                if len(self.stored_act_dists[data[0]].best_action().action_seq) > 0:
                    self.stored_act_dists[data[0]] = ActionDistribution(
                        [State([], -1)], [1])
                    # self.event = True
                    # self.expected_event = False
            elif tag == "Initiate":
                self.stored_act_dists[data[0]] = data[1]
                if self.my_action_dist != None:
                    content = (self.id, self.sim_data["m_id"], "Update",
                               (self.id, self.my_action_dist))
                    self.send_msg_up_chain(comms_mgr, content)
            elif tag == "Update" and data[0] == self.id:
                # Receiving own schedule
                # Compare schedule to stored elites, update action distro
                print(self.id, "Received schedule from M:", data[1])
                if self.last_msg_content == data[1]:
                    # No need for repeated updates from same msgs
                    return
                else:
                    self.last_msg_content = data[1]
                    self.new_states_to_eval += data[1].X
                    # BELOW ADDED TO FORCE UPDATES
                    # Update act dist
                    if self.action[1] == "Init":
                        return
                    self._update_my_best_action_dist(force_new=True)
                    # Select schedule
                    self.schedule = self.my_action_dist.best_action(
                    ).action_seq[:]
                    # Don't continue optimizing
                    self.event = False
                    # Broadcast new schedule dist
                    for a in self.agent_list:
                        if a.id != self.id:
                            if a.type == self.PASSENGER and self.neighbors_status[a.id]:
                                content = (self.id, a.id,
                                           "Update", (self.id, self.my_action_dist))
                                self.send_msg_up_chain(comms_mgr, content)
                    content = (
                        self.id, self.sim_data["m_id"], "Update", (self.id, self.my_action_dist))
                    self.send_msg_up_chain(comms_mgr, content)

            elif tag == "Update" and data[0] != self.id:
                # Receiving other robot's schedule
                self.stored_act_dists[data[0]] = data[1]
            elif tag == "New Task":
                for task in data:
                    if task.id not in self.task_dict.keys():
                        self.load_task(task.id,
                                       task.location,
                                       task.work,
                                       task.reward)
            elif tag == "Complete Task":
                for task in data:
                    if task not in self.glob_completed_tasks:
                        self.glob_completed_tasks.append(task)
                    if task in self.task_dict.keys():
                        self.task_dict[task].complete = True
                    else:
                        self.load_task(task, (0, 0, 0), 0, 0)
                        self.task_dict[task].complete = True
        else:
            # Processing messages received from origin other groups
            if tag == "Update":
                # print(self.id, "!!! Received act dist update for",
                #       data[0], ":", data[1])
                self.stored_act_dists[data[0]] = data[1]

            # Return copy of current act dis if sending agent has initiated comms
            elif tag == "Initiate":
                # self.stored_act_dists[data[0]] = data[1]
                if self.my_action_dist != None:
                    content = (self.id, self.sim_data["m_id"], "Update",
                               (self.id, self.my_action_dist))
                    self.send_msg_up_chain(comms_mgr, content)

            elif tag == "Dead":
                if data[0] == self.id:
                    return
                if len(self.stored_act_dists[data[0]].best_action().action_seq) > 0:
                    self.stored_act_dists[data[0]] = ActionDistribution(
                        [State([], -1)], [1])
                    # self.event = True
                    # self.expected_event = False

            elif tag == "Complete Task":
                for task in data:
                    if task in self.task_dict.keys():
                        self.task_dict[task].complete = True
                    else:
                        self.load_task(task, (0, 0, 0), 0, 0)
                        self.task_dict[task].complete = True



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
                                    rand_base=None
                                    # planning_graph
                                    ) -> list[Passenger]:
    print("Load passengers...")
    sim_data, dec_mcts_data, _, merger_data = load_data_from_config(
        solver_config_fp, problem_config_fp, rand_base)

    return generate_passengers_with_data(dec_mcts_data, sim_data, merger_data)
