import random
from copy import deepcopy

from sim.comms_manager import CommsManager_Basic, Message
from solvers.sim_brvns import sim_brvns


class Mothership:

    def __init__(self, id, prob_data, tasks_work) -> None:
        self.id = id
        self.data = prob_data
        self.tasks_work = tasks_work
        self.received_comms = {}

    def solve_schedules(self):
        # Get an initial solution
        solution = sim_brvns(self.data["graph"],
                             self.data["budget"],
                             self.data["num_robots"],
                             self.data["end"],
                             self.data["alpha"],
                             self.data["beta"],
                             self.data["k_initial"],
                             self.data["k_max"],
                             self.data["t_max"],
                             self.data["explore_iters"],
                             self.data["intense_iters"]
                             )
        print("Solution:", solution)
        # Pair solution with robot ids
        paired_solution = {}
        for i, route in enumerate(solution):
            paired_solution[i] = deepcopy(route)
            print("Assign robot", i, " route", route)

        for i in range(len(solution), self.data["num_robots"]):
            route = random.choice(solution)
            paired_solution[i] = deepcopy(route)
            print("Assign robot", i, " route", route)

        return paired_solution

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
        self.received_comms[msg.sender_id] = msg.content
