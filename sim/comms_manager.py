
import math

import numpy as np

from sim.environment import Environment

# from control.agent import Agent # TODO circular import

# Message class


class Message:

    def __init__(self, sender_id: int, receiver_id: int, content=None) -> None:
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content

        self.success_prob = 1.0
        self.delay = 1

    def set_delay(self, val):
        self.delay = val

    # Also might want a class to modify success prob


class CommsManager:

    def __init__(self, env: Environment, agent_list) -> None:
        self.env = env
        self.agent_dict = {}
        for a in agent_list:
            self.agent_dict[a.id] = a

        self.COMMS_RANGE = 10000  # TODO
        self.COMMS_RATE = 100

        self.agent_comms_dict = {}
        self.update_connections()  # Form connections between agents

        # list of "active messages" being passed by manager
        self.active_msgs = []

    def update_connections(self):
        """
        Updates agent_comms_dict to reflect agent connections as function of distance
        """
        # Iterate through agent connections to update comms dict
        for agent1 in self.env.agent_loc_dict:
            agent_connections = {}
            for agent2 in self.env.agent_loc_dict:
                if agent1 != agent2:
                    agent1_loc = np.array(self.env.agent_loc_dict[agent1])
                    agent2_loc = np.array(self.env.agent_loc_dict[agent2])
                    # Assign 1 if distance within range, 0 otherwise
                    if np.linalg.norm(agent1_loc - agent2_loc) < self.COMMS_RANGE:
                        agent_connections[agent2] = True
                    else:
                        agent_connections[agent2] = False

            self.agent_comms_dict[agent1] = agent_connections

    def get_connections(self, agent_id):
        """
        Returns dictionary of connection status for agent_id
        (to be accessed by an agent to update local status)
        """
        return self.agent_comms_dict[agent_id]

    # Function to receive new messages for passing
    # Called by agent to add a message to comms manager
    def add_message_for_passing(self, msg: Message):
        # Process delay time
        agent1_loc = np.array(self.agent_dict[msg.sender_id].location)
        agent2_loc = np.array(self.agent_dict[msg.receiver_id].location)
        dist = np.linalg.norm(agent1_loc-agent2_loc)
        # print("Message dist:", dist, " Message delay:", msg.delay)
        msg.set_delay(math.ceil(dist/self.COMMS_RATE))
        # Add to active messages
        self.active_msgs.append(msg)

    # function to manage message passing with each time step (considering delays, packet drop)

    def step(self):

        # Update message passing
        for msg in self.active_msgs:
            # Reduce delay if message still passing
            if msg.delay > 0:
                msg.set_delay(msg.delay - 1)
            # Else receive message
            else:
                # TODO - maybe insert some probability that message is lost here
                self.agent_dict[msg.receiver_id].receive_message(self, msg)

                # Remove received message
                self.active_msgs.remove(msg)

        # Update comms graph
        self.update_connections()


class CommsManager_Basic:

    def __init__(self, agent_list, comms_succ_prob) -> None:
        self.agent_dict = {}
        for a in agent_list:
            self.agent_dict[a.id] = a
        self.success_prob = comms_succ_prob
        self.agent_comms_dict = {}

        # list of "active messages" being passed by manager
        self.active_msgs = []

    def add_message_for_passing(self, msg: Message):
        # Function to receive new messages for passing
        # Called by agent to add a message to comms manager
        # Process delay time
        # NOTE may mess around with delay times in future
        msg.set_delay(0)
        # Add to active messages
        self.active_msgs.append(msg)
        # TODO this is for inst. delivery (delete later, probably)
        while self.active_msgs:
            self.step()

    def step(self):
        # function to manage message passing with each time step (considering delays, packet drop)
        # NOTE The pop line here is added to address the inst delivery above. Can likely refactor this code if no longer doing inst delivery
        # Update message passing
        for i, msg in enumerate(self.active_msgs):
            # Reduce delay if message still passing
            if msg.delay > 0:
                msg.set_delay(msg.delay - 1)
            else:
                # Else receive message
                arrive_msg = self.active_msgs.pop(i)
                # probability that message is lost
                samp = np.random.random()
                if samp <= self.success_prob:
                    # Remove & send received message
                    self.agent_dict[msg.receiver_id].receive_message(
                        self, arrive_msg)
