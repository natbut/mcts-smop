import yaml
import math
import numpy as np

from control.env_model import EnvironmentModel, TaskNode
from control.op_solver import Scheduler
from sim.environment import Environment
from sim.comms_manager import CommsManager, Message

# Set up with local version of env_model and op_solver

# May also want a queue of comms stuff to process

class Agent:
    
    def __init__(self, id: int, energy: int, env_dims: tuple, location: list[int], velocity: float = 1.0) -> None:
        # Energy and position
        self.id = id
        self.energy = energy
        self.velocity = velocity
        self.location = location
        
        # Environment status
        self.env_dim_ranges = None
        self.env_model = self.initialize_model(env_dims)
        self.observations = [] # list of observations (grows over time)
        
        # Communication variables TODO
        self.msg_queue = ...
        self.neighbors_status = {} # dictionary of True/False reachable status, indexed by agent ID
        
        # Scheduling & control variables TODO
        self.event = False
        self.task_dict = {} # dictionary of TaskNode objects, indexed by task IDs
        self.scheduler = Scheduler() # sop_solver
        self.schedule = [] # List of Task IDs (queue?)
        self.stored_schedules = {} # Stored copies of other agents' schedules, indexed by agent IDs
        
        # Action variables
        self.IDLE = 0
        self.TRAVELING = 1
        self.WORKING = 2
        self.action = [self.IDLE, -1] # Current action tuple (Traveling/Working/Idle, Task ID)
        self.THRESHOLD = 1000 # TODO
        self.work_remaining = 0
        self.position_mod_vector = None
        self.flow = 0.0 # local flow
        self.energy_burn_rate = 0.001 # Wh / (m/s) # TODO
        
    
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
        if z_size == 0: # 2D environment
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
        x_model, y_model = self.env_model.convert_location_to_model_coord(self.env_dim_ranges, self.location)
        
        # Add to observation list with model location (if new obs)
        obs = ((y_model, x_model), (self.flow[0],self.flow[1])) # y,x for row, col
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
                    dist_vec, mean, variance = self.env_model.get_travel_cost_distribution(task.location,
                                                                                    neighbor.location,
                                                                                    self.env_dim_ranges,
                                                                                    self.velocity)
                    task.set_distance_to_neighbor(id, dist_vec, mean, variance)

    # === SCHEDULING ===
    
    # TODO
    def optimize_schedule(self):
        self.event = False
        # optimize schedule
        self.schedule = self.scheduler.optimize_schedule(self.task_dict, self.schedule, self.stored_schedules)
        # communicate optimized schedule
        
    # === COMMS FUNCTIONS ===
        
    # TODO check that this works
    def update_reachable_neighbors(self, comms_mgr: CommsManager):
        self.neighbors_status = comms_mgr.agent_comms_dict[self.id]
        
    # create a message & send to neighbor via comms manager
    # TODO add consideration for available neighbors
    def send_message(self, comms_mgr: CommsManager, target_id: int, content=None):
        msg = Message(self.id, target_id, content)
        comms_mgr.add_message_for_passing(msg)
    
    # receive a message
    def receive_message(self, msg: Message):
        print("Message received by robot", self.id, ':', msg.content)
        
    
    # TODO function to process a message
    
    
    # === ACTIONS ===

    def action_update(self):
        """
        Update action according to current agent status and local schedule
        """
        # print("Current agent action is", self.action) # ("IDLE", -1)
        # If out of energy, don't do anything
        if not self.have_energy_remaining():
            self.action[0] = self.IDLE
            print('AGENT OUT OF ENERGY')
            return
        
        # If home and idle with no schedule, do nothing
        if self.action[0] == self.IDLE and self.action[1] == -1 and len(self.schedule) == 0:
            return
        
        # 0) If Idle, Check if tour is complete
        if self.action[0] == self.IDLE and len(self.schedule) == 0:
            # print("Idle, empty schedule")
            # If schedule is empty, return home
            self.action[0] = self.TRAVELING
            self.action[1] = -1
        elif self.action[0] == self.IDLE and len(self.schedule) > 0:
            # otherwise, start tour & remove first element from schedule
            # print("Traveling to new task")
            self.action[0] = self.TRAVELING
            self.action[1] = self.schedule.pop(0)
        
        task = self.task_dict[self.action[1]]
        self.update_position_mod_vector()
        
        arrived = self.env_model.check_location_within_threshold(self.location,
                                                                task.location,
                                                                self.THRESHOLD
                                                                )
        # 1) If traveling and arrived at task, begin work
        if self.action[0] == self.TRAVELING and arrived:
            # print("Arrived at task. starting Work. Work remaining:", task.work)
            self.action[0] = self.WORKING
            self.work_remaining = task.work
        
        # 2) If working and work is complete, become Idle
        elif self.action[0] == self.WORKING and self.work_remaining <= 0:
            # print("Work complete, becoming Idle")
            self.action[0] = self.IDLE
        elif self.action[0] == self.WORKING and self.work_remaining > 0:
            # otherwise, continue working
            # print("Work in progress")
            self.work_remaining -= 1 # TODO modifier for work rate?

    def have_energy_remaining(self) -> bool:
        return self.energy > 0

    def get_target_location(self):
        return self.task_dict[self.action[1]].location       
    
    def update_position_mod_vector(self):
        dest_loc = self.task_dict[self.action[1]].location
        vector = self.env_model.get_scaled_travel_vector(self.location,
                                                          dest_loc,
                                                          self.velocity) 
        self.position_mod_vector = vector
        # print("Current loc is", self.location, "Destination is", dest_loc)
        # print("Position mod vector is then", vector)
        
    def get_command_velocity(self):
        """
        Returns velocity command required to reach waypoint given 
        local flows
        """
        cmd_vel = tuple(self.position_mod_vector[i] - self.flow[i] for i in 
                                range(len(self.flow)))
        resultant_cmd_vel = round(math.sqrt(cmd_vel[0]**2 + cmd_vel[1]**2), 2)
        # print("Command Vel", resultant_cmd_vel)
        return resultant_cmd_vel

    def reduce_energy(self, vel_mag):
        """
        Given a velocity, reduce energy for 1 timestep of holding
        that velocity
        
        @param vel_mag: commanded velocity m/timestep
        """

        self.energy -= self.energy_burn_rate * vel_mag
        print("Energy reduced to", self.energy)


def generate_agents_from_config(config_filepath) -> list[Agent]:
    
    agent_list = []
    
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
        
        # Environment dimensions for agent models
        dims = (tuple(config["xCoordRange"]),
                tuple(config["yCoordRange"]),
                tuple(config["zCoordRange"])
                )
        
        # Create agents
        for i in range(config["num_agents"]):
            a = Agent(i, config["energy"], dims, config["start_loc"], config["velocity"])
            # Load tasks
            for task in config["tasks"]:
                for key in task.keys():
                    a.load_task(key, task[key]["loc"], task[key]["work"])
            
            a.load_task(-1, config["start_loc"], 0) # load "home" task
            
            agent_list.append(a)
                
    return agent_list