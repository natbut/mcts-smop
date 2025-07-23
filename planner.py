from copy import deepcopy
from control.passenger import generate_passengers_with_data
from control.mothership import generate_mothership_with_data
from sim.comms_manager import CommsManager_Basic, Message

class HybDecPlanner:

    def __init__(self, sim_data, merger_data, dec_mcts_data, sim_brvns_data):

        # Create agents
        self.passngr_list = generate_passengers_with_data(dec_mcts_data, sim_data, merger_data)

        self.mothership = generate_mothership_with_data(sim_data["m_id"],
                                                        sim_brvns_data,
                                                        sim_data,
                                                        merger_data,
                                                        self.passngr_list,
                                                        )
        
        # Intitialize models
        self.all_agents = self.passngr_list + [self.mothership]
        for a in self.all_agents:
            a.env_model = a._initialize_model(sim_data["env_dims"])
            a.env_dim_ranges = sim_data["env_dims"]

        # Create comms framework
        self.comms_mgr = CommsManager_Basic(self.all_agents,
                                            1.0,
                                            1.0,
                                            self.passngr_list[0].sim_data["m_id"])
        


    def prepare_init_plans(self,
                           task_dict,
                           workers_pos,
                           base_loc,
                           x_ranges=(-1, 1),
                           y_ranges=(-1, 1)):

        x_min, y_min = x_ranges
        x_max, y_max = y_ranges
        self.task_dict = task_dict


        # Set up initial env state
        self.mothership.agent_list = self.all_agents
        self.mothership.group_list = self.passngr_list
        self.mothership.update_reachable_neighbors(self.comms_mgr)
        self.mothership.task_dict = {} # reset
        self.mothership.stored_act_dists = {} # reset
        self.mothership.my_action_dist = None # reset
        self.mothership.load_tasks_on_agent(task_dict)
        for i, p in enumerate(self.passngr_list):
            p.agent_list = self.all_agents
            p.group_list = self.passngr_list
            p.mothership = self.mothership
            p.base_loc = base_loc
            p.update_reachable_neighbors(self.comms_mgr)
            p.env_dim_ranges = ((x_min, x_max), (y_min, y_max), (0, 0))
            p.location = workers_pos[i]
            p.task_dict = {} # reset
            p.my_action_dist = None # reset
            p.stored_act_dists = {} # reset
            p.schedule = [] # reset
            p.load_tasks_on_agent(task_dict)

        self.mothership.env_dim_ranges = ((x_min, x_max), (y_min, y_max), (0, 0))
        self.mothership.base_loc = base_loc
        self.mothership.location = base_loc
        
        # Generate initial plan & Load on Passengers
        print("Solving team schedules...")
        self.mothership.solve_team_schedules(self.comms_mgr)
        for p in self.passngr_list:
            p.initialize_schedule()

    
    def get_agent_plan(self, agent_id):
        plan_ids = self.passngr_list[agent_id].schedule
        print(f"Passenger {agent_id} plan: {plan_ids}")
        locs = [self.task_dict[id].location[:2] for id in plan_ids if id != self.mothership.sim_data["start"]]
        return locs


    def solve_worker_schedules(self, locations, task_dict, planning_iters=5):
        """Uses HybDec solver to decentrally solve schedules for worker robots"""

        self.task_dict = task_dict

        self.mothership.task_dict = {} # reset
        self.mothership.stored_act_dists = {} # reset
        self.mothership.my_action_dist = None
        self.mothership.load_tasks_on_agent(task_dict)
        for i, p in enumerate(self.passngr_list):
            # Update observations
            p.location = locations[i]
            p.task_dict = {} # reset
            p.my_action_dist = None # reset
            p.stored_act_dists = {} # reset
            p.schedule = [] # reset
            p.load_tasks_on_agent(task_dict)

        sim_config = "DHyb"
        for i in range(planning_iters):
            for p in self.passngr_list:
                if p.type == p.PASSENGER:
                    # Solve for new action distro
                    p.event = True
                    p.optimize_schedule_distr(self.comms_mgr, sim_config)

        print("Scheduling Done")