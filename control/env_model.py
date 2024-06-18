import math
import numpy as np
import matplotlib.pyplot as plt
    
# Environment model as lattice graph, where each node contains 
class EnvironmentModel:
    def __init__(self, y_dim, x_dim, mean=0, variance=1):
        self.rows = y_dim
        self.cols = x_dim
        # Initialize the lattice with vectors (x, y) where both x and y are normally distributed
        self.means = np.full((self.rows, self.cols, 2), mean, dtype=float) # np.random.normal(mean, variance, (rows, cols, 2))
        self.variances = np.full((self.rows, self.cols, 2), variance, dtype=float)
        
    def apply_observation(self, observation: tuple):
        """
        Apply an environment observation to model
        
        @param observation: ((x,y), (u,v)); (x,y) location, (u,v) flow vector
        """
        location = observation[0]
        flow_comps = observation[1]
        x, y = location # TODO will need to update for 3D case
        obs_x, obs_y = flow_comps
        
        # Update the mean and variance at the observed cell
        self.means[x, y] = np.asarray([obs_x, obs_y], dtype=float)
        self.variances[x, y] = np.asarray([0, 0], dtype=float)
        
        # Propagate the mean and variance to the neighboring cells
        self._propagate(x, y)
        
    # TODO propagate beyond just neighboring cells
    def _propagate(self, x, y):
        """
        Propagate ocean flow means and variances through model
        @param x: x-coord of applied observation
        @param y: y-coord of applied observation
        """
        neighbors = [
            (x - 1, y), (x + 1, y), 
            (x, y - 1), (x, y + 1)
        ]
        
        for nx, ny in neighbors:
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                # Update the mean and variance of the neighboring cell
                # TODO round these here in attempt to fix overflow error
                self.means[nx, ny] = (self.means[nx, ny] + self.means[x, y]) / 2
                self.variances[nx, ny] = (self.variances[nx, ny] + self.variances[x, y]) / 2
    
    # Converts from location units (m) to model coordinates
    def convert_location_to_model_coord(self, dim_ranges, location):
        if dim_ranges[0][0] >= 0:
            x_scaling = (location[0]) / (abs(dim_ranges[0][0]) + abs(dim_ranges[0][1]))
        else:
            x_scaling = (abs(dim_ranges[0][0])+location[0]) / (abs(dim_ranges[0][0]) + abs(dim_ranges[0][1]))
        if dim_ranges[1][0] >= 0:
            y_scaling = (location[1]) / (abs(dim_ranges[1][0]) + abs(dim_ranges[1][1]))
        else:
            y_scaling = (abs(dim_ranges[1][0])+location[1]) / (abs(dim_ranges[1][0]) + abs(dim_ranges[1][1]))
        
        x_model = int(self.cols * x_scaling)
        y_model = int(self.rows * y_scaling)
        
        return x_model, y_model
    
    def get_scaled_travel_vector(self, start_loc, end_loc, agent_velocity):  
        """
        Find the expected position vector traversed by an agent over one time step given start
        and end locations
        
        @param start_loc: starting location (environment units)
        @param end_loc: ending location (environment units)
        @param agent_velocity: travel velocity of agent (env units / time step)
        
        @returns vector change of agent position over one time step
        """     
        travel_vec = np.subtract(end_loc, start_loc) #end - start # x,y,(z) vector from current pos to target
        # print('Travel vec:', travel_vec)
        
        travel_vec_mag = np.linalg.norm(travel_vec) # get magnitude
        # print('Travel vec mag:', travel_vec_mag)
        
        if travel_vec_mag == 0.0:
            travel_unit_vec = (0.0, 0.0, 0.0)
        else:
            travel_unit_vec = tuple(val / travel_vec_mag for val in travel_vec) # calculate unit vector
        # print('Travel unit vec:', travel_unit_vec)
        
        scaled_travel_vec = tuple(val * agent_velocity for val in travel_unit_vec) # scale by agent velocity
        # print('Scaled travel vector: ', scaled_travel_vec)
        
        return scaled_travel_vec
    
    def check_location_within_threshold(self, current_loc, target_loc, threshold):
        arrived = True
        for c, e in zip(current_loc, target_loc):
                if (c < e - threshold or c > e + threshold):
                    arrived = False
        return arrived 
    
    # TODO estimate travel distance of directed edge between two locations
    def get_travel_cost_distribution(self, loc1, loc2, env_dim_ranges, agent_vel):
        """
        
        @returns (travel_vec, mean_sum, var_sum) where travel_vec is euclidean distance between
        locations, mean_sum and var_sum are normal distribution components of expected flows acting
        against agent along travel path
        """
        # Sum the means and sum the variances
        THRESHOLD = 1000 # TODO
        
        # print("Edge from", loc1, "to", loc2, ":")
        travel_vec = np.subtract(loc1, loc2)
        # 1) Get the position vector for agent displacement with one time step
        agent_travel_vec = np.array(self.get_scaled_travel_vector(loc1, loc2, agent_vel))
        
        # 2) Starting from location 1, move to location 2 and sum means & variances of flow vectors
        current_env_loc = np.array(loc1)
        current_model_loc = self.convert_location_to_model_coord(env_dim_ranges, current_env_loc)
        mean_flow_sum = self.means[current_model_loc[1], current_model_loc[0]] # y,x for row, col
        var_sum = self.variances[current_model_loc[1], current_model_loc[0]]
        # print("init mean_sum, var_sum:", mean_flow_sum, var_sum)
        
        end_reached = False
        timesteps = 1
        while not end_reached:
            current_env_loc = current_env_loc + agent_travel_vec
            # print("Agent current loc in env:", current_env_loc)
            current_model_loc = self.convert_location_to_model_coord(env_dim_ranges, current_env_loc)
            # print("Agent current loc in model:", current_model_loc)
            mean_flow_sum += self.means[current_model_loc[1], current_model_loc[0]] # y,x for row, col
            var_sum += self.variances[current_model_loc[1], current_model_loc[0]]
            
            end_reached = self.check_location_within_threshold(current_env_loc, loc2, THRESHOLD)
            # print(mean_flow_sum, var_sum)
            timesteps += 1
            
        # print("final mean_sum, var_sum:", mean_flow_sum, var_sum)
        mean_dist_sum = mean_flow_sum * timesteps
        # print("mean distance sum", mean_dist_sum)
        
        return (travel_vec, mean_dist_sum, var_sum)
        
    
                
    def visualize(self):
        X, Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        U = self.means[:, :, 0]
        V = self.means[:, :, 1]
        
        scale = max([abs(np.max(self.means)), abs(np.min(self.means))]) # 2
        
        plt.figure(figsize=(12, 6))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale)
        plt.xlim(-1, self.cols)
        plt.ylim(-1, self.rows)
        # plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Lattice with Vector Observations')
        plt.grid()
        plt.show()


class TaskNode:
    def __init__(self, id: int, location: tuple, work: int) -> None:
        self.id = id
        self.location = location
        self.work = work
        self.distances_to_tasks = {} # Normal distribution edge costs
        self.complete = False

    def set_distance_to_neighbor(self, task_id, distance_vec, mean, variance):
        """
        Define distance to task with euclidean distance, plus an additional distance
        component sampled from distribution resulting from flow field
        """
        self.distances_to_tasks[task_id] = (distance_vec, mean, variance)
        
    def sample_distance_to_neighbor(self, task_id):
        """
        @returns distance (m) from this task to input neighbor location
        """
        sampled_comps = np.random.normal(self.distances_to_tasks[task_id][1],
                                        self.distances_to_tasks[task_id][2]
                                        )
        distance_comp_sample = self.distances_to_tasks[task_id][0][:2] + sampled_comps # mod for 3D case
        
        return np.linalg.norm(distance_comp_sample)


if __name__ == "__main__":
    # Example usage:
    env_model = EnvironmentModel(5, 5)
    # print("Initial Means:\n", env_model.means)
    # print("Initial Variances:\n", env_model.variances)

    observation_location = (2, 2)
    observation_vector = (10, -6)
    env_model.apply_observation(observation_location, observation_vector)

    # print("\nMeans after observation:\n", env_model.means)
    # print("Variances after observation:\n", env_model.variances)

    # Visualize the grid with vectors
    env_model.visualize()



    
    # def generate_edge_cost_distributions(self, consider_flow=False):
    #     """
    #     For each task in env, generate a dictionary of travel time distributions to their neighbors
    #     """
    #     for task in self.tasks_list:
    #         task_loc = task.loc
    #         for neighbor in self.tasks_list:
    #             if task.id != neighbor.id:
    #                 neighbor_loc = neighbor.loc
    #                 if not consider_flow:
    #                     mean = self.get_travel_cost(task_loc, neighbor_loc)
    #                 else:
    #                     mean = self.get_travel_cost_with_flow(task_loc, neighbor_loc)
    #                 print('Cost from', task.id, ' to', neighbor.id, ':', mean)
    #                 # print('\nNo Flow Cost from', task.id, ' to', neighbor.id, ':', self.get_travel_cost(task_loc, neighbor_loc))
    #                 # print('Flow Cost from', task.id, ' to', neighbor.id, ':', self.get_travel_cost_with_flow(task_loc, neighbor_loc))
    #                 stddev = 0 # TODO will need to update this for local_env class (partial observ.)
    #                 key = str(neighbor.id)
    #                 task.travel_cost_to_neighbors_distros[key] = TravelDistro(mean, stddev)          
    
    # def get_scaled_travel_vector(self, start_loc, end_loc, velocity):       
    #     travel_vec = tuple(np.subtract(end_loc, start_loc)) #end - start # x,y,(z) vector from current pos to target
    #     # print('Travel vector: ', travel_vec)
        
    #     if not self.SLICE:
    #         travel_vec_mag = math.sqrt(travel_vec[0]**2 + travel_vec[1]**2 + travel_vec[2]**2)
    #     else:
    #         travel_vec_mag = math.sqrt(int(travel_vec[0])**2 + int(travel_vec[1])**2)
    #     # print('Travel vec mag:', travel_vec_mag)
        
    #     travel_unit_vec = tuple(val / travel_vec_mag for val in travel_vec) # calculate unit vector
    #     # print('Travel unit vec:', travel_unit_vec)
        
    #     scaled_travel_vec = tuple(val * velocity for val in travel_unit_vec)
    #     #robot_travel_vel * travel_vec_unit # contains travel dist components for this time step
    #     # print('Scaled travel vector: ', scaled_travel_vec)
        
    #     return scaled_travel_vec
        
    # def check_have_arrived(self, current_loc, end_loc):
    #     arrived = True
    #     for i in range(len(current_loc)):
    #             if (current_loc[i] < end_loc[i] - self.arrival_range or current_loc[i] > end_loc[i] + self.arrival_range):
    #                 arrived = False
    #     return arrived
                
    # def get_travel_cost_with_flow(self, start_loc, end_loc):
    #     """
    #     Process cost of traversing from loc1 to loc2 in environment, including impact of
    #     local flows.
    #     """
    #     robot_travel_vel = self.robots_list[0].velocity
    #     scaled_travel_vec = self.get_scaled_travel_vector(start_loc, end_loc, robot_travel_vel)
        
    #     # Step along position vector, sum cmd vel cost
    #     current_loc = start_loc
        
    #     # LOOP TODO - Will eventually insert uncertainty summation here for local env copies
    #     edge_cost = 0
    #     target_reached = False
    #     while not target_reached:
    #         # Using robot velocity, advance along path
    #         current_loc = tuple(current_loc[i] + scaled_travel_vec[i] for i in 
    #                         range(len(current_loc))) #current_loc + scaled_travel_vec
            
    #         local_flow = self.get_local_flow(current_loc)
            
    #         # Using robot velocity & local flow velocity, find velocity cmd and reduce energy level
    #         cmd_vel = tuple(scaled_travel_vec[i] - (local_flow[i]*robot_travel_vel) for i in range(len(scaled_travel_vec)))
    #             #scaled_travel_vec - local_flow # cmd vel is vel needed to reach waypoint given local flow
                
    #         if not self.SLICE:
    #             resultant_cmd_vel = round(math.sqrt(cmd_vel[0]**2 + cmd_vel[1]**2 + cmd_vel[2]**2), 2)
    #         else:
    #             resultant_cmd_vel = round(math.sqrt(cmd_vel[0]**2 + cmd_vel[1]**2), 2)
            
    #         edge_cost += self.robots_list[0].get_energy_burn_from_cmd_vel(resultant_cmd_vel) # TODO Energy cost math running on robot
    #         target_reached = self.check_have_arrived(current_loc, end_loc)
            
    #     return edge_cost
    
    # def get_travel_cost(self, start_loc, end_loc):
    #     """
    #     Process cost of traversing from loc1 to loc2 in environment, without impact of
    #     local flows.
    #     """
    #     robot_travel_vel = self.robots_list[0].velocity
    #     scaled_travel_vec = self.get_scaled_travel_vector(start_loc, end_loc, robot_travel_vel)

    #     # Step along position vector, sum cmd vel cost
    #     current_loc = start_loc
    #     cmd_vel = scaled_travel_vec
        
    #     edge_cost = 0
    #     target_reached = False
    #     while not target_reached:
    #         # Using robot velocity, advance along path
    #         current_loc = tuple(current_loc[i] + scaled_travel_vec[i] for i in 
    #                         range(len(current_loc))) #current_loc + scaled_travel_vec
                
    #         if not self.SLICE:
    #             resultant_cmd_vel = round(math.sqrt(cmd_vel[0]**2 + cmd_vel[1]**2 + cmd_vel[2]**2), 2)
    #         else:
    #             resultant_cmd_vel = round(math.sqrt(cmd_vel[0]**2 + cmd_vel[1]**2), 2)
            
    #         edge_cost += self.robots_list[0].get_energy_burn_from_cmd_vel(resultant_cmd_vel) # TODO Energy cost math running on robot
            
    #         target_reached = self.check_have_arrived(current_loc, end_loc)
            
    #     return edge_cost
        