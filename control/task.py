import numpy as np
import yaml

from utils.helpers import sample_from_range


class Task:
    # NOTE Potentially make seperate function for representing environment model as graph with stochastic edges
    def __init__(self,
                 id: int,
                 location: tuple,
                 work: int,
                 reward: int
                 ) -> None:
        self.id = id
        self.location = location
        self.work = work
        self.reward = reward
        self.distances_to_tasks = {}  # Normal distribution edge costs
        self.complete = False

    def calc_dist_to_task(self, task_id, location):
        diff_vec = self.location - location
        vec_mag = np.linalg.norm(diff_vec)
        if vec_mag == 0.0:
            vec_mag = 0.05  # Set to number close to 0.0 to prevent error
        self.distances_to_tasks[task_id] = vec_mag
        print("Distance", (self.id, task_id), ":", vec_mag)

    # def set_distance_to_neighbor(self, task_id, distance_vec, mean, variance):
    #     """
    #     Define distance to task with euclidean distance, plus an additional distance
    #     component sampled from distribution resulting from flow field
    #     """
    #     self.distances_to_tasks[task_id] = (distance_vec, mean, variance)

    # def sample_distance_to_neighbor(self, task_id):
    #     """
    #     @returns distance (m) from this task to input neighbor location
    #     """
    #     sampled_comps = np.random.normal(self.distances_to_tasks[task_id][1],
    #                                      self.distances_to_tasks[task_id][2]
    #                                      )
    #     # mod for 3D case
    #     distance_comp_sample = self.distances_to_tasks[task_id][0][:2] + sampled_comps

    #     return np.linalg.norm(distance_comp_sample)


def add_tasks_to_dict(problem_config_fp, env, task_dict, num_tasks, high_rew=False):
    with open(problem_config_fp, "r") as f:
        config = yaml.safe_load(f)

        for i in range(num_tasks):
            dim_ranges = env.get_dim_ranges()
            x = sample_from_range(dim_ranges[0][0],
                                  dim_ranges[0][1])
            y = sample_from_range(dim_ranges[1][0],
                                  dim_ranges[1][1])
            z = sample_from_range(dim_ranges[2][0],
                                  dim_ranges[2][1])

            work = sample_from_range(config["work_range"][0],
                                     config["work_range"][1])
            if high_rew:
                reward = config["reward_range"][1]
            else:
                reward = sample_from_range(config["reward_range"][0],
                                           config["reward_range"][1])

            id = len(task_dict) - 2

            task_dict["v"+str(id)] = Task("v"+str(id),
                                          np.array([x, y, z]), work, reward)
    return task_dict


def generate_tasks_from_config(problem_config_fp, env, rand_base=None):

    with open(problem_config_fp, "r") as f:
        config = yaml.safe_load(f)

        if rand_base:
            base_loc = rand_base
        else:
            base_loc = np.array(config["base_loc"])

        task_dict = {}
        task_dict[config["start"]] = Task(config["start"], base_loc, 0, 1)
        task_dict[config["end"]] = Task(config["end"], base_loc, 0, 1)

        for i in range(config["problem_size"]):

            dim_ranges = env.get_dim_ranges()
            x = sample_from_range(dim_ranges[0][0],
                                  dim_ranges[0][1])
            y = sample_from_range(dim_ranges[1][0],
                                  dim_ranges[1][1])
            z = sample_from_range(dim_ranges[2][0],
                                  dim_ranges[2][1])

            work = sample_from_range(config["work_range"][0],
                                     config["work_range"][1])
            reward = sample_from_range(config["reward_range"][0],
                                       config["reward_range"][1])

            task_dict["v"+str(i)] = Task("v"+str(i),
                                         np.array([x, y, z]), work, reward)

    return task_dict
