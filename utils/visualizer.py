import random
import time

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from control.agent import Agent
from sim.environment import Environment


# Class for visualizing environmnent with tasks and agents
class Visualizer:

    def __init__(self, env: Environment, task_dict):
        self.env = env
        self.task_dict = task_dict

        env_dims = self.env.get_dim_ranges()
        x_size = abs(env_dims[0][1] - env_dims[0][0])
        y_size = abs(env_dims[1][1] - env_dims[1][0])
        scaling = x_size/y_size
        y_dim = 6
        x_dim = int(y_dim * scaling)

        plt.ion()
        # TODO - scale figure size by env subsample
        self.fig, self.ax = plt.subplots(figsize=(x_dim, y_dim))
        # self.fig.tight_layout()

    def _prepare_plot(self, static=True):
        x_coords = self.env.cropped_coords['x']
        y_coords = self.env.cropped_coords['y']
        z_coords = self.env.cropped_coords['z']

        self.ax.cla()

        # self.fig = plt.figure(figsize=(12, 6))
        # if self.env.SLICE:
        #     self.ax = self.fig.add_subplot()
        # else:
        #     self.ax = self.fig.add_subplot(projection='3d') #3d

        # Flatten the coordinates
        if self.env.SLICE:
            y, x = np.meshgrid(y_coords, x_coords, indexing='ij')
            z = None
        else:
            z, y, x = np.meshgrid(z_coords,
                                  y_coords,
                                  x_coords,
                                  indexing='ij')  # 3d

        return x, y, z

    def _add_flow_to_plot(self, x, y, z):
        if self.env.SLICE:
            self.ax.quiver(x, y,
                           self.env.current_flow_state['u'][0],
                           self.env.current_flow_state['v'][0],
                           pivot='mid',
                           headwidth=2)
        else:
            self.ax.quiver(x, y, z,
                           self.current_flow_state['u'],
                           self.current_flow_state['v'],
                           self.current_flow_state['w'],
                           length=50,
                           normalize=True)  # 3d

    def _add_tasks_to_plot(self):
        # Locate tasks
        task_x = []
        task_y = []
        task_z = []
        for task in self.task_dict.values():
            loc = task.location
            task_x.append(loc[0])
            task_y.append(loc[1])
            if not self.env.SLICE:
                task_z.append(loc[2])

        # Plot the tasks
        if self.env.SLICE:
            self.ax.scatter(task_x, task_y, c='r', marker='o')
        else:
            self.ax.scatter(task_x, task_y, task_z, c='r', marker='o')

        for i, id in enumerate(self.task_dict.keys()):
            if not self.env.SLICE:
                self.ax.annotate(id, (task_x[i], task_y[i], task_z[i]))
            else:
                self.ax.annotate(id, (task_x[i], task_y[i]))

    def _add_robots_to_plot(self):
        # Locate robots
        robot_locs = [loc for loc in self.env.agent_loc_dict.values()]

        rob_x = [loc[0] for loc in robot_locs]
        rob_y = [loc[1] for loc in robot_locs]
        rob_z = [loc[2] for loc in robot_locs]

        # Plot the robots
        if self.env.SLICE:
            self.ax.scatter(rob_x, rob_y, c='b', marker='o')
        else:
            self.ax.scatter(rob_x, rob_y, rob_z, c='b', marker='o')

    def _show_plot(self, static=True):

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        if not self.env.SLICE:
            self.ax.set_zlabel('Z')  # 3d
        if static:
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # time.sleep(0.1)

    def display_env(self, agent_list: list[Agent], static=True):
        # Prepare plot
        x, y, z = self._prepare_plot(static)

        # Plot the flow
        # print('Plotting flow ...')
        self._add_flow_to_plot(x, y, z)

        # Plot the tasks
        self._add_tasks_to_plot()

        # Plot the robots
        # print('Plotting robots ...')
        self._add_robots_to_plot()

        # Plot the tours
        self._add_tour_lines_to_plot(agent_list)

        self._show_plot(static)
        # print('Plot Complete')

    def close_viz(self):
        plt.close()

    def _get_cmap(self, n, name='viridis', vmin=0, vmax=100):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(name)
        color = cmap(norm(n))
        return mcolors.to_hex(color)

    def _add_tour_lines_to_plot(self, agent_list: list[Agent]):
        """

        """
        for a in agent_list:
            cmap = self._get_cmap(n=a.id, vmax=len(agent_list))
            start_loc = a.location
            if len(a.schedule) == 0:
                end_loc = a.base_loc
            else:
                # if a.schedule[0] != a.sim_data["rob_task"]:
                if a.action[1] != "Init":
                    end_loc = self.task_dict[a.action[1]].location
                # else:
                #     end_loc = a.location
            self.ax.arrow(start_loc[0], start_loc[1], end_loc[0] - start_loc[0], end_loc[1] - start_loc[1],
                          shape='full', width=400, length_includes_head=True, ec=cmap, fc=cmap)
            if len(a.schedule) > 0:
                start_loc = end_loc
                for task_id in a.schedule:
                    # if task_id != a.sim_data["rob_task"]:
                    end_loc = self.task_dict[task_id].location
                    # else:
                    #     end_loc = a.location
                    # Plot the edge as directed arrow
                    self.ax.arrow(start_loc[0], start_loc[1], end_loc[0] - start_loc[0], end_loc[1] - start_loc[1],
                                  shape='full', width=400, length_includes_head=True, ec=cmap, fc=cmap)
                    start_loc = end_loc

    # def display_env_with_tasks_and_solution(self, tour):
    #     if not self.PLOTTING:
    #         # plt.ion()

    #         # Prepare plot
    #         self.fig, self.ax, self.x, self.y, self.z = self._prepare_plot()
    #         self.PLOTTING = True

    #     fig, ax, x, y, z = self.fig, self.ax, self.x, self.y, self.z

    #     ax.clear()

    #     # Plot the flow
    #     self._add_flow_to_plot(ax, x, y, z)

    #     # Plot the tasks
    #     self._add_tasks_to_plot(ax)

    #     # Plot the tour lines
    #     self._add_tour_lines_to_plot(ax, tour)

    #     self._show_plot(fig, ax, static=True)


def set_up_visualizer(
    env: Environment,
    task_dict
) -> Visualizer:

    # Load up task list locations for visualizer
    # with open(config_filepath, "r") as f:
    #     config = yaml.safe_load(f)

    #     # Make task dict to tie IDs to locations
    #     task_dict = {}
    #     for task in config["tasks"]:
    #         for key in task.keys():
    #             task_dict[key] = task[key]["loc"]
    #     task_dict[-1] = config["start_loc"]

    return Visualizer(env,
                      task_dict
                      )
