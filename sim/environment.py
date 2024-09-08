import math
import os
import random

import numpy as np
import xarray as xr
import yaml


class Environment:

    def __init__(
        self,
        topography_file: str,
        flow_data_files: list[str],
        env_dimensions: tuple,
        agent_loc_dict: dict,
        base_loc,
        thin_params: tuple = (1, 1, 1),
        time_step_size: int = 1,
    ):
        """
        @param topography: xarray dataset
        @param flow_data_list: list of hourly environment state xarray datasets
        @param env_dimenions: tuple ((x1,x2),(y1,y2),(z1,z2)) environment dimensions to sample from dataset.
        z = 1 for 2D environment.
        @param thin_params: tuple (x,y,z) params for thinning dataset.
        z = 1 for 2D environment.
        @param robots_list: list of all robots in environment (possibly includes mothership)
        @param time_step_size: int minutes of one time step for incrementing environment
        """
        self.env_dimensions = env_dimensions
        # divide hour by time step size to get number of steps
        self.time_steps = 60 / time_step_size

        self.FLOW_MULTIPLIER = 1000  # TODO FLOW MAGNITUDE MODIFIER

        self.agent_loc_dict = agent_loc_dict
        self.base_loc = base_loc

        # Process datasets (thin dataset & crop to dimensions)
        x_thinning = thin_params[0]
        y_thinning = thin_params[1]
        z_thinning = thin_params[2]

        self.SLICE = False
        if env_dimensions[2][0] == env_dimensions[2][1]:
            self.SLICE = True

        # Crop & thin topography data
        topography = xr.open_dataset(topography_file)
        x_coords = topography["xx"][
            env_dimensions[0][0]: env_dimensions[0][1]: x_thinning
        ]
        y_coords = topography["yy"][
            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
        ]
        z_coords = topography["zz"][
            env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
        ]
        self.cropped_coords = {"x": x_coords, "y": y_coords, "z": z_coords}

        # Crop & thin flow field data
        self.processed_flow_data = []
        for filename in flow_data_files:
            # (z: 68, y: 200, x: 294)
            # zonal_vel = data['uu'] # zonal velocity (along the slope, in the x-direction, positive eastward) m/s
            # merid_vel = data['vv'] # meridional velocity (across the slope, in the y-direction, positive northward) m/s
            # vert_vel = data['ww'] # vertical velocity (positive upward) m/s
            data = xr.open_dataset(filename)
            if not self.SLICE:
                u_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in plane[
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                    for plane in data["uu"][
                        env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                    ]
                ]
                v_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in plane[
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                    for plane in data["vv"][
                        env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                    ]
                ]
                w_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in plane[
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                    for plane in data["ww"][
                        env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                    ]
                ]
            else:
                u_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in data["uu"][env_dimensions[2][0]][
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                ]
                v_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in data["vv"][env_dimensions[2][0]][
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                ]
                w_vecs = [
                    [
                        row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                        for row in data["ww"][env_dimensions[2][0]][
                            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                        ]
                    ]
                ]

            processed_data = {"u": u_vecs, "v": v_vecs, "w": w_vecs}
            self.processed_flow_data.append(processed_data)

        # Set current flow data reference frame & current flow state
        # index of first file we are interpolating flow from (idx+1 for next file)
        self.flow_data_idx = 0
        # current environment frame
        self.current_flow_state = self.processed_flow_data[self.flow_data_idx]

        # Create & load flow vector modifiers (for transitioning from one dataset to the next over time_steps - 1hr transition)
        # Only if we loaded multiple flow files for dynamic environmnent
        if len(self.processed_flow_data) > 1:
            self.flow_data_modifiers = self._update_flow_modifiers(
                self.processed_flow_data[self.flow_data_idx],
                self.processed_flow_data[self.flow_data_idx + 1],
            )

    def reset(self):

        for a_id in self.agent_loc_dict.keys():
            self.agent_loc_dict[a_id] = self.base_loc

    def _update_flow_modifiers(self, flow_hour1, flow_hour2):
        """
        Process modifiers for interpolating between two datasets. Modifiers are added to current
        flow state at each time step

        @param flow_hour1: first hour ocean currents dataset
        @param flow_hour2: second hour ocean currents dataset

        @returns list of flow modifiers to be applied at each env step
        """
        # Find differences
        u_diff = np.subtract(
            np.array(flow_hour2["u"]), np.array(flow_hour1["u"]))
        v_diff = np.subtract(
            np.array(flow_hour2["v"]), np.array(flow_hour1["v"]))
        w_diff = np.subtract(
            np.array(flow_hour2["w"]), np.array(flow_hour1["w"]))

        # Find gradients
        u_step_mod = u_diff / self.time_steps
        v_step_mod = v_diff / self.time_steps
        w_step_mod = w_diff / self.time_steps

        modifiers = {"u": u_step_mod, "v": v_step_mod, "w": w_step_mod}

        return modifiers

    def _check_is_loc_in_env(self, dims_ranges, loc):
        """
        Returns true if loc coordinates are within dims_ranges. False otherwise.

        @param dims_ranges:
        @param loc: (x,y) location to evaluate
        """
        x_check = loc[0] >= dims_ranges[0][0] and loc[0] <= dims_ranges[0][1]
        y_check = loc[1] >= dims_ranges[1][0] and loc[1] <= dims_ranges[1][1]
        z_check = True
        if not self.SLICE:
            z_check = loc[2] >= dims_ranges[2][1] and loc[2] <= dims_ranges[2][0]
        return x_check and y_check and z_check

    def get_local_flow(self, loc):
        """
        Get the local flow vector at a given location

        @param loc: Coordinate location from which to extract flows

        @returns list of [x,y,z] flow components
        """

        # Get from loc (km range) to nearest pos (list idxs)
        x_coords = self.cropped_coords["x"]
        y_coords = self.cropped_coords["y"]
        z_coords = self.cropped_coords["z"]

        local_x = np.argmin(np.abs(x_coords.values - loc[0]))
        local_y = np.argmin(np.abs(y_coords.values - loc[1]))

        # Get flow vector closest to this list idx
        if not self.SLICE:
            local_z = np.argmin(np.abs(z_coords.values - loc[2]))
            # z,y,x idx
            local_flow_x = self.current_flow_state["u"][local_z][local_y][
                local_x
            ].values
            local_flow_y = self.current_flow_state["v"][local_z][local_y][
                local_x
            ].values
            local_flow_z = self.current_flow_state["w"][local_z][local_y][
                local_x
            ].values
            local_flow = [local_flow_x, local_flow_y, local_flow_z]
        else:
            local_flow_x = self.current_flow_state["u"][0][local_y][local_x].values
            local_flow_y = self.current_flow_state["v"][0][local_y][local_x].values
            # print('Flow vect at:', local_x, local_y, ' is', local_flow_x, local_flow_y)
            local_flow = [local_flow_x, local_flow_y]

        modified_flows = np.multiply(self.FLOW_MULTIPLIER, local_flow)

        return modified_flows

    def get_dim_ranges(self):
        x_min = min(self.cropped_coords["x"].values)
        x_max = max(self.cropped_coords["x"].values)
        y_min = min(self.cropped_coords["y"].values)
        y_max = max(self.cropped_coords["y"].values)
        if not self.SLICE:
            z_min = min(self.cropped_coords["z"].values)
            z_max = max(self.cropped_coords["z"].values)
            return ((x_min, x_max), (y_min, y_max), (z_min, z_max))  # meters
        else:
            return ((x_min, x_max), (y_min, y_max), (0, 0))

    def setup_random_base_loc(self):
        ranges = self.get_dim_ranges()
        base_x = np.random.randint(ranges[0][0], ranges[0][1])
        base_y = np.random.randint(ranges[1][0], ranges[1][1])
        if not self.SLICE:
            base_z = np.random.randint(ranges[2][0], range[2][1])
        else:
            base_z = 0
        base_loc = [base_x, base_y, base_z]
        self.base_loc = base_loc
        for a in self.agent_loc_dict.keys():
            loc = base_loc[:]
            loc[0] = base_loc[0] + random.randint(10, 1000)
            loc[1] = base_loc[1] + random.randint(10, 1000)
            loc[2] = base_loc[2] + random.randint(10, 1000)

            print("setting", a, "start loc to", loc)
            self.agent_loc_dict[a] = loc

        return base_loc

    def step(self, agent_list):
        """
        Advance global actual environment by one time step. Updates robot locations & energy levels. Updates
        flow field.
        """
        # Advance actual positions of traveling agents, update energy levels
        # Otherwise, update energy level of working agents as required to maintain position

        for a in agent_list:
            scaled_travel_vec = a.position_mod_vector
            # target_loc = a.get_target_location()
            agent_loc = self.agent_loc_dict[a.id]

            # If agent is idle, hold location and power
            if a.action[0] == a.IDLE:
                pass
            elif a.action[0] == a.TRAVELING:
                # If agent is traveling, update location and energy

                # Using agent velocity, move closer to destination
                new_loc = tuple(
                    agent_loc[i] + scaled_travel_vec[i] for i in range(len(agent_loc))
                )

                self.agent_loc_dict[a.id] = new_loc  # Update agent location

                # Using agent velocity & local flow velocity, find velocity cmd and reduce energy level
                cmd_vel = a.get_command_velocity()
                # print('Agent', a.id, 'traveling to', target_loc)
                a.reduce_energy(cmd_vel)

            elif a.action[0] == a.WORKING:
                # Else if agent is working, update energy only
                # self.agent_loc_dict[a.id] = target_loc
                cmd_vel = a.get_command_velocity()
                # print('Agent', a.id, 'working on', a.action[1])
                a.reduce_energy(cmd_vel)

        # Nudge flow field toward next data file state
        # if self.current_step < self.time_steps:
        #     self.current_flow_state['u'] += self.flow_data_modifiers['u']
        #     self.current_flow_state['v'] += self.flow_data_modifiers['v']
        #     self.current_flow_state['w'] += self.flow_data_modifiers['w']

        #     self.current_step += 1 # TODO - verify time steps are correct
        # # If next data file state reached, do updates
        # else:
        #     self.flow_data_idx += 1
        #     self.current_flow_state = self.processed_flow_data[self.flow_data_idx]
        #     self.flow_data_modifiers = self.update_flow_modifiers(self.processed_flow_data[self.flow_data_idx],
        #                                                           self.processed_flow_data[self.flow_data_idx+1])


# TODO Update to take in a folder of tidal files for dynamic env
def make_environment_from_config(
    config_filepath, topo_filepath: str, tidal_folderpath: str
) -> Environment:
    """
    Create an environmnet from parameters

    @param topo_fp: filepath to environment topogrophy xarray file
    @param tidal_fp: filepath to environment tides xarray file
    @param dims: dimensions of environment

    @returns: an Environment
    """
    # Load the environment configuration
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

        dims = (
            tuple(config["xCoordRange"]),
            tuple(config["yCoordRange"]),
            tuple(config["zCoordRange"]),
        )

        thinning = (config["xThin"], config["yThin"], config["zThin"])

        # Load in agents at starting locations
        agent_loc_dict = {}
        for i in range(config["num_robots"]):
            loc = list(config["base_loc"])
            # Add coords plus some noise NOTE - maybe update noise input
            loc[0] = loc[0] + random.randint(10, 1000)
            loc[1] = loc[1] + random.randint(10, 1000)
            loc[2] = loc[2] + random.randint(10, 1000)

            agent_loc_dict[i] = loc

        # Load in support robots - each group has own set of support robots
        for g_id in range(config["num_robots"]):
            for i in range(config["support_robots"]):

                loc = list(config["base_loc"])
                loc[0] = loc[0] + random.randint(10, 500)
                loc[1] = loc[1] + random.randint(10, 500)
                loc[2] = loc[2] + random.randint(10, 500)

                id = (g_id, i)

                agent_loc_dict[id] = loc

        agent_loc_dict[config["m_id"]] = list(config["base_loc"])

    # TODO Process folder of tidal filepaths into list here once we start using time-varying environment
    tidal_fp = random.choice(os.listdir(tidal_folderpath))
    tidal_fp = os.path.join(tidal_folderpath, tidal_fp)
    print("Selected flow data file", tidal_fp)
    tidal_fps = [tidal_fp]

    return Environment(
        topo_filepath,
        tidal_fps,
        dims,
        agent_loc_dict,
        loc,
        thinning,
    )
