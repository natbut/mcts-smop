
from copy import deepcopy

import numpy as np

from control.agent import Agent, load_data_from_config
from sim.comms_manager import CommsManager, Message


class Support(Agent):
    IDLE = 0
    TRAVELING = 1
    WORKING = 2

    def __init__(self,
                 id: int,
                 group_id,
                 solver_params: dict,
                 sim_data: dict,
                 merger_params: dict
                 ) -> None:

        super().__init__(id, solver_params, sim_data, merger_params)
        self.type = self.SUPPORT

        self.mothership = None

        self.group_assn = group_id
        self.group_loc = []
        self.mother_loc = []

        self.action = [self.TRAVELING, None]

    def _get_group_loc(self):
        for g in self.group_list:
            if g.id == self.group_assn:
                return np.array(g.location)

    def _get_mother_loc(self):
        if self.mothership != None:
            return np.array(self.mothership.location)

    def action_update(self):
        # Compute vector between group_assn and mothership
        self.group_loc = self._get_group_loc()
        self.mother_loc = self._get_mother_loc()

        # print("Mothership loc:", self.mother_loc,
        #   " Group loc:", self.group_loc)

        pos_vec = np.subtract(self.group_loc, self.mother_loc)

        # print("Pos vec:", pos_vec)

        # pos_vec_mag = np.linalg.norm(pos_vec)  # get magnitude
        # print('Travel vec mag:', travel_vec_mag)

        # print("Pos vec mag:", pos_vec_mag)

        # calculate unit vector
        pos_unit_vec = np.array(
            [val / self.sim_data["support_robots"] for val in pos_vec])

        # Update target location to be self.id[1] * unit vector pos
        # print("Pos unit vec:", pos_unit_vec)
        target_dest = self.mother_loc + (pos_unit_vec * (self.id[1] + 0.5))

        # print("Target dest:", target_dest)

        self.update_position_mod_vector(target_dest)


def generate_supports_with_data(solver_params, sim_data, merger_params) -> list[Support]:
    pssngr_list = []
    for g_id in range(solver_params["num_robots"]):
        for j in range(sim_data["support_robots"]):
            id = (g_id, j)
            p = Support(id,
                        g_id,
                        solver_params=deepcopy(solver_params),
                        sim_data=deepcopy(sim_data),
                        merger_params=deepcopy(merger_params)
                        )
            # for v in sim_data["graph"].vertices:
            #     p.load_task(v, None, sim_data["graph"].works[v])
            pssngr_list.append(p)
    return pssngr_list


def generate_supports_from_config(solver_config_fp,
                                  problem_config_fp,
                                  ) -> list[Support]:

    sim_data, dec_mcts_data, _, merger_data = load_data_from_config(
        solver_config_fp, problem_config_fp)

    return generate_supports_with_data(dec_mcts_data, sim_data, merger_data)
