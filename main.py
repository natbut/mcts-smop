from argparse import ArgumentParser, Namespace
import time

from sim.comms_manager import CommsManager
from control.agent import generate_agents_from_config
from sim.environment import make_environment_from_config
from utils.visualizer import set_up_visualizer_from_config


def get_args() -> Namespace:
    """Parse the script arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        help="Full path to the problem instance configuration file",
    )
    parser.add_argument(
        "topo_file",
        type=str,
        help="Full path to the topography data file",
    )
    parser.add_argument(
        "tide_folder",
        type=str,
        help="Full path to the tidal data folder",
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    
    print('Initialize')
    
    # Create environment
    env = make_environment_from_config(args.config, args.topo_file, args.tide_folder)
    
    # Create agents
    agent_list = generate_agents_from_config(args.config)
    
    # Create comms framework (TBD - may be ROS)
    comms_mgr = CommsManager(env, agent_list)
    
    # Set up initial state
    for a in agent_list:
        a.sense_location_from_env(env)
        a.set_up_dim_ranges(env)
        a.schedule = [0,1,-1] # TODO for testing (delete)
    agent_list[0].schedule = [2,3,-1]
    
    # Run simulation (planning, etc.)
    # agent_list[0].send_message(comms_mgr, agent_list[1].id, "Hello!")
    i = 0
    viz = set_up_visualizer_from_config(env, args.config)
    while True:
        print("TIME STEP:", i)
        
        # Update agents (scheduling & control happens here)
        for a in agent_list: # TODO consider a.step() for all this
            # Consider condensing this all into an a.step()
            # get updated observations
            a.sense_location_from_env(env)
            a.sense_flow_from_env(env)
            
            # If new flow found (EVENT), do rescheduling
            if a.event:
                a.apply_observations_to_model()
                # a.optimize_schedule() # TODO
            
                # TODO Send necessary comms
            
            
            # Update current action
            a.action_update()
        
        # Apply actions to get new environment state (update agent positions & observations, comms update)
        # TODO confirm that flow is correctly being applied to reduce energy (seems like something may be inverted)
        env.step(agent_list)
        
        # Update message passing, also update comms graph
        comms_mgr.step()
        
        viz.display_env(agent_list, static=False) # TODO add dynamic tour lines to graph
        
        i+=1
        time.sleep(0.1)