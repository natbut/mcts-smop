import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_mpc_results_from_log(log_fp, title="MPC Results"):
    """
    Plots the results of MPC from given log file.

    Args:
        log_fp (str): filepath to log (csv)
        title (str): Title of the plot.
    """
    print("Plotting results...")

    basename = os.path.basename(log_fp)  # parse log file name from log_fp
    name = os.path.splitext(basename)[0]

    df = pd.read_csv(log_fp)

    time = df["timestamp"]
    joint_positions = df[[
        col for col in df.columns if "position" in col]].values
    joint_velocities = df[[
        col for col in df.columns if "velocity" in col]].values
    control_inputs = df[[
        col for col in df.columns if "control_input" in col]].values
    end_eff = df[[col for col in df.columns if "end_effector" in col]].values

    plot_mpc_results(
        time, joint_positions, joint_velocities, control_inputs, end_eff, title, name
    )


def plot_mpc_results(
    time,
    joint_positions,
    joint_velocities,
    control_inputs,
    end_eff,
    title="MPC Results",
    figname="fig",
):
    """
    Plots the results of MPC from given joint positions, velocities, and control inputs.

    Args:
        time (np.ndarray): Time array.
        joint_positions (np.ndarray): Joint positions array with shape (n_steps, n_joints).
        joint_velocities (np.ndarray): Joint velocities array with shape (n_steps, n_joints).
        control_inputs (np.ndarray): Control inputs (torques) array with shape (n_steps, n_joints).
        end_eff (np.ndarray):
        title (str): Title of the plot.
    """

    n_joints = joint_positions.shape[1]

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    # Plot joint positions
    axes[0].set_title("Joint Positions")
    for i in range(n_joints):
        axes[0].plot(time, joint_positions[:, i], label=f"Joint {i+1}")
    axes[0].set_ylabel("Position (rad)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot joint velocities
    axes[1].set_title("Joint Velocities")
    for i in range(n_joints):
        axes[1].plot(time, joint_velocities[:, i], label=f"Joint {i+1}")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot control inputs
    axes[2].set_title("Control Inputs")
    for i in range(n_joints):
        axes[2].plot(time, control_inputs[:, i], label=f"Joint {i+1}")
    axes[2].set_ylabel("Torque? (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(title)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{figname}.png", bbox_inches="tight")

    # Plot end effector trajectory
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # axes[0:2, 1].set_title("End Effector")
    ax.scatter(xs=end_eff[0], ys=end_eff[1], zs=end_eff[2])
    # axes[0:2, 1].set_ylabel("Y Pos")
    # axes[0:2, 1].set_xlabel("X Pos")
    # axes[0:2, 1].set_zlabel("Z Pos")
    # axes[0:2, 1].grid(True)

    fig.suptitle(title)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{figname}_endeff.png", bbox_inches="tight")
