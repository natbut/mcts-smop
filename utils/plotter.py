import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results_from_log(log_fp):
    """
    Plots the results of simulation from given log file.

    Args:
        log_fp (str): filepath to log (csv)
        title (str): Title of the plot.
    """
    print("Plotting results...")

    basename = os.path.basename(log_fp)  # parse log file name from log_fp
    name = os.path.splitext(basename)[0]

    df = pd.read_csv(log_fp)

    trial = df["trial"]
    test = df["test"]

    frontEndOnly = df[[
        col for col in df.columns if "frontEndOnly" in col]].values
    distrOnly = df[[
        col for col in df.columns if "distrOnly" in col]].values
    twoStep = df[[
        col for col in df.columns if "twoStep" in col]].values
    distHybrid = df[[
        col for col in df.columns if "dist_hybrid" in col]].values
    fullHybrid = df[[
        col for col in df.columns if "full_hybrid" in col]].values

    title = "Results for test " + name

    plot_results(trial,
                 test,
                 frontEndOnly,
                 distrOnly,
                 twoStep,
                 distHybrid,
                 fullHybrid,
                 title,
                 name
                 )


def plot_results(trial,
                 test,
                 frontEnd_results,
                 distrOnly_results,
                 twoPart_results,
                 dist_hybrid_results,
                 full_hybrid_results,
                 title="Results",
                 figname="Fig"
                 ):

    # Mean Rewards
    frontEnd_rew = round(np.mean([res[0] for res in frontEnd_results]), 2)
    distrOnly_rew = round(np.mean([res[0] for res in distrOnly_results]), 2)
    twoPart_rew = round(np.mean([res[0] for res in twoPart_results]), 2)
    dist_hybrid_rew = round(
        np.mean([res[0] for res in dist_hybrid_results]), 2)
    full_hybrid_rew = round(
        np.mean([res[0] for res in full_hybrid_results]), 2)

    # Mean potential rewards
    frontEnd_pot = round(np.mean([res[1] for res in frontEnd_results]), 2)
    distrOnly_pot = round(np.mean([res[1] for res in distrOnly_results]), 2)
    twoPart_pot = round(np.mean([res[1] for res in twoPart_results]), 2)
    dist_hybrid_pot = round(np.mean([res[1]
                            for res in dist_hybrid_results]), 2)
    full_hybrid_pot = round(np.mean([res[1]
                            for res in full_hybrid_results]), 2)
    # Mean robots lost
    frontEnd_fails = round(np.mean([res[2] for res in frontEnd_results]), 2)
    distrOnly_fails = round(np.mean([res[2] for res in distrOnly_results]), 2)
    twoPart_fails = round(np.mean([res[2] for res in twoPart_results]), 2)
    dist_hybrid_fails = round(
        np.mean([res[2] for res in dist_hybrid_results]), 2)
    full_hybrid_fails = round(
        np.mean([res[2] for res in full_hybrid_results]), 2)

    # StdErr
    frontEnd_rew_se = np.std([res[0] for res in frontEnd_results]) / \
        np.sqrt(len(frontEnd_results))
    distrOnly_rew_se = np.std([res[0] for res in distrOnly_results]) / \
        np.sqrt(len(distrOnly_results))
    twoPart_rew_se = np.std([res[0] for res in twoPart_results]) / \
        np.sqrt(len(twoPart_results))
    dist_hybrid_rew_se = np.std([res[0] for res in dist_hybrid_results]) / \
        np.sqrt(len(full_hybrid_results))
    full_hybrid_rew_se = np.std([res[0] for res in full_hybrid_results]) / \
        np.sqrt(len(full_hybrid_results))
    # Potentials
    frontEnd_pot_se = np.std([res[1] for res in frontEnd_results]) / \
        np.sqrt(len(frontEnd_results))
    distrOnly_pot_se = np.std([res[1] for res in distrOnly_results]) / \
        np.sqrt(len(distrOnly_results))
    twoPart_pot_se = np.std([res[1] for res in twoPart_results]) / \
        np.sqrt(len(twoPart_results))
    dist_hybrid_pot_se = np.std([res[1] for res in dist_hybrid_results]) / \
        np.sqrt(len(dist_hybrid_results))
    full_hybrid_pot_se = np.std([res[1] for res in full_hybrid_results]) / \
        np.sqrt(len(full_hybrid_results))

    avg_rew = [frontEnd_rew, distrOnly_rew,
               twoPart_rew, dist_hybrid_rew, full_hybrid_rew]

    avg_pot = [frontEnd_pot, distrOnly_pot,
               twoPart_pot, dist_hybrid_rew, full_hybrid_pot]

    error_rew = [frontEnd_rew_se, distrOnly_rew_se,
                 twoPart_rew_se, dist_hybrid_rew_se, full_hybrid_rew_se]

    error_pot = [frontEnd_pot_se, distrOnly_pot_se,
                 twoPart_pot_se, dist_hybrid_pot_se, full_hybrid_pot_se]

    rew_content = {
        "Tasks Visited": (avg_pot, error_pot),
        "Tasks Returned": (avg_rew, error_rew),
    }

    labels = ["Front-End Only", "Distr. Only", "Front End\n+ Dist Replan",
              "Hybrid Replan", "Front End\n+ Hybrid Replan"]

    # Plot results
    fig, ax = plt.subplots()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    x = np.arange(len(labels))
    width = 0.3
    multiplier = 0
    start = x
    for attribute, measurements in rew_content.items():
        offset = width * multiplier
        x_temp = start + offset
        rects = ax.bar(
            x_temp, measurements[0], width, yerr=measurements[1],  label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # ax.bar(x, avg_tasks, yerr=error_bars, capsize=5,
    #        color=['blue', 'lightblue', 'red', 'green', 'darkviolet'])

    ax.set_xticks(x+width/2, labels)
    ax.set_ylabel('Percent Task Completion')
    ax.set_title(title)
    ax.set_ybound(0.0, 1.0)
    if full_hybrid_rew < 0.5:
        ax.legend(loc='upper right', ncols=1)
    else:
        ax.legend(loc='lower right', ncols=1)

    fig.savefig(f"{figname}.png")

    print("Done")

    plt.show()


def grouped_plot(fail_log_fps, task_log_fps, subplot_titles, plot_title=None, group_labels=None, save_name="results"):
    """
    Plots the results of simulation from the given log files in grouped plot.

    Args:
        log_fps (list): list of filepaths to logs (csv)
        plot_type (str): either "reward" or "potential" to determine which data to plot.
        save_name (str): the name to use when saving the plot.
    """

    # Set up subplots here
    fig, axes = plt.subplots(2, 2, layout='constrained', figsize=(12, 6))
    if plot_title:
        fig.suptitle(plot_title)

    fps = [fail_log_fps, task_log_fps]
    # == PLOT RESULTS ==
    for i, fp in enumerate(fps):
        for log_fps, ax, sub_title, group_label in zip(fp, axes[:,i], subplot_titles, group_labels):

            data = {"Sim-BRVNS": [[], []],
                        "Dec-MCTS": [[], []],
                        "2-Stage": [[], []],
                        "2-Stage Hybrid": [[], []], }

            for j, log_fp in enumerate(log_fps):
                print(f"Plotting results from log file: {log_fp}")
                # parse log file name from log_fp
                basename = os.path.basename(log_fp)
                name = os.path.splitext(basename)[0]

                df = pd.read_csv(log_fp)

                # Extract values for each algorithm
                frontEndOnly = df[["frontEndOnly_rew",
                                "frontEndOnly_potent"]].values
                distrOnly = df[["distrOnly_rew", "distrOnly_potent"]].values
                twoStep = df[["twoStep_rew", "twoStep_potent"]].values
                distHybrid = df[["dist_hybrid_rew", "dist_hybrid_rew"]].values
                fullHybrid = df[["full_hybrid_rew", "full_hybrid_rew"]].values

                # Mean rewards
                data["Sim-BRVNS"][0].append(
                    round(np.mean(frontEndOnly[:, 0]), 2))
                data["Dec-MCTS"][0].append(round(np.mean(distrOnly[:, 0]), 2))
                data["2-Stage"][0].append(round(np.mean(twoStep[:, 0]), 2))
                data["2-Stage Hybrid"][0].append(
                    round(np.mean(fullHybrid[:, 0]), 2))

                # SE of Means
                data["Sim-BRVNS"][1].append(
                    np.std(frontEndOnly[:, 0]) / np.sqrt(len(frontEndOnly[:, 0])))
                data["Dec-MCTS"][1].append(
                    np.std(distrOnly[:, 0]) / np.sqrt(len(distrOnly[:, 0])))

                data["2-Stage"][1].append(np.std(twoStep[:, 0]) /
                                            np.sqrt(len(twoStep[:, 0])))
                data["2-Stage Hybrid"][1].append(
                    np.std(fullHybrid[:, 0]) / np.sqrt(len(fullHybrid[:, 0])))

            x = np.arange(len(log_fps))
            width = 0.2
            multiplier = 0

            for attribute, measurements in data.items():
                offset = width * multiplier
                rects = ax.bar(
                    x + offset, measurements[0], width, yerr=measurements[1], label=attribute)
                ax.bar_label(rects, padding=0)
                multiplier += 1

            ax.set_ylabel('% Total Reward')
            # ax.set_xlabel('Test Instance')
            ax.set_title(sub_title[i])
            # ax.legend(loc='upper right', ncols=1)
            if group_labels:
                ax.set_xticks(x + 1.5*width, group_labels[i])
            ax.set_ylim(0, 1.0)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
            ncol=4)#, bbox_to_anchor=(0.5, 0.955))  # (0.5, 0.955)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    
    
    # == SAVE FIG ==
    parent_path = os.getcwd()
    # parent_path = os.path.dirname(current_path)
    if not os.path.exists(f"{parent_path}/imgs"):
        os.makedirs(f"{parent_path}/imgs")   
        
    fig.savefig(f"{parent_path}/imgs/{save_name}.png")
    print(f"Plot saved to: {parent_path}/imgs/{save_name}.png")
    plt.show()



if __name__ == "__main__":
    # Predefined file paths
    parent_path = os.getcwd()
    # print("Current path", current_path)
    # parent_path = os.path.dirname(current_path)
    
    log_fps_fails = [
        [
            f'{parent_path}/results/30n3r_fails0_tasks0_30tr.csv',
            f'{parent_path}/results/30n3r_fails025_tasks0_30tr.csv',
            f'{parent_path}/results/30n3r_fails05_tasks0_30tr.csv',
        ],
        [
            f'{parent_path}/results/30n6r_fails0_tasks0_30tr.csv',
            f'{parent_path}/results/30n6r_fails025_tasks0_30tr.csv',
            f'{parent_path}/results/30n6r_fails05_tasks0_30tr.csv',
        ]
    ]

    log_fps_tasks = [
        [
            f'{parent_path}/results/20n3r_fails0_tasks0_30tr.csv',
            f'{parent_path}/results/20n3r_fails0_tasks05_30tr.csv',
            f'{parent_path}/results/20n3r_fails0_tasks10_30tr.csv',
        ],
        [
            f'{parent_path}/results/20n6r_fails0_tasks0_30tr.csv',
            f'{parent_path}/results/20n6r_fails0_tasks05_30tr.csv',
            f'{parent_path}/results/20n6r_fails0_tasks10_30tr.csv',
        ]
    ]

    plot_title = "Experimental Results"
    subplot_titles = [("Fails: 3 Workers", "Tasks: 3 Workers"),
                      ("Fails: 6 Workers", "Tasks: 6 Workers")
                      ]

    group_labels = [("Fail Rate 0.0%",
                          "Fail Rate 2.5%",
                          "Fail Rate 5.0%",),
                    ("Task Rate 0.0%",
                          "Task Rate 5.0%",
                          "Task Rate 10.0%",)
                    ]

    # Plot rewards and save
    grouped_plot(log_fps_fails,
                 log_fps_tasks,
                 subplot_titles,
                 plot_title,
                 group_labels,
                 save_name="plot_config_test")
