import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results_from_folder(folder_fp):
    return


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
