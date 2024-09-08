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
    try:
        best = df["best"]
    except:
        best = False
    frontEndOnly = df[[
        col for col in df.columns if "frontEndOnly" in col]].values
    distrOnly = df[[
        col for col in df.columns if "distrOnly" in col]].values
    twoStep = df[[
        col for col in df.columns if "twoStep" in col]].values
    hybrid = df[[
        col for col in df.columns if "hybrid" in col]].values

    title = "Results for test " + name

    plot_results(trial,
                 test,
                 best,
                 frontEndOnly,
                 distrOnly,
                 twoStep,
                 hybrid,
                 title,
                 name
                 )


def plot_results(trial,
                 test,
                 best,
                 frontEnd_results,
                 distrOnly_results,
                 twoPart_results,
                 hybrid_results,
                 title="Results",
                 figname="Fig"
                 ):
    print("BEST", best)

    # Mean Rewards
    if best:
        best_rew = round(np.mean(best), 2)
    frontEnd_rew = round(np.mean([res[0] for res in frontEnd_results]), 2)
    distrOnly_rew = round(np.mean([res[0] for res in distrOnly_results]), 2)
    twoPart_rew = round(np.mean([res[0] for res in twoPart_results]), 2)
    hybrid_rew = round(np.mean([res[0] for res in hybrid_results]), 2)
    # Mean potential rewards
    frontEnd_pot = round(np.mean([res[1] for res in frontEnd_results]), 2)
    distrOnly_pot = round(np.mean([res[1] for res in distrOnly_results]), 2)
    twoPart_pot = round(np.mean([res[1] for res in twoPart_results]), 2)
    hybrid_pot = round(np.mean([res[1] for res in hybrid_results]), 2)
    # Mean robots lost
    frontEnd_fails = round(np.mean([res[2] for res in frontEnd_results]), 2)
    distrOnly_fails = round(np.mean([res[2] for res in distrOnly_results]), 2)
    twoPart_fails = round(np.mean([res[2] for res in twoPart_results]), 2)
    hybrid_fails = round(np.mean([res[2] for res in hybrid_results]), 2)

    # StdErr
    # Rewards
    if best:
        best_se = np.std(best) / np.sqrt(len(best))
    frontEnd_rew_se = np.std([res[0] for res in frontEnd_results]) / \
        np.sqrt(len(frontEnd_results))
    distrOnly_rew_se = np.std([res[0] for res in distrOnly_results]) / \
        np.sqrt(len(distrOnly_results))
    twoPart_rew_se = np.std([res[0] for res in twoPart_results]) / \
        np.sqrt(len(twoPart_results))
    hybrid_rew_se = np.std([res[0] for res in hybrid_results]) / \
        np.sqrt(len(hybrid_results))
    # Potentials
    frontEnd_pot_se = np.std([res[1] for res in frontEnd_results]) / \
        np.sqrt(len(frontEnd_results))
    distrOnly_pot_se = np.std([res[1] for res in distrOnly_results]) / \
        np.sqrt(len(distrOnly_results))
    twoPart_pot_se = np.std([res[1] for res in twoPart_results]) / \
        np.sqrt(len(twoPart_results))
    hybrid_pot_se = np.std([res[1] for res in hybrid_results]) / \
        np.sqrt(len(hybrid_results))

    avg_rew = [frontEnd_rew, distrOnly_rew, twoPart_rew, hybrid_rew]

    avg_pot = [frontEnd_pot, distrOnly_pot, twoPart_pot, hybrid_pot]

    error_rew = [frontEnd_rew_se, distrOnly_rew_se,
                 twoPart_rew_se, hybrid_rew_se]

    error_pot = [frontEnd_pot_se, distrOnly_pot_se,
                 twoPart_pot_se, hybrid_pot_se]

    rew_content = {
        "Tasks Visited": (avg_pot, error_pot),
        "Tasks Returned": (avg_rew, error_rew),
    }

    if best:
        labels = ["Best", "Front-End Only",
                  "Distr. Only", "Front End\n+ Dist Replan",
                  "Front End\n+ Hybrid Replan"]
    else:
        labels = ["Front-End Only",
                  "Distr. Only", "Front End\n+ Dist Replan",
                  "Front End\n+ Hybrid Replan"]

    # Plot results
    fig, ax = plt.subplots()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    x = np.arange(len(labels))
    width = 0.3
    multiplier = 0
    if best:
        rects = ax.bar(x[0]+(width/2), best_rew, width,
                       yerr=best_se,  label="Best")
        ax.bar_label(rects, padding=3)
        start = x[1:]
    else:
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
    if hybrid_rew < 0.5:
        ax.legend(loc='upper right', ncols=1)
    else:
        ax.legend(loc='lower right', ncols=1)

    fig.savefig(f"{figname}.png")

    print("Done")

    plt.show()
