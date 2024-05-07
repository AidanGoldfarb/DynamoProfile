import seaborn as sns
import matplotlib.pyplot as plt
from util import *
import numpy as np

def plot_benchmark_results(data,modelname):
    # Labels for the bars and groups
    bar_labels = ['CUDA_total', 'CUDA_Self', 'Arr sum', 'End to end']
    group_labels = ['pure_cuda', 'pure_triton', 'sync_cuda', 'sync_triton', 'timed_cuda', 'timed_triton', 'timed_sync_cuda', 'timedsync_triton']
    
    # Colors for each bar
    colors = ['red', 'green', 'blue', 'yellow']
    
    # Number of groups
    n_groups = len(data)
    
    # Setting the bar width and the positions of the bars and groups
    bar_width = 0.25
    group_width = n_groups * bar_width * 1.2  # spacing between groups
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generating the bars
    for i, (group_data) in enumerate(data):
        # Each tuple in data corresponds to one group
        for j, value in enumerate(group_data):
            if value is not None:  # Only plot if the value is not None
                # Plot each bar in the group
                ax.bar(i * group_width + j * bar_width, value, color=colors[j], width=bar_width)

    # Setting the x-axis labels and title
    ax.set_xticks([i * group_width + bar_width for i in range(n_groups)])
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    ax.set_title(modelname)
    plt.ylabel("runtime [us]")
    
    # Creating custom legend handles manually
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=bar_labels[i]) for i in range(len(bar_labels))]

    # Adding a legend
    ax.legend(handles=legend_handles)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('figs/full/'+modelname)

def plot_arrsum_vs_total(modelname, arr_sums, tots):
    # Ensure the inputs are correctly sized
    assert len(arr_sums) == 6 and len(tots) == 6, "arr_sums and tots must each have 3 elements."

    n_groups = 6
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, arr_sums, bar_width, alpha=opacity, color='b', label='Layer Sum')
    rects2 = ax.bar(index + bar_width, tots, bar_width, alpha=opacity, color='g', label='End to End')

    ax.set_ylabel('Runtime [ns]')
    ax.set_title(modelname + " e2e runtime vs sum of layer times")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['CUDA', 'Triton', 'Custom', 'Oracle','CUDA_pure', 'Triton_pure'])
    ax.legend()

    plt.tight_layout()
    plt.savefig("figs/"+modelname)

def line_plot(xs,ys,title="title"):
    plt.scatter(xs,ys,marker='x')
    
    b, a = np.polyfit(xs, ys, deg=1)
    xseq = np.linspace(0, np.max(xs), num=10)
    plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
    
    correlation_matrix = np.corrcoef(xs, ys)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(r_squared)

    plt.title(title)
    plt.xlabel("Number of parameters")
    plt.ylabel("Speedup")
    plt.ylim(0,2)
    # plt.legend()
    plt.savefig("testfig")
    plt.close()

def bar_plot(xs, ys, title):
    # Number of bar groups
    plt.bar(xs,ys,color="teal")
    plt.xlabel("Config")
    plt.ylabel("Runtime [μs]")
    plt.title(title)
    plt.savefig(DIR + "figs/rt/"+title, bbox_inches='tight')
    plt.close()

"""
    Plot density of 'var'
"""
def density_plot_model(data, var: str, title="", savepath="", verbose=False):
    sns.displot(data, x=var, kind="kde")
    plt.axvline(1, color='red', linestyle="--")
    plt.title(title)
    plt.savefig(savepath+title, bbox_inches='tight')
    if verbose:
        print(f"wrote fig to {savepath+title}")
    plt.close()

"""
    figure out how to combine all models
"""
def density_plot_all(data, var: str, title=""):
    for df in data:
        sns.displot(data, x=var, kind="kde")
    plt.axvline(1, color='red', linestyle="--")
    plt.title(title)
    plt.savefig("figs/"+title.replace(" ","_"), bbox_inches='tight')