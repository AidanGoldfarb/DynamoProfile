import seaborn as sns
import matplotlib.pyplot as plt
from util import *
import numpy as np

import matplotlib.pyplot as plt

def plot_benchmark_results(data, unprof_data, modelname):
    # Updated labels to include the new bars for unprof_data tuple
    bar_labels = ['CUDA_total', 'CUDA_Self', 'CPU_Self', 'Arr sum', 'e2e', 'Arrsum_Unprof', 'Et2_Unprof']
    group_labels = ['pure_cuda', 'pure_triton', 'sync_cuda', 'sync_triton', 'timed_cuda', 'timed_triton', 'timed_sync_cuda', 'timedsync_triton']
    
    # Updated colors for each bar
    colors = ['lightcoral', 'forestgreen', 'orange', 'steelblue', 'lightseagreen', 'purple', 'darkblue']
    
    # Number of groups
    n_groups = len(data)
    
    # Setting the bar width and the positions of the bars and groups
    bar_width = 0.15
    group_width = len(bar_labels) * bar_width + 0.1  # spacing between groups
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generating the bars
    for i, group_data in enumerate(data):
        base_position = i * group_width
        
        # Extract the values
        cuda_total, cuda_self, cpu_self, arr_sum, end_to_end = group_data
        
        # Plot each bar in the group
        if cuda_total is not None:
            ax.bar(base_position, cuda_total, color=colors[0], width=bar_width)
        
        if cuda_self is not None:
            ax.bar(base_position + bar_width, cuda_self, color=colors[1], width=bar_width)
            
            # Stack the CPU_Self bar on top of CUDA_Self
            if cpu_self is not None:
                ax.bar(base_position + bar_width, cpu_self, bottom=cuda_self, color=colors[2], width=bar_width)
        
        if arr_sum is not None:
            ax.bar(base_position + 2 * bar_width, arr_sum, color=colors[3], width=bar_width)
        
        if end_to_end is not None:
            ax.bar(base_position + 3 * bar_width, end_to_end, color=colors[4], width=bar_width)
        
        if unprof_data[i] is not None:
            arrsum_unprof, et2_unprof = unprof_data[i]
            if arrsum_unprof is not None:
                ax.bar(base_position + 4 * bar_width, arrsum_unprof, color=colors[5], width=bar_width)
            if et2_unprof is not None:
                ax.bar(base_position + 5 * bar_width, et2_unprof, color=colors[6], width=bar_width)

    # Setting the x-axis labels and title
    ax.set_xticks([i * group_width + 2 * bar_width for i in range(n_groups)])
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
    plt.savefig('figs/full/' + modelname)

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

def plot_sublayer_times():
    file_path = 'layertimes.txt'
    
    def read_data(file_path):
        with open(file_path, 'r') as file:
            raw_data = file.read().strip().split('\n\n')
        data_sets = [np.array([list(map(float, line.split())) for line in section.split('\n')]) for section in raw_data]
        return data_sets

    def create_bar_plot(data, index, modelname):
        x = np.arange(data.shape[0])
        width = 0.25

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width, data[:, 0], width, label='interpreted')
        bars2 = ax.bar(x, data[:, 1], width, label='compiled')
        bars3 = ax.bar(x + width, data[:, 2], width, label='custom')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Runtime [ns]')
        ax.set_xticks(x)
        ax.legend()

        plt.title(modelname)
        plt.savefig(f'figs/subtimebars/{modelname}_subtimebar.png')
        plt.close()

    data_sets = read_data(file_path)
    modelnames = ['resnet','googlenet','densenet','squeezenet','alexnet','mobilenetv2']
    for i, data in enumerate(data_sets):
        create_bar_plot(data, i, modelnames[i])