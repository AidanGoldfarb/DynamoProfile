import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# def bar_plot(xs,ys):
#     for x,y in zip(xs,ys):
#         plt.bar(x,y)
#     plt.savefig("testbar", bbox_inches='tight')

def bar_plot(xs, ys):
    # Number of bar groups
    n_groups = len(xs[0])
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    
    # Set the width of a bar
    bar_width = 0.35
    
    # Calculate the positions of the bars
    index = np.arange(n_groups)
    
    # Plot each group
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.bar(index + i * bar_width, y, bar_width, label=f'Group {i+1}')
    
    # Add x-axis labels, title, etc
    ax.set_xlabel('Name')
    ax.set_ylabel('Duration')
    ax.set_title('Duration by Name and Group')
    ax.set_xticks(index + bar_width / len(xs))
    ax.set_xticklabels(xs[0])  # Assuming all xs are the same
    ax.legend()
    
    # Tweak spacing to prevent clipping of tick-labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("testbar", bbox_inches='tight')

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