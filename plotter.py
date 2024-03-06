import seaborn as sns
import matplotlib.pyplot as plt
from util import *
import numpy as np

# def bar_plot(xs,ys):
#     for x,y in zip(xs,ys):
#         plt.bar(x,y)
#     plt.savefig("testbar", bbox_inches='tight')

def bar_plot(xs, ys, title):
    # Number of bar groups
    plt.bar(xs,ys,color="teal")
    plt.xlabel("Config")
    plt.ylabel("Runtime [μs]")
    plt.title(title)
    plt.savefig(DIR + "figs/rt/"+title, bbox_inches='tight')

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