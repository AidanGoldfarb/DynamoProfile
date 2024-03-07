import seaborn as sns
import matplotlib.pyplot as plt
from util import *
import numpy as np

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