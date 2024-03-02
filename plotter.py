import seaborn as sns
import matplotlib.pyplot as plt

"""
    Plot density of 'var'
"""
def density_plot_model(data, var: str, title=""):
    sns.displot(data, x=var, kind="kde")
    plt.axvline(1, color='red', linestyle="--")
    plt.title(title)
    plt.savefig("figs/"+title.replace(" ","_"), bbox_inches='tight')

"""
    figure out how to combine all models
"""
def density_plot_all(data, var: str, title=""):
    for df in data:
        sns.displot(data, x=var, kind="kde")
    plt.axvline(1, color='red', linestyle="--")
    plt.title(title)
    plt.savefig("figs/"+title.replace(" ","_"), bbox_inches='tight')