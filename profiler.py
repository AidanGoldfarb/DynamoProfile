from util import *
from plotter import density_plot

"""
    Finds the optimal runtime given different configurations
"""
def oracle_runtime(dfs):
    assert len(dfs) > 1, "supply >1 dfs to the oracle"
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df,df, on='Type')

    return merged_df.iloc[:,1:].min(axis=1).sum()

""" 
    Runtime of a configuration
"""
def runtime(df):
    return df.iloc[:,1].sum()

def plot_rt_diff(df0,df1):
    df = df_w_diff(df0,df1)
    density_plot(df,'Diff')

def profile(filenames):
    interp_df = pickle_to_df(filenames[INTERP])
    # compiled_df = pickle_to_df(filenames[COMPILED])
    # gpu_df = pickle_to_df(filenames[GPU])
    # triton_df = pickle_to_df(filenames[TRITON])

    # print(oracle_runtime([interp_df,compiled_df,gpu_df,triton_df]))
    plot_rt_diff(interp_df,interp_df)
    
