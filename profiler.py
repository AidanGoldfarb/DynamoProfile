from util import *
from plotter import density_plot

"""
    Finds the optimal runtime given different configurations
"""
def oracle_runtime(dfs):
    assert len(dfs) > 1, "supply >1 dfs to the oracle"
    
    merged_df = dfs[0]
    for i,df in enumerate(dfs[1:]):
        # print(df.columns)
        # print(merged_df.columns)
        merged_df = pd.merge(merged_df,df, on='Layer')
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
    compiled_df = pickle_to_df(filenames[COMPILED])
    gpu_df = pickle_to_df(filenames[GPU])
    triton_df = pickle_to_df(filenames[TRITON])
    print(filenames[0].split('_')[0].upper())
    print(f"\tinterp:   {runtime(interp_df):.4}")
    print(f"\tcompiled: {runtime(compiled_df):.4}")
    print(f"\tgpu:      {runtime(gpu_df):.4}")
    print(f"\ttriton:   {runtime(triton_df):.4}")
    print(f"\toracle:   {oracle_runtime([interp_df,compiled_df,gpu_df,triton_df]):.4}")
    # print(oracle_runtime([interp_df,compiled_df,gpu_df,triton_df]))
    # plot_rt_diff(interp_df,interp_df)
    
