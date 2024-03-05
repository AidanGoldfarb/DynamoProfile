from util import *
from plotter import *


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

def find_best_device_for_layers(dfs):
    res = {}
    df = merge_frames(dfs)
    
    df['Layer_Modified'] = df['Layer'].str.split("(").str[0]
    min_value_columns = df.drop(columns=['Layer', 'Layer_Modified']).idxmin(axis=1)
    df['MinValueColumn'] = min_value_columns
    dict = df.groupby('Layer_Modified')['MinValueColumn'].agg(lambda x: x.iloc[0]).to_dict()
    for k,v in dict.items():
        print(k,v)
        

def plot_rt_diff(df0,df1,autograd,config="",modelname="",filter=None,verbose=False):
    title = modelname+"_"+config
    title = title.replace(" ","_")
    config = config.replace(" ","_")
    df = df_w_speedup(df0,df1,config,autograd).dropna(axis=0, subset='Speedup')
    if filter: 
        savepath = DIR + f"figs/{config}/{filter.lower()}_only/"
        if autograd:
            df = df[df['Layer'].str.startswith("aten::"+filter)]
        else:
            df = df[df['Layer'].str.startswith(filter)]
    else:
        savepath = DIR + f"figs/{config}/all/"
    if autograd:
        title = title+"_autograd"
    density_plot_model(df,'Speedup',title,savepath,verbose)

def profile_hooktraces(filenames):
    modelname = filenames[0].split('_')[0]
    interp_df = pickle_to_df(filenames[INTERP])
    cpp_df = pickle_to_df(filenames[COMPILED])
    gpu_df = pickle_to_df(filenames[GPU])
    triton_df = pickle_to_df(filenames[TRITON])

    """
        Best device per layer
    """
    dict = find_best_device_for_layers([interp_df,cpp_df,gpu_df,triton_df])

    """
        Runtime
    """
    # interp_rt = runtime(interp_df)
    # cpp_rt = runtime(cpp_df)
    # gpu_rt = runtime(gpu_df)
    # triton_rt = runtime(triton_df)
    # oracle_rt = oracle_runtime([interp_df,cpp_df,gpu_df,triton_df])

    # bar_plot(["interp","cpp","gpu","triton","oracle"],[interp_rt,cpp_rt,gpu_rt,triton_rt,oracle_rt], modelname)
    # print(filenames[0].split('_')[0].upper())
    # print(f"\tinterp:   {runtime(interp_df):.4}")
    # print(f"\tcompiled: {runtime(compiled_df):.4}")
    # print(f"\tgpu:      {runtime(gpu_df):.4}")
    # print(f"\ttriton:   {runtime(triton_df):.4}")
    # print(f"\toracle:   {oracle_runtime([interp_df,compiled_df,gpu_df,triton_df]):.4}")
    # print(oracle_runtime([interp_df,compiled_df,gpu_df,triton_df]))
    # plot_rt_diff(gpu_df,triton_df, modelname=modelname,config="gpu vs triton")
    # plot_rt_diff(gpu_df,triton_df, modelname=modelname,config="gpu vs triton",filter="Conv")
    # plot_rt_diff(interp_df,compiled_df,title= modelname +" interp vs cpp")
    
def profile_autogradtraces(filenames,verbose=False):
    modelname = filenames[0].split('_')[0]
    
    interp_df = parse_autograd_json(filenames[INTERP]+"_nohooks.trace")
    compiled_df = parse_autograd_json(filenames[COMPILED]+"_nohooks.trace")
    gpu_df = parse_autograd_json(filenames[GPU]+"_nohooks.trace")
    triton_df = parse_autograd_json(filenames[TRITON]+"_nohooks.trace")
    
    plot_rt_diff(interp_df,compiled_df,autograd=True, modelname=modelname,config="interp vs cpp",verbose=verbose)
    plot_rt_diff(gpu_df,triton_df,autograd=True, modelname=modelname,config="gpu vs triton",verbose=verbose)

    plot_rt_diff(interp_df,compiled_df,autograd=True, modelname=modelname,config="interp vs cpp",filter="conv",verbose=verbose)
    plot_rt_diff(gpu_df,triton_df,autograd=True, modelname=modelname,config="gpu vs triton",filter="conv",verbose=verbose)