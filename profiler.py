from util import *
from plotter import *


"""
    Finds the optimal runtime given different configurations
"""
def oracle_runtime(df):
    return df.iloc[:,1:].min(axis=1).sum()

""" 
    Runtime of a configuration
"""
def runtime(df):
    return df.iloc[:,1].sum()

def find_best_device_for_layers(df):
    res = {}
    for i,series in df.iterrows():
        layer = series["Layer"]
        res[layer] = CONFIG_MAP[series[["Time_interp","Time_cpp","Time_gpu","Time_triton"]].argmin()]
    return res

def plot_rt_diff(df0,df1,autograd,config,modelname,filter=None,verbose=False):
    title = modelname+"_"+config
    title = title.replace(" ","_")
    config = config.replace(" ","_")
    df = df_w_speedup(df0,df1,config).dropna(axis=0, subset='Speedup')
    if filter: 
        savepath = DIR + f"figs/{config}/{filter.lower()}_only/"
        df = df[df['Layer'].str.startswith(filter)]
    else:
        savepath = DIR + f"figs/{config}/all/"
    if autograd:
        title = title+"_autograd"
    density_plot_model(df,'Speedup',title,savepath,verbose)

"""
    Input: merged [interp_df, cpp_df, gpu_df, triton_df
"""
def compare_runtimes(modelname, df):
    irt = 0
    crt = 0
    grt = 0
    trt = 0
    ort = oracle_runtime(df)
    for i,series in df.iterrows():
        irt += series["Time_interp"]
        crt += series["Time_cpp"]
        grt += series["Time_gpu"]
        trt += series["Time_triton"]
    nano = float(1e3)
    # bar_plot(["interp","cpp","gpu","triton","oracle"],[irt/nano,crt/nano,grt/nano,trt/nano,ort/nano],modelname)
    # print(f"{modelname.upper()}")
    # print(f"\tInterp runetime {irt:.8}")
    # print(f"\tCPP runtime     {crt:.8}")
    # print(f"\tGPU runtime     {grt:.8}")
    # print(f"\tTriton runtime  {trt:.8}")
    # print(f"\toracle runtime  {ort:.8}")


def profile_hooktraces(filenames):
    modelname = filenames[0].split('_')[0]
    interp_df = pickle_to_df(filenames[INTERP])
    cpp_df = pickle_to_df(filenames[COMPILED])
    gpu_df = pickle_to_df(filenames[GPU])
    triton_df = pickle_to_df(filenames[TRITON])

    """
        Best device per layer
    """
    dict = find_best_device_for_layers([interp_df,cpp_df,gpu_df,triton_df],autograd=False)

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


def find_slow_layers(dct,fdf):
    indxs = []
    for i,(k,v) in enumerate(dct.items()):
        if v != 'triton':
            indxs.append(i)
    indxs = np.array(indxs)

    slow = fdf.iloc[indxs][["Layer","args.Input Dims"]].dropna(axis=0,subset=["args.Input Dims"])
    fast = fdf.iloc[~fdf.index.isin(indxs)][["Layer","args.Input Dims"]].dropna(axis=0,subset=["args.Input Dims"])

    return slow,fast

def profile_autogradtraces(filenames,verbose=False):
    modelname = filenames[0].split('_')[0]
    id,fid = parse_autograd_json(filenames[INTERP]+"_nohooks.trace")
    cd,fcd = parse_autograd_json(filenames[COMPILED]+"_nohooks.trace")
    gd,fgd = parse_autograd_json(filenames[GPU]+"_nohooks.trace")
    td,ftd = parse_autograd_json(filenames[TRITON]+"_nohooks.trace")
    df = merge_frames([id,cd,gd,td])
    # fdf = merge_frames([fid,fcd,fgd,ftd])

    # compare_runtimes(modelname,df)

    # plot_rt_diff(id,cd,True,"interp vs cpp",modelname)
    # plot_rt_diff(gd,td,True,"gpu vs triton",modelname)

    dct = find_best_device_for_layers(df)
    slow,fast = find_slow_layers(dct,ftd)
    print("##########SLOW##########")
    print(slow.to_string())
    print("##########FAST##########")
    print(fast.to_string())
    exit()
    
