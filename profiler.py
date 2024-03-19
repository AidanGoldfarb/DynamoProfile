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
def compare_runtimes(modelname, df, device='all'):
    irt = 0
    crt = 0
    grt = 0
    trt = 0
    #ort = oracle_runtime(df)
    gpu_col = f"Time_my{modelname}_gpu_default"
    triton_col = f"Time_{modelname}_triton_default"
    grt = df[gpu_col].sum()
    trt = df[triton_col].sum()
    # for i,series in df.iterrows():
    #     if device == 'all':
    #         irt += series["Time_interp"]
    #         crt += series["Time_cpp"]
    #         grt += series["Time_gpu"]
    #         trt += series["Time_triton"]
    #     elif device == 'cpu':
    #         irt += series["Time_interp"]
    #         crt += series["Time_cpp"]
    #     elif device == 'gpu':
    #         grt += series[gpu_col]
    #         trt += series[triton_col]
    nano = float(1e3)
    print(modelname,grt,trt)
    # bar_plot(["interp","cpp","gpu","triton","oracle"],[irt/nano,crt/nano,grt/nano,trt/nano,ort/nano],modelname)
    # print(f"{modelname.upper()}")
    # print(f"\tInterp runetime {irt:.8}")
    # print(f"\tCPP runtime     {crt:.8}")
    # print(f"\tGPU runtime     {grt:.8}")
    # print(f"\tTriton runtime  {trt:.8}")
    # print(f"\toracle runtime  {ort:.8}")

def get_cuda_triton():
    rct = merge_frames([
        parse_autograd_json("resnet_default_gpu_nohooks.trace"),
        parse_autograd_json("resnet_default_triton_nohooks.trace")
    ])
    act = merge_frames([
        parse_autograd_json("alexnet_default_gpu_nohooks.trace"),
        parse_autograd_json("alexnet_default_triton_nohooks.trace")
    ])
    dct = merge_frames([
        parse_autograd_json("densenet_default_gpu_nohooks.trace"),
        parse_autograd_json("densenet_default_triton_nohooks.trace")
    ])
    gct = merge_frames([
        parse_autograd_json("googlenet_default_gpu_nohooks.trace"),
        parse_autograd_json("googlenet_default_triton_nohooks.trace")
    ])
    sct = merge_frames([
        parse_autograd_json("squeezenet_default_gpu_nohooks.trace"),
        parse_autograd_json("squeezenet_default_triton_nohooks.trace")
    ])
    mct = merge_frames([
        parse_autograd_json("mobilenetv2_default_gpu_nohooks.trace"),
        parse_autograd_json("mobilenetv2_default_triton_nohooks.trace")
    ])
    return (rct,act,dct,gct,sct,mct)

def get_interp_cpp():
    rct = merge_frames([
        parse_autograd_json("resnet_default_interp_nohooks.trace"),
        parse_autograd_json("resnet_default_cpp_nohooks.trace")
    ])
    act = merge_frames([
        parse_autograd_json("alexnet_default_interp_nohooks.trace"),
        parse_autograd_json("alexnet_default_cpp_nohooks.trace")
    ])
    dct = merge_frames([
        parse_autograd_json("densenet_default_interp_nohooks.trace"),
        parse_autograd_json("densenet_default_cpp_nohooks.trace")
    ])
    gct = merge_frames([
        parse_autograd_json("googlenet_default_interp_nohooks.trace"),
        parse_autograd_json("googlenet_default_cpp_nohooks.trace")
    ])
    sct = merge_frames([
        parse_autograd_json("squeezenet_default_interp_nohooks.trace"),
        parse_autograd_json("squeezenet_default_cpp_nohooks.trace")
    ])
    mct = merge_frames([
        parse_autograd_json("mobilenetv2_default_interp_nohooks.trace"),
        parse_autograd_json("mobilenetv2_default_cpp_nohooks.trace")
    ])
    return (rct,act,dct,gct,sct,mct)

def find_slow_layers():
    rct,act,dct,gct,sct,mct = get_cuda_triton()
    df = mct
    df = add_speedup(reorder_cols(df))
    # df = df.loc[df['Layer'].str.contains('conv')]
    df = df.sort_values(by="Speedup",ascending=True)
    df = df.loc[df['Speedup'] != 0]
    df = df.dropna(axis=0,subset=['Speedup'])
    print(df.to_string())
    # df = rct
    # df = add_speedup(df)
    # dimcol = df.columns[4]
    # df.dropna(axis=0,subset="Speedup",inplace=True)
    # df.sort_values(by=dimcol,ascending=True, inplace = True)
    # df = df.loc[df[dimcol] != -1]
    # df = df.loc[df['Layer'].str.contains('conv')]
    # #print(df.iloc[:,0:6].to_string())
    # line_plot(df.iloc[:,4],df.iloc[:,3],title="df conv only")


def compare_cust_configs():
    triton = {}
    custom = {}
    for filename in os.listdir(DIR+"cache/autogradtraces/"):
        model,_,device,_ = filename.split("_")
        if 'my' in filename and 'gpu' in device:
            custom[model] = filename
        elif "my" not in filename and 'triton' in device:
            triton[model] = filename 
    
    for tf,cf in zip(triton.values(),custom.values()):
        modelname = tf.split("_")[0]
        compare_runtimes(modelname,merge_frames([parse_autograd_json(tf),parse_autograd_json(cf)]),device='gpu')
            

def profile_hooktraces(filenames, device='all'):
    modelname = filenames[0].split('_')[0]
    id = pickle_to_df(filenames[INTERP])
    cd = pickle_to_df(filenames[COMPILED])
    gd = pickle_to_df(filenames[GPU])
    td = pickle_to_df(filenames[TRITON])
    
    df = merge_frames([id,cd,gd,td])
    print(df.shape)

    """
        Best device per layer
    """
    #dict = find_best_device_for_layers([interp_df,cpp_df,gpu_df,triton_df],autograd=False)

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

def profile_autogradtraces(filenames,device='all',verbose=False):
    modelname = filenames[0].split('_')[0]
    # id,fid = parse_autograd_json(filenames[INTERP]+"_nohooks.trace")
    # cd,fcd = parse_autograd_json(filenames[COMPILED]+"_nohooks.trace")
    # gd,fgd = parse_autograd_json(filenames[GPU]+"_nohooks.trace")
    # td,ftd = parse_autograd_json(filenames[TRITON]+"_nohooks.trace")
    #df = merge_frames([id,cd,gd,td])
    # fdf = merge_frames([fid,fcd,fgd,ftd])

    # compare_runtimes(modelname,df)

    # plot_rt_diff(id,cd,True,"interp vs cpp",modelname)
    # plot_rt_diff(gd,td,True,"gpu vs triton",modelname)

    # dct = find_best_device_for_layers(df)
    # slow,fast = find_slow_layers(dct,ftd)
    # print("##########SLOW##########")
    # print(slow.to_string())
    # print("##########FAST##########")
    # print(fast)
    
