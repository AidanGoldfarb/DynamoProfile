from util import *
from plotter import *


"""
    Finds the optimal runtime given different configurations
"""
def oracle_runtime(modelname):
    filenames = FILENAMES[modelname]
    id = parse_autograd_json(filenames[INTERP]+"_nohooks.trace")
    cd = parse_autograd_json(filenames[COMPILED]+"_nohooks.trace")
    gd = parse_autograd_json(filenames[GPU]+"_nohooks.trace")
    td = parse_autograd_json(filenames[TRITON]+"_nohooks.trace")
    cd = parse_autograd_json("my"+filenames[GPU]+"_nohooks.trace")
    df = merge_frames([id,cd,gd,td,cd])
    return df.iloc[:,1:5].min(axis=1).sum()

""" 
    Runtime of a configuration
"""
def runtime(df):
    return df.iloc[:,1].sum()

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
    gpu_col = f"Time_my{modelname}_gpu_default"
    triton_col = f"Time_{modelname}_triton_default"
    nano = float(1e3)

    irt = parse_autograd_json(modelname+"_default_interp_nohooks.trace")[f"Time_{modelname}_interp_default"].sum()
    crt = parse_autograd_json(modelname+"_default_cpp_nohooks.trace")[f"Time_{modelname}_cpp_default"].sum()
    grt = parse_autograd_json(modelname+"_default_gpu_nohooks.trace")[f"Time_{modelname}_gpu_default"].sum()
    trt = df[triton_col].sum()
    cust_rt = df[gpu_col].sum()
    ort = oracle_runtime(modelname)
    
    #print(f"MODEL: {modelname}\n\tinterp: {irt}\n\tcpp:    {crt}\n\tgpu:    {grt}\n\ttriton: {trt}\n\tcustom: {cust_rt}\n\toracle: {ort}\n\tspeedup: {trt/cust_rt:.4}")
    bar_plot(["interp","cpp","gpu","triton","custom","oracle"],[irt/nano, crt/nano, grt/nano, trt/nano, cust_rt/nano, ort/nano],modelname)

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
        parts = filename.split("_")
        model_name_with_prefix, _, device, _ = parts
        model_name = model_name_with_prefix.replace('my', '')  # Remove 'my' prefix for matching
        
        if 'my' in model_name_with_prefix and 'gpu' in device:
            custom[model_name] = filename
        elif 'my' not in model_name_with_prefix and 'triton' in device:
            triton[model_name] = filename

    for model_name, tf in triton.items():
        if model_name in custom:
            cf = custom[model_name]
            df = merge_frames([parse_autograd_json(tf),parse_autograd_json(cf)])
            compare_runtimes(model_name,df,device='gpu')
            
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

def profile_autogradtraces(verbose=False):
    filenames = FILENAMES["resnet"]
    cuda = parse_autograd_json(filenames[GPU]+"_nohooks.trace")
    triton = parse_autograd_json(filenames[TRITON]+"_nohooks.trace")
    
    df = merge_frames([cuda,triton])
    #df = df.loc[df['Layer'].str.contains('conv')]
    df.dropna(axis=1,how='all',inplace=True)
    df = reorder_cols(df)
    df = add_speedup(df)
    df.dropna(axis=0,subset=['Speedup'])
    df.sort_values(by="Speedup",axis=0,ascending=False,inplace=True)

    
    print(df.to_string())
    
    
