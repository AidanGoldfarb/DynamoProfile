from util import *
from plotter import *
from itertools import islice



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


def _extract_dict():
    data_dict = {}
    with open("layertimes.txt", 'r') as f:
        group = []  
        for line in f:
            line = line.strip()
            if not line:
                continue 
            group.append(line)
            if len(group) == 6:
                data_dict[' '.join(group[-1].split()[0:2])] = (np.array(eval(group[-2])),int(group[-1].split()[-1]))
                group = []
    return data_dict


#Given a run, return a list of layers which triton accelerated
def spedup_layers(dct,th=10):
    df = pd.DataFrame.from_dict(dct)#, orient='index')
    cuda_arr = np.array(df['cuda_arr'])
    triton_arr = np.array(df['triton_arr'])
    lst = []
    for i,(cuda,triton) in enumerate(zip(cuda_arr,triton_arr)):
        #print(cuda,triton,cust)
        if triton < cuda and diff(triton,cuda) > th:
            lst.append(i)
    return lst

"""
    
"""
def compare_runtimes():
    dct = unpickle_obj("raw_run_dct")
    df = pd.DataFrame.from_dict(dct, orient='index')
    for i,series in df.iterrows():
        cuda_arr = np.array(series['cuda_arr'])
        triton_arr = np.array(series['triton_arr'])
        cust_arr = np.array(series['cust_arr'])

        oracle = 0
        for cuda,triton,cust in zip(cuda_arr,triton_arr,cust_arr):
            oracle += np.min([cuda,triton,cust])
        
        cuda_e2e = series['cuda_e2e']
        triton_e2e = series['triton_e2e']
        cust_e2e = series['cust_e2e']
        cuda_pure_e2e = series['cuda_pure_e2e']
        triton_pure_e2e = series['triton_pure_e2e']

        # for i,(cuda,triton,cust) in enumerate(zip(cuda_arr,triton_arr,cust_arr)):
        #     #print(cuda,triton,cust)
        #     if triton < cuda and diff(triton,cuda) > 10 and diff(triton,cust) > 10:
        #         print(i,diff(triton,cuda))

        plot_arrsum_vs_total(
            series.name,[cuda_arr.sum(),triton_arr.sum(),cust_arr.sum(),oracle,0,0],
            [cuda_e2e,triton_e2e,cust_e2e,0,cuda_pure_e2e,triton_pure_e2e
        ])
    # for modelname,data in dct.items():
    #     print(modelname)
    #     for k,v in data.items():
    #         print('\t',k)

    #sanity check
    #plot_arrsum_vs_total(modelname,[arr0.sum(),arr1.sum(),arr2.sum()],[tot0,tot1,tot2])
    
    # cuda = v0.sum()
    # triton = v1.sum()
    # cust = v2.sum()
    #bar_plot(["cuda","triton","custom"],[cuda,triton,cust],title=modelname)
        

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

def pull_data(file):
    df,arr,ete = unpickle_obj('autogradtraces/'+file)

    ete = ete/1e3
    arrsum = None
    cuda_total_sum = None
    cuda_self_sum = None
    cpu_self_sum = None
    if arr:
        arrsum = np.sum(arr)/1e3
    if df is not None:
        cuda_total_sum = df['CUDA_total_us'].sum()
        cuda_self_sum = df['Self_CUDA_us'].sum()
        cpu_self_sum = df['Self_CPU_us'].sum()
    #print(cuda_total_sum,cuda_self_sum,arrsum,ete)
    return cuda_total_sum,cuda_self_sum,cpu_self_sum,arrsum,ete

def profile_autogradtraces(verbose=False):
    files = [f for f in sorted(os.listdir(DIR+"cache/autogradtraces")) if os.path.isfile(os.path.join(DIR+"cache/autogradtraces", f))]

    for file_lst in chunker(files,16):
        modelname = file_lst[0].split('_')[0]

        pure_cuda_ete = pull_data(file_lst[0])[-2:]
        pure_triton_ete = pull_data(file_lst[2])[-2:]
        sync_cuda_ete = pull_data(file_lst[4])[-2:]
        sync_triton_ete = pull_data(file_lst[6])[-2:]
        timed_cuda_ete = pull_data(file_lst[8])[-2:]
        timed_triton_ete = pull_data(file_lst[10])[-2:]
        timedsync_cuda_ete = pull_data(file_lst[12])[-2:]
        timedsync_triton_ete = pull_data(file_lst[14])[-2:]

        pure_cuda = pull_data(file_lst[1])
        pure_triton = pull_data(file_lst[3])
        sync_cuda = pull_data(file_lst[5])
        sync_triton = pull_data(file_lst[7])
        timed_cuda = pull_data(file_lst[9])
        timed_triton = pull_data(file_lst[11])
        timedsync_cuda = pull_data(file_lst[13])
        timedsync_triton = pull_data(file_lst[15])

        plot_benchmark_results(
            [pure_cuda,pure_triton,sync_cuda,sync_triton,timed_cuda,timed_triton,timedsync_cuda,timedsync_triton],
            [pure_cuda_ete,pure_triton_ete,sync_cuda_ete,sync_triton_ete,timed_cuda_ete,timed_triton_ete,timedsync_cuda_ete,timedsync_triton_ete],
            modelname
        )
            
def find_bestconfig_dyn(model,numlayers):
    def measure_time(comp_layers=None, stop_at_layer=None):
        modelmade = model(timed=False,sync=False,cust=False,comp_arr=comp_layers).to('cuda').eval()
        times = []
        input_data = torch.rand(1, 3, 224, 224, device='cuda')
        for _ in range(10):
            start_time = time.perf_counter_ns()
            output = modelmade(input_data, stop_at_layer=stop_at_layer)
            end_time = time.perf_counter_ns()
            times.append(end_time - start_time)

        return np.median(times)

    dp = []
    path = []

    #fist case
    interp = measure_time([],1)
    comp = measure_time([0],1)
    dp.append([interp, comp]) # append(fastest path to each node)
    path.append([[], [0]])

    #A = interp, B = compile
    for i in tqdm(range(1, numlayers)):
        # Measure time for the four possible configurations
        aa = measure_time(path[i-1][0], stop_at_layer=i+1)  # prev: no compile, curr: no compile
        ab = measure_time(path[i-1][0] + [i], stop_at_layer=i+1)  # prev: no compile, curr: compile
        ba = measure_time(path[i-1][1], stop_at_layer=i+1)  # prev: compile, curr: no compile
        bb = measure_time(path[i-1][1] + [i], stop_at_layer=i+1)  # prev: compile, curr: compile

        # Determine the best paths
        if aa <= ba:
            best_path_to_A = path[i-1][0]  # best path to not compile current layer
            best_time_A = aa
        else:
            best_path_to_A = path[i-1][1]  # best path to not compile current layer
            best_time_A = ba

        if ab <= bb:
            best_path_to_B = path[i-1][0] + [i]  # best path to compile current layer
            best_time_B = ab
        else:
            best_path_to_B = path[i-1][1] + [i]  # best path to compile current layer
            best_time_B = bb

        # Append the best paths and times
        dp.append([best_time_A, best_time_B])
        path.append([best_path_to_A, best_path_to_B])

    if dp[numlayers-1][0] <= dp[numlayers-1][1]:
        optimal_path = path[numlayers-1][0]
    else:
        optimal_path = path[numlayers-1][1]
    return optimal_path

def run_configs():
    input_data = torch.rand(1, 3, 224, 224, device='cuda')
    for model,config in zip(VISION_MODELS,VISION_CONFIGS):
        default = model(timed=False,sync=False,cust=False,comp_arr=[]).to('cuda').eval()
        comp = torch.compile(model(timed=False,sync=False,cust=False,comp_arr=[])).to('cuda').eval()
        cust = model(timed=False,sync=False,cust=False,comp_arr=[0]).to('cuda').eval()

        defaults = []
        comps = []
        custs = []

        for _ in range(10): 
            st = time.perf_counter_ns()
            default(input_data)
            et = time.perf_counter_ns()
            defaults.append(et-st)
        default_rt = np.median(defaults)

        for _ in range(10): 
            st = time.perf_counter_ns()
            comp(input_data)
            et = time.perf_counter_ns()
            comps.append(et-st)
        comp_rt = np.median(comps)

        for _ in range(10): 
            st = time.perf_counter_ns()
            cust(input_data)
            et = time.perf_counter_ns()
            custs.append(et-st)
        cust_rt = np.median(custs)

        print(default_rt,comp_rt,cust_rt)