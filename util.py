import warnings,os,time,pickle,typing,json,re
warnings.simplefilter(action='ignore', category=FutureWarning) #for googlenet
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from io import StringIO
import numbers
import copy
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'


"""
    Models
"""
#Unsynced, Timed
VISION_MODELS = [models.resnet50,models.googlenet,models.densenet121,models.squeezenet1_1,models.alexnet,models.mobilenet_v2]

#Unsynced, Untimed
PURE_VISION_MODELS = [models.presnet50,models.pgooglenet,models.pdensenet121,models.psqueezenet1_1,models.palexnet,models.pmobilenet_v2]

#Synced, Timed
CUSTOM_VISION_MODELS = [models.myresnet50,models.mygooglenet,models.mydensenet121,models.mysqueezenet1_1,models.myalexnet,models.mymobilenet_v2]

#Synced, Untimed
SYNC_VISION_MODELS = [models.sresnet50,models.sgooglenet,models.sdensenet121,models.ssqueezenet1_1,models.salexnet,models.smobilenet_v2]

VISION_CONFIGS = [
    [3, 4, 5, 6, 7, 8, 10],
    [4, 5, 6, 7],
    [5, 6, 7, 8, 9, 10, 12, 14, 15],
    [0, 4, 5, 10],
    [3, 8],
    [4, 10],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
]
""""""

"""
    Model paths
    [Interp (Python), Compiled (CPU), GPU (CUDA), GPU_COMP (Triton)]
"""
INTERP = 0
COMPILED = 1
GPU = 2
TRITON = 3
CONFIG_MAP = {0:"interp",1:"cpp",2:"gpu",3:"triton"}
RESNET_MODELS_FILENAMES = ["resnet_default_interp","resnet_default_cpp","resnet_default_gpu","resnet_default_triton"]
MYRESNET_MODEL_FILENAMES = ["myresnet_default_interp", "resnet_default_gpu"]
GOOGLENET_MODELS_FILENAMES = ["googlenet_default_interp","googlenet_default_cpp","googlenet_default_gpu","googlenet_default_triton"]
DENSENET_MODELS_FILENAMES = ["densenet_default_interp","densenet_default_cpp","densenet_default_gpu","densenet_default_triton"]
SQUEEZENET_MODELS_FILENAMES = ["squeezenet_default_interp","squeezenet_default_cpp","squeezenet_default_gpu","squeezenet_default_triton"]
ALEXNET_MODELS_FILENAMES = ["alexnet_default_interp","alexnet_default_cpp","alexnet_default_gpu","alexnet_default_triton"]
MOBILENET_MODELS_FILENAMES = ["mobilenetv2_default_interp","mobilenetv2_default_cpp","mobilenetv2_default_gpu","mobilenetv2_default_triton"]
FILENAMES = {'resnet': RESNET_MODELS_FILENAMES,
            'googlenet': GOOGLENET_MODELS_FILENAMES,
            'densenet': DENSENET_MODELS_FILENAMES,
            'squeezenet': SQUEEZENET_MODELS_FILENAMES,
            'alexnet': ALEXNET_MODELS_FILENAMES,
            'mobilenetv2': MOBILENET_MODELS_FILENAMES}
""""""


DATA_DTYPE = np.dtype([('Type', 'U16000'), ('Time', float)])
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'
ENDC = '\033[0m'
#DIR = "/Users/aidangoldfarb/Projects/DynamoProfile/"
DIR = "/data/agoldf6/DynamoProfile/"

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def convert_to_microseconds(value):
    return float(str(value).replace('us', '').replace('%', '').replace('ms','').replace('s',''))

def trace_to_df(trace):
    trace = trace.replace('-','') #i hate you
    last_header = 'Input Shapes'
    mark = re.search(rf'{last_header}(\D)', trace)
    assert mark

    header_end_index = mark.start(1)

    header_part = trace[:header_end_index]
    data_part = trace[header_end_index:]
    modified_header = re.sub(r'(\S) (\S)', r'\1_\2', header_part)
    modified_data_str = modified_header + data_part

    df = pd.read_fwf(StringIO(modified_data_str))


    df.rename(columns={'Self_CPU': 'Self_CPU_us'},inplace=True)
    df.rename(columns={'CPU_total': 'CPU_total_us'},inplace=True)
    df.rename(columns={'CPU_time_avg': 'CPU_time_avg_us'},inplace=True)
    df.rename(columns={'Self_CUDA': 'Self_CUDA_us'},inplace=True)
    df.rename(columns={'CUDA_total': 'CUDA_total_us'},inplace=True)
    df.rename(columns={'CUDA_time_avg': 'CUDA_time_avg_us'},inplace=True)

    for c in df.columns:
        if 'us' in c:
            df[c] = df[c].apply(convert_to_microseconds)
        if '%' in c:
            df[c] = df[c].apply(convert_to_microseconds)

    return df

def pickle_obj(obj,filename):
    dr = ""
    if '_prof' in filename or "e2e" in filename:
        dr = "autogradtraces/"
    with open(os.path.join(DIR, f"cache/{dr}"+filename+".pkl"), 'wb') as f:
        pickle.dump(obj, f)

def unpickle_obj(filename):
    dr = ""
    with open(os.path.join(DIR, f"cache/{dr}"+filename), 'rb') as f:
        return pickle.load(f)

def is_cached(config):
    if "prof" in config:
        filename=config+".pkl"
        for file in os.listdir(DIR+"cache/autogradtraces/"):
            if filename in str(file):
                return True

def diff(f0,f1):
    if f0 == 0 and f1 == 0:
        return False 
    percentagedifference = abs((float(f0) - f1) / ((f0 + f1) / 2)) * 100
    return percentagedifference #> threshold

#given two df, returns a new df with the difference in 1st column as a new column
def add_speedup(df):
    df.insert(3,"Speedup", np.where(df.iloc[:,2] != 0, df.iloc[:,1] / df.iloc[:,2], 0))
    return df

"""
    Reorder df columns s.t. the order is [Layer, Time,...Time,...other]
"""
def reorder_cols(df):
    layer_col = ['Layer']
    time_cols = [col for col in df.columns if 'Time' in col]
    dim_col = [col for col in df.columns if 'args.Input Dims' in col]
    mem_cols = [col for col in df.columns if 'args.Total' in col]
    other_cols = [col for col in df.columns if col not in layer_col and col not in time_cols and col not in dim_col and col not in mem_cols]
    return df[layer_col+time_cols+dim_col+mem_cols+other_cols]

def condense_input_dims(df):
    id_name = None
    for col in df.columns:
        if "args.Input Dims" in col:
            id_name = col
    assert id_name is not None
    id = df[id_name]
    new_id = []
    for e in id:
        if isinstance(e,list):
            arr = np.concatenate([np.array(sl).flatten() for sl in e])
            new_id.append(arr.sum())
        else:
            new_id.append(-1)
    df[id_name] = new_id
    return df

def merge_frames(dfs):
    assert len(dfs) > 0
    res = dfs[0]
    for df in dfs[1:]:
        res = pd.merge(res,df, on='Layer')
        res = reorder_cols(res)
    return res

#pickle file to np array
def pickle_to_np(filename):
    if "_nohooks" in filename:
        return np.array(unpickle_obj(filename))
    else:
        return np.array(unpickle_obj(filename),dtype=DATA_DTYPE)

#pickle file to pd dataframe, layers numbered
def pickle_to_df(filename):
    arr = pickle_to_np(filename)
    if "_nohooks" in filename:
        return pd.DataFrame(arr)

    count = 0
    for i,_ in enumerate(arr):
        arr[i][0] = arr[i][0]+f"_{count}" 
        count+=1
    
    df = pd.DataFrame(arr)
    column_name = "Time_"+filename.split('_')[-1]
    df.rename(columns={'Type': 'Layer', 'Time': column_name}, inplace=True)
    return df

def gen_metadata(model,hooks,compiled,gpu,mode):
    metadata = model.__class__.__name__ + "_"+mode
    if not compiled and not gpu:
        metadata += "_interp"
    elif compiled and not gpu:
        metadata += "_cpp"
    elif not compiled and gpu:
        metadata += "_gpu"
    else:
        metadata += "_triton"
    if not hooks:
        metadata += "_nohooks"
    return metadata.lower()

def parse_autograd_json(filename):
    filename = DIR + "cache/autogradtraces/" + filename
    arr = filename.split("/")[-1].split("_")
    model = arr[0]
    mode = arr[1]
    device = arr[2]
    suffix = "_"+model+"_"+device+"_"+mode
    with open(filename,'r') as file:
        profiler_data = json.load(file)
        df = pd.json_normalize(profiler_data["traceEvents"])
        
        df.rename(columns={'name':'Layer','dur': "Time"}, inplace=True)
        df['Layer'] = df['Layer'].str.replace('^aten::', '', regex=True) 
        df['Layer_Count'] = df.groupby('Layer').cumcount() + 1
        df['Layer'] = df['Layer'] + '_' + df['Layer_Count'].astype(str)
        df.drop(columns=['Layer_Count'], inplace=True)
        df = df.add_suffix(suffix,axis=1)
        df.rename(columns={f'Layer{suffix}': 'Layer'}, inplace=True)
        df = condense_input_dims(df)

        return df#df,fdf

def df_filter(df,column,keep):
    return df[df[column].str.contains(keep, case=False)]


"""
    Timers
"""
def _call_w_arg(f,args):
    output = None
    if type(args) is tuple:
        start = time.perf_counter_ns()
        output = f(*args)
        tm = time.perf_counter_ns()-start
    elif type(args) is list or type(args) is np.ndarray or type(args) is torch.Tensor or callable(args) or isinstance(args, numbers.Number):
        start = time.perf_counter_ns()
        output = f(args)
        tm = time.perf_counter_ns()-start
    else:
        print(type(args))
        raise NotImplemented
    return tm,output

def _call_w_both(f,args,kwargs):
    output = None
    if type(args) is tuple:
        start = time.perf_counter_ns()
        output = f(*args,**kwargs)
        tm = time.perf_counter_ns()-start
    elif type(args) is list or type(args) is np.ndarray or callable(args) or isinstance(args, numbers.Number):
        start = time.perf_counter_ns()
        output = f(args,**kwargs)
        tm = time.perf_counter_ns()-start
    else:
        print(type(args),type(kwargs))
        raise NotImplemented
    return tm,output

def timeit(f, args=None, kwargs=None, keepRes=False) -> float:
    output = None
    tm = -1
    match args,kwargs:
        case None,None: 
            start = time.perf_counter_ns()
            output = f()
            tm = time.perf_counter_ns()-start
        case arg,None: 
            tm,output = _call_w_arg(f,arg)
        case None,kwarg: 
            raise NotImplemented("timeit: just kwargs??")
        case arg,kwarg: 
            tm,output = _call_w_both(f,arg,kwarg)
        case _: raise NotImplemented("timeit notimplemented")
    if keepRes:
        return tm, output
    else:
        return tm


