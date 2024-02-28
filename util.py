import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #for googlenet

import torchvision.models as models
import numpy as np
import pandas as pd

"""
    Models
"""
MODELS = [models.resnet50,models.googlenet,models.densenet121,models.squeezenet1_1,models.alexnet,models.mobilenet_v2]
""""""

"""
    Model paths
    [Interp (Python), Compiled (CPU), GPU (CUDA), GPU_COMP (Triton)]
"""
INTERP = 0
COMPILED = 1
GPU = 2
TRITON = 3
RESNET_MODELS_FILENAMES = ["resnet50_default_interp.pkl","resnet50_default_compiled.pkl","resnet50_default_gpu.pkl","resnet50_default_gpu_comp.pkl"]
GOOGLENET_MODELS_FILENAMES = ["googlenet_default_interp.pkl","googlenet_default_compiled.pkl","googlenet_default_gpu.pkl","googlenet_default_gpu_comp.pkl"]
DENSENET_MODELS_FILENAMES = ["densenet_default_interp.pkl","densenet_default_compiled.pkl","densenet_default_gpu.pkl","densenet_default_gpu_comp.pkl"]
SQUEEZENET_MODELS_FILENAMES = ["squeezenet1_1_default_interp.pkl","squeezenet1_1_default_compiled.pkl","squeezenet1_1_default_gpu.pkl","squeezenet1_1_default_gpu_comp.pkl"]
ALEXNET_MODELS_FILENAMES = ["alexnet_default_interp.pkl","alexnet_default_compiled.pkl","alexnet_default_gpu.pkl","alexnet_default_gpu_comp.pkl"]
MOBILENET_MODELS_FILENAMES = ["mobilenet_v2_default_interp.pkl","mobilenet_v2_default_compiled.pkl","mobilenet_v2_default_gpu.pkl","mobilenet_v2_default_gpu_comp.pkl"]
""""""


DATA_DTYPE = np.dtype([('Type', 'U16000'), ('Time', float)])
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'
ENDC = '\033[0m'
DIR = "/data/agoldf6/DynamoBench/"

def pickle_lst(lst,filename):
    assert type(lst) is np.ndarray
    assert type(lst[0][0]) is np.str_
    assert type(lst[0][1]) is np.float64
    with open(os.path.join(DIR, "cache/"+filename), 'wb') as f:
        pickle.dump(lst, f)

def unpickle_lst(filename):
    with open(os.path.join(DIR, "cache/"+filename), 'rb') as f:
        return pickle.load(f)

def is_cached(model_name,mode="default"):
    for file in os.listdir(DIR+"cache/"):
        if model_name+"_"+mode in str(file):
            return True
    return False

def diff(f0,f1):
    if f0 == 0 and f1 == 0:
        return False 
    percentagedifference = abs((float(f0) - f1) / ((f0 + f1) / 2)) * 100
    return percentagedifference #> threshold

#pickle file to np array
def pickle_to_np(filename):
    return np.array(unpickle_lst(filename+".pkl"),dtype=DATA_DTYPE)

#pickle file to pd dataframe, layers numbered
def pickle_to_df(filename):
    arr = pickle_to_np(filename)
    count = 0

    for i,_ in enumerate(arr):
        arr0[i][0] = arr0[i][0]+f"_{count}" 
        count+=1

    return pd.DataFrame(arr)

def gen_metadata(model,compiled,gpu,mode):
    metadata = model.__class__.__name__ + "_"+mode
    if not compiled and not gpu:
        metadata += "_interp"
    elif compiled and not gpu:
        metadata += "_compiled"
    elif not compiled and gpu:
        metadata += "_gpu"
    else:
        metadata += "_gpu_comp"
    return metadata.lower()
