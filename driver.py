import torch
from tqdm import tqdm
import torchvision.models as models
from util import *
from prepare_model_and_data import prepare_model
from runner import run
from profiler import *
from plotter import *

# print(torch.__version__) #2.2.0+cu121
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.cache_size_limit = 8

def run_all(verbose,device):
    #Vision
    for model in tqdm(VISION_MODELS,disable=True):#not verbose):
        #autograd profiler
        for model_config in prepare_model(model,hooks=False,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

        #custom profiler
        for model_config in prepare_model(model,hooks=True,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)
    
    #Custom vision
    for model in tqdm(CUSTOM_VISION_MODELS,disable=True):
        for model_config in prepare_model(model,hooks=False,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

        for model_config in prepare_model(model,hooks=True,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

def profile_all_hooktraces():
    profile_hooktraces(RESNET_MODELS_FILENAMES)
    profile_hooktraces(GOOGLENET_MODELS_FILENAMES)
    profile_hooktraces(DENSENET_MODELS_FILENAMES)
    profile_hooktraces(SQUEEZENET_MODELS_FILENAMES)
    profile_hooktraces(ALEXNET_MODELS_FILENAMES)
    profile_hooktraces(MOBILENET_MODELS_FILENAMES)

def profile_all_autgradtraces(verbose=False):
    find_slow_layers()
    # profile_autogradtraces(RESNET_MODELS_FILENAMES,verbose=verbose)
    # profile_autogradtraces(GOOGLENET_MODELS_FILENAMES,verbose=verbose)
    # profile_autogradtraces(DENSENET_MODELS_FILENAMES,verbose=verbose)
    # profile_autogradtraces(SQUEEZENET_MODELS_FILENAMES,verbose=verbose)
    # profile_autogradtraces(ALEXNET_MODELS_FILENAMES,verbose=verbose)
    # profile_autogradtraces(MOBILENET_MODELS_FILENAMES,verbose=verbose)

def run_custom_resnet():
    # res = prepare_model(models.resnet50,hooks=False)[3] # triton compiled
    # myres = prepare_model(models.myresnet50,hooks=False)[2] #[0] for uncompiled, as we will do manual compilation
    # run(res,profile=True,verbose=True)
    # run(myres,profile=True,verbose=True)

    res = parse_autograd_json("resnet_default_triton_nohooks.trace")
    myres = parse_autograd_json("myresnet_default_gpu_nohooks.trace")

    df = merge_frames([myres,res])
    df = add_speedup(reorder_cols(df))
    #df = df.loc[df['Layer'].str.contains('conv')]
    df = df.sort_values(by="Speedup",ascending=True)
    df = df.loc[df['Speedup'] != 0]
    df = df.dropna(axis=0,subset=['Speedup'])
    # print(df.to_string())
    custom_time = 0
    triton_time = 0
    for i,series in df.iterrows():
        custom_time += series['Time_myresnet_gpu_default']
        triton_time += series['Time_resnet_triton_default']
    print(f"custom_time    {custom_time:.4}")
    print(f"triton_time    {triton_time:.4}")

#TODO 
def main():
    #run_all(device='gpu',verbose=True)
    #profile_all_hooktraces()
    compare_cust_configs()
if __name__ == "__main__":
    main()