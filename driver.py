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

def run_all(verbose):
    for model in tqdm(MODELS,disable=True):#not verbose):
        #autograd profiler
        for model_config in prepare_model(model,hooks=False,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

        #custom profiler
        for model_config in prepare_model(model,hooks=True,verbose=verbose):
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

#TODO We can merge all the dataframes, but the convs are not the same shape!!!
def main():
    #run_all(verbose=False)
    #profile_all_hooktraces()
    profile_all_autgradtraces(verbose=False)
if __name__ == "__main__":
    main()