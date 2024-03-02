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

def profile_all():
    profile(RESNET_MODELS_FILENAMES)
    profile(GOOGLENET_MODELS_FILENAMES)
    profile(DENSENET_MODELS_FILENAMES)
    profile(SQUEEZENET_MODELS_FILENAMES)
    profile(ALEXNET_MODELS_FILENAMES)
    profile(MOBILENET_MODELS_FILENAMES)

#TODO 
def main():
    #run_all(verbose=True)
    profile_all()
if __name__ == "__main__":
    main()