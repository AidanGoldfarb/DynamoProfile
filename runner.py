from util import *
import torch

def _run(model,input_data,profiler,reps):
    use_cuda = "gpu" in profiler.metadata
    autograd = "nohooks" in profiler.metadata

    for _ in range(reps):
        with torch.no_grad():
            if autograd:
                with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
                    model(input_data)
                profiler.prof = prof
            else:
                profiler.clear_layer_times()
                model(input_data)

"""
    Runs a model
"""
def run(model_config,profile=True,verbose=False):
    model,input_data,profiler = model_config
    metadata = profiler.metadata
    if verbose:
        print(f"Running {metadata}...",end='', flush=True)
    if is_cached(metadata):
        if verbose:
            print("found in cache...done!",flush=True)
        return unpickle_lst(metadata)
    else:
        _run(model,input_data,profiler,reps=5)
        if verbose:
            print("...done!",flush=True)