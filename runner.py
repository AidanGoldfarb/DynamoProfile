from util import *
import torch

def _run(model,input_data,profiler,reps):
    for _ in range(reps):
        profiler.clear_layer_times()
        with torch.no_grad():
            model(input_data)
    pickle_lst(profiler.get_layer_times(),profiler.metadata)

"""
    Runs a model
"""
def run(model_config,profile=True):
    model,input_data,profiler = model_config
    metadata = profiler.metadata
    print(f"Running {metadata}...",end='', flush=True)
    if is_cached(metadata):
        print("found in cache...done!",flush=True)
        return unpickle_lst(metadata)
    else:
        _run(model,input_data,profiler,reps=1)
        print("...done!",flush=True)