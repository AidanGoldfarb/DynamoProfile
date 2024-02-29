from util import *
import torch

def _run(model,input_data,profiler,reps):
    use_cuda = "gpu" in profiler.metadata
    autograd = "nohooks" in profiler.metadata
    print(profiler.metadata)

    for _ in range(reps):
        with torch.no_grad():
            if autograd:
                with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
                    model(input_data)
                profiler.prof = prof
            else:
                profiler.clear_layer_times()
                model(input_data)
    if autograd:
        #profiler.prof.export_chrome_trace(f"cache/autogradtraces/{profiler.metadata}.trace")
        pickle_obj(profiler.prof.key_averages(),profiler.metadata)
    else:
        pickle_obj(profiler.get_layer_times(),profiler.metadata)

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
        return unpickle_obj(metadata)
    else:
        _run(model,input_data,profiler,reps=5)
        if verbose:
            print("...done!",flush=True)