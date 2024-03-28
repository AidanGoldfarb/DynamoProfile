from util import *
import torch

def _run(model,input_data,profiler,reps):
    model.eval()
    autograd = "nohooks" in profiler.metadata
    for _ in range(reps):
        with torch.no_grad():
            if autograd:
                with torch.autograd.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as prof:
                    model(input_data)
                profiler.prof = prof
            else:
                profiler.clear_layer_times()
                model(input_data)
    if autograd:
        profiler.prof.export_stacks(f"cache/autogradtraces/{profiler.metadata}.stack",metric='self_cuda_time_total')
        profiler.prof.export_chrome_trace(f"cache/autogradtraces/{profiler.metadata}.trace")
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
