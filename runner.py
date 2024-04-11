from util import *
import torch
from torch.profiler import ProfilerActivity


# class ModelProfiler():

#     def __init__(modelname):
#         self.model = None
    
#     def run(mode,sync):
#         match mode,sync:
#             case "timer", False:
#                 pass
#             case "timer", True: 
#                 pass
#             case "profiler", False:
#                 pass
#             case "profiler", True:
#                 pass
#             case _:
#                 raise NotImplemented

def run_timed(model,input_data,config,reps,layers=1):
    outs = []
    rts = []
    with torch.no_grad():
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = model(input_data)
            et = time.perf_counter_ns()
            outs.append(out)
            rts.append(et-st)
        dct[f'{config}_arr'] = np.median([e[-1] for e in outs])
        dct[f'{config}_e2e'] = np.median(rts)

def run_profiled(model,input_data,config,reps,layers=1):    
    with torch.no_grad():
        for _ in range(reps):
            with torch.autograd.profiler.profile(
                use_cuda=True,
                record_shapes=True, 
                profile_memory=False, 
                with_stack=True
            ) as prof:
                for _ in range(reps):
                    model(input_data)
    pickle_obj(prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total'), f"{config}")
