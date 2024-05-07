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

def run_profiled(model,input_data,config,reps,layers=1,profile=True): 
    print(f'\trunning {config}...',end='',flush=True)
    if is_cached(config):
        print('...in cache')
        return
    model(input_data)

    e_to_es = []
    dfs = []
    arrs = []
    with torch.no_grad():
        if profile:
            for _ in tqdm(range(reps)):
                with torch.autograd.profiler.profile(
                    use_cuda=True,
                    record_shapes=True, 
                    profile_memory=False, 
                    with_stack=True
                ) as prof:
                    st = time.perf_counter_ns()
                    _,arr = model(input_data)
                    et = time.perf_counter_ns()
                    arrs.append(arr)
                    e_to_e = et-st
                    e_to_es.append(e_to_e)
                data = prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total')
                df = trace_to_df(data)
                dfs.append(df)
        else:
            for _ in range(reps):
                st = time.perf_counter_ns()
                _,arr = model(input_data)
                et = time.perf_counter_ns()
                e_to_es.append(et-st)
                arrs.append(arr)

    df = None
    arr = None
    
    if len(dfs) > 0 and dfs[0] is not None:
        df = dfs[-1]#median_df(dfs)
    if len(arrs) > 0 and arrs[0] is not None:
        arr = arrs[-1]#median_arr(arrs)
    
    assert len(e_to_es) == 5
    if profile:
        assert df is not None
    
    ete = median_ete(e_to_es)
    
    pickle_obj((df,arr,ete), f"{config}")
    print('...done')

def median_df(dataframes, column_name="Self_CUDA_us"):
    sums = [df[column_name].sum() for df in dataframes]
    sorted_sums = sorted(sums)

    mid_index = len(sorted_sums) // 2
    if len(sorted_sums) % 2 == 0:
        median_sum = (sorted_sums[mid_index - 1] + sorted_sums[mid_index]) / 2.0
    else:
        median_sum = sorted_sums[mid_index]

    closest = min(sums, key=lambda x: abs(x - median_sum))
    median_df_index = sums.index(closest)

    return dataframes[median_df_index]

def median_arr(lists):
    sums = [sum(sublist) for sublist in lists]
    sorted_sums = sorted(sums)

    mid_index = len(sorted_sums) // 2
    if len(sorted_sums) % 2 == 0:
        median_sum = (sorted_sums[mid_index - 1] + sorted_sums[mid_index]) / 2.0
    else:
        median_sum = sorted_sums[mid_index]

    closest = min(sums, key=lambda x: abs(x - median_sum))
    median_list_index = sums.index(closest)

    return lists[median_list_index]

def median_ete(etes):
    return np.median(etes)