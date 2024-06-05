import torch
from tqdm import tqdm
import torchvision.models as models
from util import *
from prepare_model_and_data import prepare_model
from runner import run_timed,run_profiled
from profiler import *
from plotter import *

# print(torch.__version__) #2.2.0+cu121
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.cache_size_limit = 8

def run_one(modelnm,rounds=10,reps=5,layers=1):
    for reg,cust,pure in zip(VISION_MODELS,CUSTOM_VISION_MODELS,PURE_VISION_MODELS):
        model = reg().to('cuda').eval()     
        modelcomp = torch.compile(model)
        mymodel = cust().to('cuda').eval()
        
        pmodel = pure().to('cuda').eval()
        pmodelcomp = torch.compile(pmodel)
        modelname = model.__class__.__name__.lower()
        if modelnm not in modelname:
            continue
        
        input_data = torch.rand(1, 3, 224, 224, device='cuda')
        dct = {}
        fast_layers = []
        with torch.no_grad():
            for _ in range(rounds):
                for _ in range(reps):
                    st = time.perf_counter_ns()
                    for _ in range(layers):
                        out = model(input_data)
                    et = time.perf_counter_ns()
                dct['cuda_arr'] = out[-1]
                dct['cuda_e2e'] = et-st

                #Triton
                for _ in range(reps):
                    st = time.perf_counter_ns()
                    for _ in range(layers):
                        out = modelcomp(input_data)
                    et = time.perf_counter_ns()
                dct['triton_arr'] = out[-1]
                dct['triton_e2e'] = et-st
                fast_layers.append(spedup_layers(dct))
        for e in fast_layers:
            print(e)

#wall clock time
def raw_run_all(reps=5, layers=1):
    """
        pure: Unsynced, Untimed
        timed: Unsynced, Timed
        sync: Synced, Untimed
        timed_sync: Synced, Timed
        
    """
    input_data = torch.rand(1, 3, 224, 224, device='cuda')
    for model in VISION_MODELS:
        pure = model(timed=False,sync=False,cust=False).to('cuda').eval()
        modelname = pure.__class__.__name__.lower()
        # if 'res' in modelname:# or 'dense' in modelname or 'squeeze' in modelname:
        #     continue 
        timed = model(timed=True,sync=False,cust=False).to('cuda').eval()
        sync = model(timed=False,sync=True,cust=False).to('cuda').eval()
        timed_sync = model(timed=True,sync=True,cust=False).to('cuda').eval()

        pureComp = torch.compile(pure)
        timedComp = torch.compile(timed)
        syncComp = torch.compile(sync)
        timed_syncComp = torch.compile(timed_sync)

        # pure_cust = model(timed=False,sync=False,cust=True)
        # timed_cust = model(timed=True,sync=False,cust=True)
        # sync_cust = model(timed=False,sync=True,cust=True)
        # timed_sync_cust = model(timed=True,sync=True,cust=True)

        modelname = pure.__class__.__name__.lower()
        #print(modelname,flush=True)

        #CUDA 
        run_profiled(pure,input_data,f"{modelname}_pure_cuda_prof",reps,layers)
        run_profiled(timed,input_data,f"{modelname}_timed_cuda_prof",reps,layers)
        run_profiled(sync,input_data,f"{modelname}_sync_cuda_prof",reps,layers)
        run_profiled(timed_sync,input_data,f"{modelname}_timedsync_cuda_prof",reps,layers)
        

        #Triton
        run_profiled(pureComp,input_data,f"{modelname}_pure_triton_prof",reps,layers)
        run_profiled(timedComp,input_data,f"{modelname}_timed_triton_prof",reps,layers)
        run_profiled(syncComp,input_data,f"{modelname}_sync_triton_prof",reps,layers)
        run_profiled(timed_syncComp,input_data,f"{modelname}_timedsync_triton_prof",reps,layers)


        #CUDAnoprof
        run_profiled(pure,input_data,f"{modelname}_pure_cuda_e2e",reps,layers,profile=False)
        run_profiled(timed,input_data,f"{modelname}_timed_cuda_e2e",reps,layers,profile=False)
        run_profiled(sync,input_data,f"{modelname}_sync_cuda_e2e",reps,layers,profile=False)
        run_profiled(timed_sync,input_data,f"{modelname}_timedsync_cuda_e2e",reps,layers,profile=False)
        

        #Tritonnoprof
        run_profiled(pureComp,input_data,f"{modelname}_pure_triton_e2e",reps,layers,profile=False)
        run_profiled(timedComp,input_data,f"{modelname}_timed_triton_e2e",reps,layers,profile=False)
        run_profiled(syncComp,input_data,f"{modelname}_sync_triton_e2e",reps,layers,profile=False)
        run_profiled(timed_syncComp,input_data,f"{modelname}_timedsync_triton_e2e",reps,layers,profile=False)

def run_all(verbose,device):
    #Vision
    for model in tqdm(VISION_MODELS,disable=True):#not verbose):
        #autograd profiler
        for model_config in prepare_model(model,hooks=False,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

        #custom profiler
        # for model_config in prepare_model(model,hooks=True,device=device,verbose=verbose):
        #     run(model_config,profile=True,verbose=verbose)
    
    #Custom vision
    for model in tqdm(CUSTOM_VISION_MODELS,disable=True):
        for model_config in prepare_model(model,hooks=False,device=device,verbose=verbose):
            run(model_config,profile=True,verbose=verbose)

        # for model_config in prepare_model(model,hooks=True,device=device,verbose=verbose):
        #     run(model_config,profile=True,verbose=verbose)

def find_config_dyn():
    for model in VISION_MODELS:
        if 'google' not in str(model):
            continue
        num_layers = len(model(False,False,False,[]).features)
        res = find_bestconfig_dyn(model,num_layers)
        print(res)
   

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

#TODO check out https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py#L1224
# maybe there is a simple test that can be run here with 'example_inputs'
def main():
    #raw_run_all()
    #profile_autogradtraces()
    find_config_dyn()
    #compare_runtimes()
    #run_all(device='gpu',verbose=True)
    #profile_all_hooktraces()
    #compare_cust_configs()
    #profile_autogradtraces()
    #run_oracle_config()
if __name__ == "__main__":
    main()