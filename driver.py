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

#wall clock time
def raw_run_all(reps=5):
    input_data = torch.rand(1, 3, 224, 224, device='cuda')
    run_data = {}

    for reg,cust,pure in zip(VISION_MODELS,CUSTOM_VISION_MODELS,PURE_VISION_MODELS):
        model = reg().to('cuda').eval()     
        modelcomp = torch.compile(model)
        mymodel = cust().to('cuda').eval()
        
        pmodel = pure().to('cuda').eval()
        pmodelcomp = torch.compile(pmodel)
        modelname = model.__class__.__name__.lower()

        dct = {}

        #CUDA Reg
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = model(input_data)
            et = time.perf_counter_ns()
        dct['cuda_arr'] = out[-1]
        dct['cuda_e2e'] = et-st

        #Triton
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = modelcomp(input_data)
            et = time.perf_counter_ns()
        dct['triton_arr'] = out[-1]
        dct['triton_e2e'] = et-st

        #Custom
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = mymodel(input_data)
            et = time.perf_counter_ns()
        dct['cust_arr'] = out[-1]
        dct['cust_e2e'] = et-st

        #CUDA Pure
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = pmodel(input_data)
            et = time.perf_counter_ns()
        dct['cuda_pure_e2e'] = et-st

        #TRITON Pure
        for _ in range(reps):
            st = time.perf_counter_ns()
            out = pmodelcomp(input_data)
            et = time.perf_counter_ns()
        dct['triton_pure_e2e'] = et-st

        run_data[modelname] = dct
    
    
    for k,v in run_data.items():
        print(k,v)
    #return run_data

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
    

#TODO Check out graphs in figs, cust should be optimal
def main():
    raw_run_all()
    #compare_runtimes()
    #run_all(device='gpu',verbose=True)
    #profile_all_hooktraces()
    #compare_cust_configs()
    #profile_autogradtraces()
    #run_oracle_config()
if __name__ == "__main__":
    main()