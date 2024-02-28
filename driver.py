import torch
import torchvision.models as models
from prepare_model_and_data import *
from runner import run

#print(torch.__version__) #2.2.0+cu121
#torch._dynamo.config.suppress_errors = True

torch._dynamo.config.cache_size_limit = 8
layer_times = []

def _run_and_profile():
    global layer_times
    layer_times = []

    

    return np.array(layer_times,dtype=DATA_DTYPE)

    # Plot data on each subplot
    axs[0].plot(x_values, y_values_interp, color='b', label='Interpreted')
    axs[1].plot(x_values, y_values_compiled, color='g', label='Compiled')
    axs[2].plot(x_values, y_values_gpu, color='r', label='GPU')
    axs[3].plot(x_values, y_values_gpu_comp, color='k', label='GPU_and_Compiled')

    for ax in axs:
        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Runtime [sec]', fontweight='bold')
        ax.set_yscale('log')

    axs[-1].plot(x_values, y_values_interp, color='b', label='Interpreted')
    axs[-1].plot(x_values, y_values_compiled, color='g', label='Compiled')
    axs[-1].plot(x_values, y_values_gpu, color='r', label='GPU')
    axs[-1].plot(x_values, y_values_gpu_comp, color='k', label='GPU_and_Compiled')
    axs[-1].legend()
    axs[-1].set_xlabel('Layer', fontweight='bold')
    axs[-1].set_ylabel('Runtime [sec]', fontweight='bold')
    axs[-1].set_yscale('log')

    for ax in axs[:-1]:
        ax.legend()
    
    plt.savefig("/data/agoldf6/DynamoBench/benches/figs/" + filename, bbox_inches='tight')

def run_all(model_constr, mode="default"):
    model_name = model_constr.__name__.lower()

    if not torch.cuda.is_available():
        pass
        # print("no gpu device")
        # raise Exception("no gpu")
    if not is_cached(model_name,mode):
        print(model_name,"not cached, profiling...",end='')

        interp = _run_and_profile(model_constr(),mode=mode)
        compiled = _run_and_profile(model_constr(),compiled=True,mode=mode)
        gpu = _run_and_profile(model_constr(),gpu=True)
        gpu_comp = _run_and_profile(model_constr(),compiled=True,gpu=True,mode=mode)

        pickle_lst(interp,model_name+"_"+mode+"_interp.pkl")
        pickle_lst(compiled,model_name+"_"+mode+"_compiled.pkl")
        pickle_lst(gpu,model_name+"_"+mode+"_gpu.pkl")
        pickle_lst(gpu_comp,model_name+"_"+mode+"_gpu_comp.pkl")
        print("...done")
    else:
        print(model_name,"found in cache")
        interp = unpickle_lst(model_name+"_"+mode+"_interp.pkl")
        compiled = unpickle_lst(model_name+"_"+mode+"_compiled.pkl")
        gpu = unpickle_lst(model_name+"_"+mode+"_gpu.pkl")
        gpu_comp = unpickle_lst(model_name+"_"+mode+"_gpu_comp.pkl")
    return interp,compiled,gpu,gpu_comp
    
    """
        GPU vs GPU-compiled (Cuda vs Triton)
    """
    print("###########(Cuda vs Triton)####################")
    find_slow_layers_helper("resnet50_default_gpu","resnet50_default_gpu_comp")
    find_slow_layers_helper("googlenet_default_gpu","googlenet_default_gpu_comp")
    find_slow_layers_helper("densenet121_default_gpu","densenet121_default_gpu_comp")
    find_slow_layers_helper("squeezenet1_1_default_gpu","squeezenet1_1_default_gpu_comp")
    find_slow_layers_helper("alexnet_default_gpu","alexnet_default_gpu_comp")
    find_slow_layers_helper("mobilenet_v2_default_gpu","mobilenet_v2_default_gpu_comp")

    
    """
        CPU vs CPU-compiled (Interp vs C++)
    """
    print("\n###########(Interp vs C++)####################")
    find_slow_layers_helper("resnet50_default_interp","resnet50_default_compiled")
    find_slow_layers_helper("googlenet_default_interp","googlenet_default_compiled")
    find_slow_layers_helper("densenet121_default_interp","densenet121_default_compiled")
    find_slow_layers_helper("squeezenet1_1_default_interp","squeezenet1_1_default_compiled")
    find_slow_layers_helper("alexnet_default_interp","alexnet_default_compiled")
    find_slow_layers_helper("mobilenet_v2_default_interp","mobilenet_v2_default_compiled")

    """
    
    """
    print("\n###########(Interp vs GPU)####################")
    find_slow_layers_helper("resnet50_default_interp","resnet50_default_gpu")
    find_slow_layers_helper("googlenet_default_interp","googlenet_default_gpu")
    find_slow_layers_helper("densenet121_default_interp","densenet121_default_gpu")
    find_slow_layers_helper("squeezenet1_1_default_interp","squeezenet1_1_default_gpu")
    find_slow_layers_helper("alexnet_default_interp","alexnet_default_gpu")
    find_slow_layers_helper("mobilenet_v2_default_interp","mobilenet_v2_default_gpu")


def main():
    for model in MODELS:
        for model_config in prepare_model(model):
            run(model_config,profile=True)

if __name__ == "__main__":
    main()