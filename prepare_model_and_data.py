from util import *
import torch

class ModelProfiler:
    def __init__(self,metadata):
        self.layer_times = []
        self.metadata = metadata
    
    def register_hooks(self,model):
        start_time = -1
        def pre_hook_fn(module, input):
            start_time = time.time()

        def forward_hook_fn(module, input, output):
            end_time = time.time()
            rt = end_time - start_time
            self.layer_times.append( (str(module), float(rt)) )

        for module in model.modules():
            module.register_forward_pre_hook(pre_hook_fn)
            module.register_forward_hook(forward_hook_fn)

    def get_layer_times(self):
        return np.array(self.layer_times, dtype=DATA_DTYPE)
    
    def clear_layer_times(self):
        self.layer_times = []


def _gen_model(model,compiled=False, gpu=False, mode="default"):
    profiler = ModelProfiler(gen_metadata(model,compiled,gpu,mode))    
    profiler.register_hooks(model)
    
    input_data = torch.rand(1, 3, 224, 224)

    if gpu:
        input_data = input_data.cuda()
        model = model.to("cuda")
    
    if compiled:
        model = torch.compile(model,mode=mode)

    return model,input_data,profiler

"""
    Given a modelname, prepares 4 models with different configuration
    TODO: add mode support. As of now just default, as few 
    differences were observed with other modes
"""
def prepare_model(model):
    return (
        _gen_model(model(), compiled=False, gpu=False), 
        _gen_model(model(), compiled=True,  gpu=False), 
        _gen_model(model(), compiled=False, gpu=True), 
        _gen_model(model(), compiled=True,  gpu=True)
    )

def prepare_all():
    pass
    