
"""
    Runs a model
    
"""
def run(model_config,profile=True):
    model,input_data,metadata = model_config
    print(metadata)
    # is_cached(model.name)
    # for _ in range(3):
    #     with torch.no_grad():
    #         model(input_data)