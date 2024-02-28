from util import *

def profile(filenames):
    interp_df = pickle_to_df(filenames[INTERP])
    compiled_df = pickle_to_df(filenames[COMPILED])
    gpu_df = pickle_to_df(filenames[GPU])
    triton_df = pickle_to_df(filenames[TRITON])

    print(interp_df.head())