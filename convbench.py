import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure your input tensor and weights are on the GPU for CUDA and Triton execution
device = torch.device('cuda')

# Define the function to be compiled and benchmarked
def conv2d_func(input_tensor, weight, stride, padding):
    return F.conv2d(input_tensor, weight, stride=stride, padding=padding)

# Pre-compile the function for Triton outside the timing loop
compiled_func_triton = torch.compile(conv2d_func)

# Benchmarking function
def benchmark_conv2d(input_tensor, weight, stride, padding, func):
    torch.cuda.synchronize()  # Ensure CUDA operations have completed
    start_time = time.time()
    output = func(input_tensor, weight, stride, padding)
    torch.cuda.synchronize()  # Ensure CUDA operations have completed
    end_time = time.time()
    return end_time - start_time

# Input tensor
input_tensor = torch.randn(1, 3, 224, 224, device=device)

# Expanding the list of configurations
configs = []
kernel_sizes = [1, 3, 5]
strides = [1, 2]
paddings = [0, 1, 2]

for ks in kernel_sizes:
    for stride in strides:
        for padding in paddings:
            configs.append({'kernel_size': ks, 'stride': stride, 'padding': padding})

# Initialize lists to hold the results
cuda_times, triton_times, config_labels = [], [], []

# Running benchmarks
for config in configs:
    weight = torch.randn(64, 3, config['kernel_size'], config['kernel_size'], device=device)
    # Define the function for CUDA (non-compiled)
    func_cuda = lambda input_tensor, weight, stride, padding: conv2d_func(input_tensor, weight, stride, padding)
    
    times_cuda = [benchmark_conv2d(input_tensor, weight, config['stride'], config['padding'], func_cuda) for _ in range(5)]
    times_triton = [benchmark_conv2d(input_tensor, weight, config['stride'], config['padding'], compiled_func_triton) for _ in range(5)]
    
    cuda_avg_time = np.mean(times_cuda)
    triton_avg_time = np.mean(times_triton)
    
    cuda_times.append(cuda_avg_time)
    triton_times.append(triton_avg_time)
    config_labels.append(f"KS{config['kernel_size']}_S{config['stride']}_P{config['padding']}")

# Plotting the results
plt.figure(figsize=(15, 10))
index = np.arange(len(config_labels))
bar_width = 0.35

plt.bar(index, cuda_times, bar_width, label='CUDA')
plt.bar(index + bar_width, triton_times, bar_width, label='Triton')

plt.xlabel('Configurations')
plt.ylabel('Average Execution Time (s)')
plt.title('Conv2D Execution Time: CUDA vs Triton')
plt.xticks(index + bar_width / 2, config_labels, rotation=90)
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.savefig("convfig")