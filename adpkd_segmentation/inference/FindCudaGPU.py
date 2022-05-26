import subprocess  # This is found on RadDeep
import torch  # This is also found on RadDeep
"""
My Current solution for the RadDeep memory Problem. Found on ADPKD-SEGMENTATION PYTORCH

Inputs:
    min_mem: The specified minimum memory requirement of the computation. It is best to have an estimate.
             You can always set min_mem = 1 if this is not a major issue.
    
    gpu_prefix: This is going to be a hard-coded parameter for torch, and I have not thought about this
                implementation for tensorflow. 
 
This is a modified code from 'gpu_utils.py'. The function will identify valid CUDA GPUs (nVidia) and sort 
memory usage by least to most used. Then, the algorithm will look at the minimum specified acceptable
GPU memory and remove all GPUs that do not satisfy this condition. The function will pick the most
'available' GPU id of the available types in the array.
    Reason for Modification: It did not look like I could assign a minimum acceptable available GPU memory...
    ... value to the original code. I wanted this minimum since this can act as an extra check to ensure...
    ... that available GPUs would have the necessary memory from our estimates. Speaking of which, I will...
    ... need Kurt to confirm what the minimum accepted estimate for the RadDeep inference is. From the...
    ... last conversation I had with him, I believe it is ~4GiB. I would hope to use this function in...
    ... the following way:  

#### START EXAMPLE ####    
from FindCudaGPU import find_gpu


minMemoryReq = 4*1024  # 4 GiB to MiB, this is hard-coded in and I will need...
# ...Kurt to confirm the minimum memory usage. Also, if there is a way or us...
# ...to not hard code this in, that would be absolutely fantastic.
gpu_prefix = "cuda"
      
device_string = find_gpu(minMemoryReq, gpu_prefix)
.
.
.
device = torch.device(device_string)
#### END EXAMPLE ####
"""


def find_gpu(min_mem, gpu_prefix):
    device_name = "cpu"  # Default device setting for our torch program. We may want to implement...
    # ...error codes in the case that no computational space is available anywhere.
    gpu_num = torch.cuda.device_count()  # Number of cuda devices on the server.
    print('Number of cuda devices: ', str(gpu_num), '\n')
    if gpu_num >= 1:
        print('Finding available CUDA devices for processing (FREE MEMORY >= ', str(min_mem), ' MegaBytes minimum)...\n')
        result_total = subprocess.run('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits'.split(' '),
                                      stdout=subprocess.PIPE)  # Command that pulls the total memory of each GPU
        temp_max = result_total.stdout.splitlines()
        lines_max_mem = [int(line.decode('ascii')) for line in temp_max]  # ...
        # ...get total (max) memory (integer) list, index is gpu number
        result_used = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'.split(' '),
                                     stdout=subprocess.PIPE)  # Pulls used memory of each GPU
        temp_used = result_used.stdout.split()
        lines_used_mem = [int(line.decode('ascii')) for line in temp_used]  # get memory usage list, index is gpu number
        available_gpus = sorted(range(len(lines_used_mem)), key=lines_used_mem.__getitem__)  # sort GPU...
        # ...indices by lowest memory usage
        free_mem = []
        free_gpus = []
        for ind in range(0, len(available_gpus)):
            temp_free = lines_max_mem[available_gpus[ind]] - lines_used_mem[available_gpus[ind]]  # This is the free...
            # ...GPU memory for each GPU. If I knew how sorting algorithms worked better, I would not need this.
            # print('Iteration ', str(ind), 'GPU: ', str(available_gpus[ind]))
            # print('Free Memory: ', str(temp_free),'\n')
            if temp_free >= min_mem:
                free_mem.append(temp_free)
                free_gpus.append(available_gpus[ind])
                # if len(free_gpus) == 1:
                	# temp_counter = 1
                	
                # if (len(free_gpus) <= 4):
                	# print('Free GPU found. Will read out the top four GPUs (', str(temp_counter), '/4)')
                	# print('GPU ', str(free_gpus[ind]), ': ', str(free_mem[ind]), ' MiB\n')        	
                	# temp = temp_counter
                	# temp_counter = temp + 1                         
            
            elif temp_free < min_mem:
                print('GPU ', str(available_gpus[ind]), ' memory (', str(temp_free), ' MiB) < ',
                      ' minimum acceptable memory (', str(min_mem),
                      ' MiB). This GPU is unavailable. Removing from list.\n')        

        if len(free_gpus) == 1:
            print('1 Free GPU (id: ', str(free_gpus), ', Available Memory: ', str(free_mem),
                  ') is found. Setting the device to "cuda', str(free_gpus[0]), '"')
            device_name = gpu_prefix + str(free_gpus[0])
        elif len(free_gpus) > 1:
            print(str(len(free_gpus)),
                  ' free GPUs found. Setting the device to the largest available memory bank (id: ',
                  str(free_gpus[0]), ', Available Memory: ', str(free_mem[0]), ' MiB)')
            device_name = gpu_prefix + str(free_gpus[0])  # We are able to do this because we have sorted
        elif len(free_gpus) < 1 or len(free_gpus) == []:
            print('No Free GPU identified. Switching to cpu')
            device_name = "cpu"
    elif(gpu_num < 1) or (gpu_num == []):
        print('Warning: No cuda GPU identified on this system. Switching to CPU')
        device_name = "cpu"

    return device_name
