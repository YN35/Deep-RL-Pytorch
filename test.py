import torch

print(torch.__version__)

print(torch.cuda.is_available())

import torch
import numpy as np
import random
import time


torch_tensor = torch.rand(16 * 100000).cuda()
numpy_array = np.random.rand(16 * 100000)
print(random.randint(-torch_tensor.shape[0], torch_tensor.shape[0]))
start = time.time()
rolled = torch.roll(torch_tensor, random.randint(-torch_tensor.shape[0], torch_tensor.shape[0]))
print(time.time() - start)

start = time.time()
rolled = np.roll(numpy_array, random.randint(-numpy_array.shape[0], numpy_array.shape[0]))
print(time.time() - start)