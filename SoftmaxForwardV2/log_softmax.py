import numpy as np
import torch
from torch.functional import F

input_np = np.fromfile("input.bin", dtype=np.float32).reshape(16, 100000)
input = torch.from_numpy(input_np)

output = F.log_softmax(input, dim=-1)

with open("output.bin", "wb") as f:
    output.numpy().tofile(f)
