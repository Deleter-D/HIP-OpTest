import numpy as np
import torch
from torch.functional import F

input_np = np.fromfile("input.bin", dtype=np.float16).reshape(16, 100000)
input = torch.from_numpy(input_np)

output = F.log_softmax(input, dim=-1)
output_dcu = np.fromfile("output_dcu.bin", dtype=np.float16).reshape(16, 100000)

if not np.allclose(output, output_dcu, atol=1e-3):
    print(f"cpu: {output.numpy()}")
    print(f"dcu: {output_dcu}")
