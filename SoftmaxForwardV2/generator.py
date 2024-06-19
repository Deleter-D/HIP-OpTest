import numpy as np

with open("input.bin", "wb") as f:
    np.random.uniform(-0.02, 0.02, [16, 100000]).astype(np.float16).tofile(f)
