import numpy as np
import argparse


def check_tensors_close(file1, file2, shape_str, dtype, atol=1e-05, rtol=1e-05):
    shape = list(map(int, shape_str.split(",")))
    print(f"Comparing {file1} and {file2}")
    print(f"Shape: {shape}")
    print(f"Data type: {dtype}")

    tensor1 = np.fromfile(file1, dtype=dtype).reshape(shape)
    tensor2 = np.fromfile(file2, dtype=dtype).reshape(shape)

    close_result = np.isclose(tensor1, tensor2, atol=atol, rtol=rtol)

    if close_result.all():
        print("OK")
    else:
        abs_diff = np.abs(tensor1 - tensor2)
        rel_diff = abs_diff / np.abs(tensor1)
        max_abs_error = np.max(abs_diff)
        max_rel_error = np.max(rel_diff)
        print(f"Max absolute error: {max_abs_error}")
        print(f"Max relative error: {max_rel_error}")
        print(f"Tensor 1:\n{tensor1}")
        print(f"Tensor 2:\n{tensor2}")


def main():

    parse = argparse.ArgumentParser(description="Compare two tensors")

    parse.add_argument("file1", type=str, help="Path of the first tensor")
    parse.add_argument("file2", type=str, help="Path of the second tensor")
    parse.add_argument("shape", type=str, help="Shape of the tensors")
    parse.add_argument("dtype", type=str, help="Data type of the tensors")
    parse.add_argument("--atol", type=float, default=1e-05, help="Absolute tolerance")
    parse.add_argument("--rtol", type=float, default=1e-08, help="Relative tolerance")

    args = parse.parse_args()

    print("[begin] check tensor")
    check_tensors_close(
        args.file1,
        args.file2,
        args.shape,
        args.dtype,
        atol=args.atol,
        rtol=args.rtol,
    )
    print("[end] check tensor")


if __name__ == "__main__":
    main()
