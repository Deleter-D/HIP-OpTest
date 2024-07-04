import numpy as np
import argparse


def generateTensorToFile(file_path, min, max, shape_str, dtype):
    shape = list(map(int, shape_str.split(",")))
    print(f"file path: {file_path}")
    print(f"range: [{min}, {max})")
    print(f"shape: {shape}")
    print(f"dtype: {dtype}")
    with open(file_path, "wb") as f:
        np.random.uniform(min, max, shape).astype(dtype).tofile(f)


def main():
    parser = argparse.ArgumentParser(description="tensor data generator")

    # 定义参数
    parser.add_argument("file_path", type=str, help="the file to save the tensor")
    parser.add_argument("shape", type=str, help="the shape of the tensor")
    parser.add_argument(
        "dtype",
        type=str,
        default="float32",
        help="the data type of the tensor",
    )
    parser.add_argument(
        "--min", type=float, default=0.0, help="the minimum value of the tensor"
    )
    parser.add_argument(
        "--max", type=float, default=1.0, help="the maximum value of the tensor"
    )

    # 解析参数
    args = parser.parse_args()

    print("[begin] generate tensor")
    generateTensorToFile(args.file_path, args.min, args.max, args.shape, args.dtype)
    print("[end] generate tensor")


if __name__ == "__main__":
    main()
