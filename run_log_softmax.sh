#!/bin/bash

cur_dir=$(pwd)
dtype=float32
shape="16,100000"

python python/generator.py SoftmaxForwardV2/input.bin ${shape} ${dtype} --min -100 --max 100

bash run_demo.sh SoftmaxForwardV2

echo $cur_dir
cd ${cur_dir}/SoftmaxForwardV2
python log_softmax.py

cd ${cur_dir}
python python/check.py SoftmaxForwardV2/output.bin SoftmaxForwardV2/output_dcu.bin \
    ${shape} ${dtype} --atol 0.0001 --rtol 0.0001