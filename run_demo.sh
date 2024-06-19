#!/bin/bash

# usage: sh run_demo.sh <OP Name>
# for example: 
#   sh run_demo.sh Add
#   sh run_demo.sh Sort

TARGET_EXE=${1:-SoftmaxForwardV2}

echo "----------- buiding target : ${TARGET_EXE} --------------"

build_dir="$(pwd)/${TARGET_EXE}/build"
if [ -d ${build_dir} ];then
  rm -rf ${build_dir}
fi
mkdir -p ${build_dir} && cd ${build_dir} 

# build
mkdir -p ${build_dir} && cd ${build_dir} 
cmake -DTARGET_EXE=${TARGET_EXE} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../../
make

# run
echo "-------------- start running op : ${TARGET_EXE} --------------"
export MIOPEN_LOG_LEVEL=7
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_ENABLE_LOGGING_ELAPSED_TIME=1
./${TARGET_EXE}