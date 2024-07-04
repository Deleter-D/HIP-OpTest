cmake_minimum_required(VERSION 3.21)

project(miopen_demo)

message(STATUS "Building target TARGET_EXE=${TARGET_EXE}")

# envs
if(DEFINED ENV{DTKROOT})
    set(DTK_DIR $ENV{DTKROOT})
else()
    set(DTK_DIR /opt/dtk-23.10.1)
endif()
if(DEFINED ENV{HYHAL_PATH})
    set(HYHAL_DIR $ENV{HYHAL_PATH})
else()
    set(HYHAL_DIR /opt/hyhal)
endif()
set(ROCM_DIR ${DTK_DIR})
set(HIP_DIR ${ROCM_DIR}/hip)

set(CMAKE_HIP_COMPILER ${HIP_DIR}/bin/hipcc)
set(CMAKE_C_COMPILER ${CMAKE_HIP_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_HIP_COMPILER})

# compile options
add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# depedencies
find_package(HIP REQUIRED)
include_directories(${ROCM_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

macro(find_package_and_include PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" REQUIRED)
  include_directories("${ROCM_PATH}/${PACKAGE_NAME}/include")
  message(STATUS "${PACKAGE_NAME} version: ${${PACKAGE_NAME}_VERSION}")
endmacro()

find_package_and_include(miopen)

# targets
add_executable(${TARGET_EXE} ${TARGET_EXE}/${TARGET_EXE}.cc common/logging.cc)
target_link_libraries(${TARGET_EXE} ${HIP_LIBRARIES} ${MIOpen_LIBRARIES})