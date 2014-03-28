cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
#SET(CMAKE_BUILD_TYPE RelWithDebInfo)

project(alignment)

#set(PCL_DIR "/afs/ee.cooper.edu/user/r/a/rasputin/Documents/Senior_Project/pcl/build")
find_package(PCL 1.7 REQUIRED)

#cuda info:
# http://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake
# https://devtalk.nvidia.com/default/topic/487573/cuda-programming-and-performance/linker-error-usr-bin-ld-cannot-find-lcudart/post/3497498/#3497498
find_package(CUDA)

FILE(GLOB srcs src/*.cpp)
FILE(GLOB cuda src/cuda/*.cu)

source_group("Source Files\\cuda" FILES ${cuda})
source_group("Source Files" FILES ${srcs} )

set(LIB_NAME alignment_cuda)

include_directories(${PCL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if (UNIX OR APPLE)
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
endif()

if(NOT UNIX OR APPLE)
    add_definitions(-DPCLAPI_EXPORTS)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O0 -g -G --ftz=true --prec-div=false --prec-sqrt=false -arch=sm_35")
CUDA_COMPILE(cuda_objs ${cuda})

add_executable (alignment src/alignment.cpp ${cuda_objs})
target_link_libraries (alignment ${PCL_LIBRARIES} ${CUDA_LIBRARIES})
