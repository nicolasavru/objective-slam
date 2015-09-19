# include(/opt/pcl/src/pcl/cmake/pcl_utils.cmake)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
#SET(CMAKE_BUILD_TYPE RelWithDebInfo)

project(alignment)

# set(PCL_DIR "/opt/pcl/src/pcl/build")
find_package(PCL 1.7 REQUIRED)

set(CMAKE_MODULE_PATH "/opt/pcl/src/pcl/cmake/Modules/" ${CMAKE_MODULE_PATH})
find_package(Eigen REQUIRED)

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


get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()


if (UNIX OR APPLE)
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-ccbin g++-4.9")
endif()

if(NOT UNIX OR APPLE)
    add_definitions(-DPCLAPI_EXPORTS)
endif()

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true --prec-div=false --prec-sqrt=false -arch=sm_35 -DTHRUST_DEBUG -G -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true --prec-div=false --prec-sqrt=false -arch=sm_35 -DTHRUST_DEBUG")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true --prec-div=false --prec-sqrt=false -arch=sm_35")
CUDA_COMPILE(cuda_objs ${cuda})

if (WIN32)
    add_executable (alignment src/alignment.cpp ${cuda})
else()
    add_executable (alignment src/alignment.cpp ${cuda_objs})
endif()

target_link_libraries (alignment ${PCL_LIBRARIES} ${CUDA_LIBRARIES})
