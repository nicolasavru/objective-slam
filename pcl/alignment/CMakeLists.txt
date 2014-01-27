cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

project(alignment)

#set(PCL_DIR "/afs/ee.cooper.edu/user/r/a/rasputin/Documents/Senior_Project/pcl/build")
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (alignment alignment.cpp)
target_link_libraries (alignment ${PCL_LIBRARIES})
