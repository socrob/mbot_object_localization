cmake_minimum_required(VERSION 2.8.3)
project(mbot_object_localization)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        darknet_ros_py
        cv_bridge
        image_transport
        message_filters
        message_generation
        geometry_msgs
        mbot_perception_msgs
        tf
        image_geometry
)

# Find BOOST
find_package(Boost REQUIRED system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS})
if(NOT Boost_FOUND)
    message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)

# Find Eigen3
find_package(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})
link_directories(${Eigen3_LIBRARY_DIRS})
add_definitions(${Eigen3_DEFINITIONS})
if(NOT Eigen3_FOUND)
    message(ERROR "Eigen3 not found!")
endif(NOT Eigen3_FOUND)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)

catkin_package(CATKIN_DEPENDS
        geometry_msgs
        message_runtime
        sensor_msgs
        std_msgs
        mbot_perception_msgs
)

include_directories(
        ros/include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(object_localization_node ros/src/object_localization_node.cpp ros/src/object_localization.cpp)
target_link_libraries(object_localization_node ${catkin_LIBRARIES} ${OpenCL_LIBRARIES} ${OpenCV_LIBRARIES})