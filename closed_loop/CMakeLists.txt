cmake_minimum_required(VERSION 3.10)

project(ClosedLoopExperiment)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Find the MaxLab library
# Adjust the path to the MaxLab installation directory
# Default location: ~/MaxLab/share/maxlab_lib
set(MAXLAB_DIR "$ENV{HOME}/MaxLab/share/maxlab_lib" CACHE PATH "Path to MaxLab library")

# Include directories
include_directories(${MAXLAB_DIR}/maxlab/include)

# Link directories
link_directories(${MAXLAB_DIR}/maxlab/lib)

# Build process_raw_data executable
add_executable(process_raw_data process_raw_data.cpp)
target_link_libraries(process_raw_data maxlab)

# Build process_spike_events executable
add_executable(process_spike_events process_spike_events.cpp)
target_link_libraries(process_spike_events maxlab)

# Print configuration info
message(STATUS "MaxLab directory: ${MAXLAB_DIR}")
message(STATUS "Include directory: ${MAXLAB_DIR}/maxlab/include")
message(STATUS "Library directory: ${MAXLAB_DIR}/maxlab/lib")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
