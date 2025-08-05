cmake_minimum_required(VERSION 3.26)
project(icp_toybox)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")

# Find required packages
find_package(gflags 2.2.2 REQUIRED)
find_package(glog 0.6.0 REQUIRED)
find_package(Ceres 2.2.0 REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(Open3D 0.17.0 REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP could not be found.")
endif()

# Fetch CLI11 library
include(FetchContent)
FetchContent_Declare(
    cli11_lib
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)
FetchContent_MakeAvailable(cli11_lib)

# Include directories
include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${Open3D_INCLUDE_DIRS}
)

# Define source files
set(ICP_SOURCES
    src/icp_base.cpp
    src/icp.cpp
    src/icp_plane.cpp
    src/gicp.cpp
    src/ceres_optimizer.cpp
    src/utils.cpp
)

# Create ICP library
add_library(${PROJECT_NAME} ${ICP_SOURCES})

# Link libraries to the main library
target_link_libraries(${PROJECT_NAME}
    ${Open3D_LIBRARIES}
    Ceres::ceres
)

# Build examples
add_executable(icp_example examples/icp_example.cpp)
target_link_libraries(icp_example PRIVATE
    ${PROJECT_NAME}
    CLI11::CLI11
)

# Install examples
install(TARGETS icp_example
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
