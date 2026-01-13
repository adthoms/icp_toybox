cmake_minimum_required(VERSION 3.5)
project(icp_toybox)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(gflags 2.2.2 REQUIRED)
find_package(glog 0.6.0 REQUIRED)
find_package(Ceres 2.2.0 REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(Open3D 0.19.0 REQUIRED)
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
    GIT_TAG v2.6.1
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
    src/ceres_optimizer.cpp
    src/icp_base.cpp
    src/icp_plane.cpp
    src/gicp.cpp
)

# Create ICP library
add_library(${PROJECT_NAME} SHARED ${ICP_SOURCES})

# Set include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link libraries to the main library
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Open3D_LIBRARIES}
    Ceres::ceres
)

# Build examples
add_executable(icp_example examples/icp_example.cpp)
target_link_libraries(icp_example PRIVATE
    ${PROJECT_NAME}
    CLI11::CLI11
)

# Install targets
install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  icp_example
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
