cmake_minimum_required(VERSION 3.14)
project(cudss_jax_nanobind LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v1.5.0
)
FetchContent_MakeAvailable(nanobind)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set path to cuDSS (optional, for non-standard installations)
set(CUDSS_ROOT_DIR "" CACHE PATH "Path to cuDSS installation (leave empty for system installation)")

# Try to find using CMake config first (preferred method)
find_package(cudss QUIET CONFIG PATHS 
    /usr/lib/x86_64-linux-gnu/cmake
    /usr/lib/x86_64-linux-gnu/libcudss/*/cmake
    ${CUDSS_ROOT_DIR}/lib/cmake
    ${CUDSS_ROOT_DIR}/lib64/cmake
)

if(cudss_FOUND)
    message(STATUS "Found cuDSS via CMake config")
    # Since we found cuDSS via CMake config, we need to ensure we have the include dir
    # Try to find the cudss.h header in standard locations
    find_path(CUDSS_INCLUDE_DIR
        NAMES cudss.h
        PATHS
            /usr/include
            /usr/include/libcudss
            /usr/include/libcudss/12
            /usr/local/include
            ${CUDSS_ROOT_DIR}/include
    )
    
    if(NOT CUDSS_INCLUDE_DIR)
        message(FATAL_ERROR "Found cuDSS library via CMake config but could not find cudss.h header")
    endif()
    
    set(CUDSS_LIBRARY cudss::cudss)
else()
    # Fall back to manual search for libraries
    if(CUDSS_ROOT_DIR)
        # If user specified a custom path, look only there
        find_path(CUDSS_INCLUDE_DIR
            NAMES cudss.h
            PATHS 
                ${CUDSS_ROOT_DIR}/include
                ${CUDSS_ROOT_DIR}/include/libcudss
                ${CUDSS_ROOT_DIR}/include/libcudss/12
            NO_DEFAULT_PATH
        )

        find_library(CUDSS_LIBRARY
            NAMES cudss
            PATHS ${CUDSS_ROOT_DIR}/lib64 ${CUDSS_ROOT_DIR}/lib ${CUDSS_ROOT_DIR}/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH
        )
    else()
        # Search standard system paths
        find_path(CUDSS_INCLUDE_DIR
            NAMES cudss.h
            PATHS
                /usr/include
                /usr/include/libcudss
                /usr/include/libcudss/12
                /usr/local/include
                /opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/include
        )

        find_library(CUDSS_LIBRARY
            NAMES cudss
            PATHS
                /usr/lib
                /usr/lib64
                /usr/lib/x86_64-linux-gnu
                /usr/local/lib
                /usr/local/lib64
                /opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/lib64
        )
    endif()

    if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
        message(FATAL_ERROR "cuDSS not found. Please specify CUDSS_ROOT_DIR or install cuDSS in a standard location")
    endif()
endif()

message(STATUS "Found cuDSS include: ${CUDSS_INCLUDE_DIR}")
message(STATUS "Found cuDSS library: ${CUDSS_LIBRARY}")

# Configure nanobind module
nanobind_add_module(
    cudss_nanobind
    cudss_nanobind.cpp
)

# Add include directories
target_include_directories(cudss_nanobind PRIVATE
    ${CUDSS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
if(cudss_FOUND)
    target_link_libraries(cudss_nanobind PRIVATE
        cudss
        cudart
    )
else()
    target_link_libraries(cudss_nanobind PRIVATE
        ${CUDSS_LIBRARY}
        cudart
    )
endif()

# Set installation directory
install(TARGETS cudss_nanobind DESTINATION .)