import sys
import os
import platform
import subprocess
from pathlib import Path
from setuptools import setup

def find_cudss_include():
    """Find the cuDSS header file location."""
    possible_locations = [
        '/usr/include/cudss.h',
        '/usr/include/libcudss/cudss.h',
        '/usr/include/libcudss/12/cudss.h'
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            print(f"Found cuDSS header at: {loc}")
            return os.path.dirname(loc)
    
    return None

# Detect if we're using conda
in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

# Find out the nanobind installation path
try:
    import nanobind
    nanobind_path = Path(nanobind.__file__).parent
except ImportError:
    raise ImportError("nanobind is required. Please install it with 'pip install nanobind'")

# Find CUDA paths - try to be smart about it
cuda_paths = [
    '/usr/local/cuda',  # Default Linux/macOS CUDA path
    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA',  # Default Windows CUDA path
    os.environ.get('CUDA_HOME', ''),  # Environment variable
    os.environ.get('CUDA_PATH', '')   # Environment variable
]

# Find cudss paths - try to be smart about it
cudss_paths = [
    '/usr',                                               # Standard system path
    '/usr/local',                                         # Standard local path
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/11.0',  # Default Linux HPC SDK path
    '/opt/nvidia/hpc_sdk/Darwin_x86_64/23.11/math_libs/11.0',  # Default macOS HPC SDK path
    'C:/Program Files/NVIDIA HPC SDK/23.11/math_libs/11.0',    # Default Windows HPC SDK path
    os.environ.get('CUDSS_ROOT', ''),  # Environment variable
]

# Filter out empty paths
cuda_paths = [p for p in cuda_paths if p]
cudss_paths = [p for p in cudss_paths if p]

def find_valid_path(paths, file_to_check):
    """Find first valid path containing the specified file."""
    for path in paths:
        check_path = os.path.join(path, file_to_check)
        if os.path.exists(check_path):
            return path
        # Check alternate locations for system installs
        if path == '/usr':
            alt_check = os.path.join(path, 'include/libcudss', file_to_check.replace('include/', ''))
            if os.path.exists(alt_check):
                return path
    return None

# Find CUDA installation
cuda_path = find_valid_path(cuda_paths, 'include/cuda.h')
if not cuda_path:
    raise ValueError("CUDA installation not found. Please set CUDA_HOME environment variable.")

# Find cuDSS installation - try both direct and versioned paths
include_checks = ['include/cudss.h', 'include/libcudss/12/cudss.h']
cudss_path = None

for check in include_checks:
    cudss_path = find_valid_path(cudss_paths, check)
    if cudss_path:
        break

if not cudss_path:
    # Try to find it directly
    if os.path.exists('/usr/include/cudss.h') or os.path.exists('/usr/include/libcudss/12/cudss.h'):
        cudss_path = '/usr'
    else:
        raise ValueError("cuDSS installation not found. Please set CUDSS_ROOT environment variable.")

# Define the build command
def build_extension():
    """Build the C++ extension."""
    cwd = Path().absolute()
    build_dir = cwd / "build"
    
    # Create build directory if it doesn't exist
    if not build_dir.exists():
        build_dir.mkdir()
    
    # Change to build directory
    os.chdir(build_dir)
    
    # Run CMake configuration
    include_dir = find_cudss_include()
    
    cmd = [
        'cmake', '..',
        f'-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}'
    ]
    
    # Only add CUDSS_ROOT_DIR if it's not the system path
    if cudss_path != '/usr':
        cmd.append(f'-DCUDSS_ROOT_DIR={cudss_path}')
    
    # If we found the cudss.h file, explicitly set the include dir
    if include_dir:
        cmd.append(f'-DCUDSS_INCLUDE_DIR={include_dir}')
    subprocess.check_call(cmd)
    
    # Run make
    subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'])
    
    # Return to original directory
    os.chdir(cwd)

# Build the extension during setup
build_extension()

setup(
    name="cudss_jax",
    version="0.1.0",
    author="Stergios Bachoumas",
    author_email="stevbach@udel.edu",
    description="JAX integration for NVIDIA cuDSS using nanobind",
    py_modules=["cudss_jax"],
    package_data={
        "": ["build/cudss_nanobind.*"],  # Include the compiled extension
    },
    install_requires=[
        "jax>=0.4.0",
        "nanobind>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)