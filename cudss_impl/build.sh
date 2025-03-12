#!/bin/bash
# build.sh - Helper script to manually build the cuDSS nanobind extension

set -e  # Exit on error

# Check for cudss.h in standard locations
CUDSS_HEADER_PATHS=(
    "/usr/include/cudss.h"
    "/usr/include/libcudss/cudss.h"
    "/usr/include/libcudss/12/cudss.h"
)

CUDSS_INCLUDE_DIR=""
for path in "${CUDSS_HEADER_PATHS[@]}"; do
    if [ -f "$path" ]; then
        CUDSS_INCLUDE_DIR=$(dirname "$path")
        echo "Found cuDSS header at: $path"
        break
    fi
done

if [ -z "$CUDSS_INCLUDE_DIR" ]; then
    echo "Warning: Could not find cudss.h in standard locations."
fi

# Create and enter build directory
mkdir -p build
cd build

# Run CMake with explicit include directory if found
if [ -n "$CUDSS_INCLUDE_DIR" ]; then
    cmake .. -DCUDSS_INCLUDE_DIR="$CUDSS_INCLUDE_DIR"
else
    cmake ..
fi

# Build
make -j4

echo "Build complete! Extension located at: $(pwd)/cudss_nanobind.*"
echo "You can now run: pip install -e .."