# cuDSS for Jax 
> Sparsify your **life**. Works *automagically* 🧙🏼
---  
This is a Minimum Working Example (MWE) to expose **cuDSS**'s sparse linear system solver to **Jax**.

Essentially it solves:
$$\mathbf{A}\mathbf{x}=\mathbf{b}$$

When $\mathbf{A}\in \mathbb{R}^{n\times m}$ is a sparse matrix in CSR format and the solution $\mathbf{x}\in \mathbb{R}^{m}$ and the right hand side (rhs) $\mathbf{b}\in \mathbb{R}^{n}$ are dense vectors. Extensions for batches dimensions should be fairly easy from this MWE but are not supported.

## Usage

Since the library is installed in the newly created `build` folder, we import it like this:
```python
from build.cudss_nanobind import CudssContext
```

It exposes the following function to **Jax**:

```python
def solve_sparse_system(csr_offsets, csr_columns, csr_values, b_values):
"""Solve a sparse linear system Ax = b.

This is a pure function interface for JAX users.

Args:
    csr_offsets: JAX array of row offsets (int32)
    csr_columns: JAX array of column indices (int32)
    csr_values: JAX array of non-zero values (float64)
    b_values: JAX array for right-hand side (float64)
    
Returns:
    JAX array with the solution
"""
```


## 🪖 Nvidia Dependencies 🪖

Of course cuda and cuDSS are necessary, install them from the NVIDIA links here:
[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
[NVIDIA cuDSS](https://developer.nvidia.com/cudss)

## ⛑ Python Dependencies ⛑
It is assumed that you are inside a virtual environment, either **venv** or **poetry** based venvs work fine.
We used [Nanobind](https://github.com/wjakob/nanobind) to generate the Python bindings of the C++ code. 

To install the dependencies:
```bash
pip install nanobind jax[cuda12]
```
## 👣 Installation 👣

The **local** installation of `cudss_jax` is very straight forward:

```bash
cd cudss_impl
mkdir build && cd build
cmake ..
make -j4
pip install -e .
```

Alternatively use the provided bash script.
**Make sure that the cudSS paths are correctly specified**, if you didn't install cuDSS in a custom folder you should be ok:
```bash
cd cudss_impl
sudo chmod+ build.sh
./build.sh
pip install -e .
```

Now go and test the `example.py`.