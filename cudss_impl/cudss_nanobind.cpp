// cudss_nanobind.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cudss.h"

namespace nb = nanobind;

// Class to manage cuDSS resources with RAII
class CudssContext {
private:
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    
    // For copying back results
    double* x_values_d = nullptr;
    double* x_values_host = nullptr;
    size_t x_size = 0;

public:
    CudssContext() {
        // Create a new CUDA stream
        cudaError_t cuda_err = cudaStreamCreate(&stream);
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " + 
                                     std::string(cudaGetErrorString(cuda_err)));
        }
        owns_stream = true;
        
        // Initialize cuDSS resources
        cudssStatus_t status = cudssCreate(&handle);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudaStreamDestroy(stream);
            throw std::runtime_error("cudssCreate failed with status " + std::to_string(status));
        }
        
        status = cudssSetStream(handle, stream);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssDestroy(handle);
            cudaStreamDestroy(stream);
            throw std::runtime_error("cudssSetStream failed with status " + std::to_string(status));
        }
        
        status = cudssConfigCreate(&config);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssDestroy(handle);
            cudaStreamDestroy(stream);
            throw std::runtime_error("cudssConfigCreate failed with status " + std::to_string(status));
        }
        
        status = cudssDataCreate(handle, &data);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssConfigDestroy(config);
            cudssDestroy(handle);
            cudaStreamDestroy(stream);
            throw std::runtime_error("cudssDataCreate failed with status " + std::to_string(status));
        }
    }
    
    // Constructor that takes an existing CUDA stream (e.g., from JAX)
    CudssContext(void* external_stream) {
        if (!external_stream) {
            throw std::runtime_error("Null external stream provided");
        }
        
        stream = static_cast<cudaStream_t>(external_stream);
        owns_stream = false;
        
        // Initialize cuDSS resources (similar to default constructor)
        cudssStatus_t status = cudssCreate(&handle);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssCreate failed with status " + std::to_string(status));
        }
        
        status = cudssSetStream(handle, stream);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssDestroy(handle);
            throw std::runtime_error("cudssSetStream failed with status " + std::to_string(status));
        }
        
        status = cudssConfigCreate(&config);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssDestroy(handle);
            throw std::runtime_error("cudssConfigCreate failed with status " + std::to_string(status));
        }
        
        status = cudssDataCreate(handle, &data);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssConfigDestroy(config);
            cudssDestroy(handle);
            throw std::runtime_error("cudssDataCreate failed with status " + std::to_string(status));
        }
    }
    
    ~CudssContext() {
        // Clean up resources in reverse order of creation
        if (A) cudssMatrixDestroy(A);
        if (b) cudssMatrixDestroy(b);
        if (x) cudssMatrixDestroy(x);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        if (stream && owns_stream) cudaStreamDestroy(stream);
    }
    
    // Set up the sparse matrix in CSR format
    void setup_matrix(const nb::ndarray<int32_t>& csr_offsets,
                     const nb::ndarray<int32_t>& csr_columns,
                     const nb::ndarray<double>& csr_values) {
        
        // Ensure arrays are 1D
        if (csr_offsets.ndim() != 1 || csr_columns.ndim() != 1 || csr_values.ndim() != 1) {
            throw std::runtime_error("All input arrays must be 1D");
        }
        
        // Get dimensions
        int64_t n = csr_offsets.shape(0) - 1;
        int64_t nnz = csr_values.shape(0);
        
        // Clean up existing matrix if any
        if (A) {
            cudssMatrixDestroy(A);
            A = nullptr;
        }
        
        // Allocate device memory for matrix data
        int *csr_offsets_d = nullptr;
        int *csr_columns_d = nullptr;
        double *csr_values_d = nullptr;
        
        cudaError_t cuda_err;
        
        // Allocate and copy offsets
        cuda_err = cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int));
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for offsets: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        cuda_err = cudaMemcpy(csr_offsets_d, csr_offsets.data(), 
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            cudaFree(csr_offsets_d);
            throw std::runtime_error("Failed to copy offsets to device: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Allocate and copy columns
        cuda_err = cudaMalloc(&csr_columns_d, nnz * sizeof(int));
        if (cuda_err != cudaSuccess) {
            cudaFree(csr_offsets_d);
            throw std::runtime_error("Failed to allocate device memory for columns: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        cuda_err = cudaMemcpy(csr_columns_d, csr_columns.data(), 
                            nnz * sizeof(int), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            cudaFree(csr_offsets_d);
            cudaFree(csr_columns_d);
            throw std::runtime_error("Failed to copy columns to device: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Allocate and copy values
        cuda_err = cudaMalloc(&csr_values_d, nnz * sizeof(double));
        if (cuda_err != cudaSuccess) {
            cudaFree(csr_offsets_d);
            cudaFree(csr_columns_d);
            throw std::runtime_error("Failed to allocate device memory for values: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        cuda_err = cudaMemcpy(csr_values_d, csr_values.data(), 
                            nnz * sizeof(double), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            cudaFree(csr_offsets_d);
            cudaFree(csr_columns_d);
            cudaFree(csr_values_d);
            throw std::runtime_error("Failed to copy values to device: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Create matrix
        cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
        cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
        cudssIndexBase_t base = CUDSS_BASE_ZERO;
        
        cudssStatus_t status = cudssMatrixCreateCsr(&A, n, n, nnz, 
                                                   csr_offsets_d, nullptr,
                                                   csr_columns_d, 
                                                   csr_values_d,
                                                   CUDA_R_32I, CUDA_R_64F,
                                                   mtype, mview, base);
        
        if (status != CUDSS_STATUS_SUCCESS) {
            cudaFree(csr_offsets_d);
            cudaFree(csr_columns_d);
            cudaFree(csr_values_d);
            throw std::runtime_error("cudssMatrixCreateCsr failed with status " + std::to_string(status));
        }
        
        // Note: We don't free the device memory here because cuDSS now owns it
    }
    
    // Set up the right-hand side and solution vectors
    void setup_vectors(const nb::ndarray<double>& b_values,
                       nb::ndarray<double>& x_values) {
        
        // Ensure arrays are 1D
        if (b_values.ndim() != 1 || x_values.ndim() != 1) {
            throw std::runtime_error("All input arrays must be 1D");
        }
        
        // Get dimensions
        int64_t n = b_values.shape(0);
        int64_t nrhs = 1;  // Single right-hand side for now
        
        // Clean up existing vectors if any
        if (b) {
            cudssMatrixDestroy(b);
            b = nullptr;
        }
        if (x) {
            cudssMatrixDestroy(x);
            x = nullptr;
        }
        
        // Allocate device memory
        double *b_values_d = nullptr;
        double *x_values_d = nullptr;
        
        cudaError_t cuda_err;
        
        // Allocate and copy b_values
        cuda_err = cudaMalloc(&b_values_d, n * sizeof(double));
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for b_values: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        cuda_err = cudaMemcpy(b_values_d, b_values.data(), 
                            n * sizeof(double), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            cudaFree(b_values_d);
            throw std::runtime_error("Failed to copy b_values to device: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Allocate memory for x_values
        cuda_err = cudaMalloc(&x_values_d, n * sizeof(double));
        if (cuda_err != cudaSuccess) {
            cudaFree(b_values_d);
            throw std::runtime_error("Failed to allocate device memory for x_values: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Initialize x_values to zero
        cuda_err = cudaMemset(x_values_d, 0, n * sizeof(double));
        if (cuda_err != cudaSuccess) {
            cudaFree(b_values_d);
            cudaFree(x_values_d);
            throw std::runtime_error("Failed to initialize x_values on device: " + 
                                   std::string(cudaGetErrorString(cuda_err)));
        }
        
        // Create vectors
        int ldb = n, ldx = n;
        
        cudssStatus_t status = cudssMatrixCreateDn(&b, n, nrhs, ldb, 
                                                  b_values_d, 
                                                  CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudaFree(b_values_d);
            cudaFree(x_values_d);
            throw std::runtime_error("cudssMatrixCreateDn for b failed with status " + std::to_string(status));
        }
        
        status = cudssMatrixCreateDn(&x, n, nrhs, ldx, 
                                    x_values_d, 
                                    CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS) {
            cudssMatrixDestroy(b);
            b = nullptr;
            cudaFree(x_values_d);
            throw std::runtime_error("cudssMatrixCreateDn for x failed with status " + std::to_string(status));
        }
        
        // Store the pointer to x_values_d to copy back later
        this->x_values_d = x_values_d;
        this->x_values_host = x_values.data();
        this->x_size = n;
    }
    
    // Run the analysis phase (symbolic factorization)
    void analyze() {
        if (!A || !x || !b) {
            throw std::runtime_error("Matrix or vectors not set up");
        }
        
        cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, 
                                           config, data, A, x, b);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssExecute for analysis failed with status " + std::to_string(status));
        }
    }
    
    // Run the factorization phase
    void factorize() {
        if (!A || !x || !b) {
            throw std::runtime_error("Matrix or vectors not set up");
        }
        
        cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, 
                                           config, data, A, x, b);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssExecute for factorization failed with status " + std::to_string(status));
        }
    }
    
    // Run the solve phase
    void solve() {
        if (!A || !x || !b) {
            throw std::runtime_error("Matrix or vectors not set up");
        }
        
        cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_SOLVE, 
                                           config, data, A, x, b);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssExecute for solve failed with status " + std::to_string(status));
        }
        
        // Copy result back to host memory
        if (x_values_d && x_values_host && x_size > 0) {
            cudaError_t cuda_err = cudaMemcpy(x_values_host, x_values_d, 
                                            x_size * sizeof(double), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                throw std::runtime_error("Failed to copy results back to host: " + 
                                       std::string(cudaGetErrorString(cuda_err)));
            }
        }
    }
    
    // Synchronize the stream to ensure operations are complete
    void synchronize() {
        cudaError_t cuda_err = cudaStreamSynchronize(stream);
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA stream: " + 
                                    std::string(cudaGetErrorString(cuda_err)));
        }
    }
};

// Define the Python module
NB_MODULE(cudss_nanobind, m) {
    // Register the CudssContext class
    nb::class_<CudssContext>(m, "CudssContext")
        .def(nb::init<>())
        .def(nb::init<void*>())
        .def("setup_matrix", &CudssContext::setup_matrix)
        .def("setup_vectors", &CudssContext::setup_vectors)
        .def("analyze", &CudssContext::analyze)
        .def("factorize", &CudssContext::factorize)
        .def("solve", &CudssContext::solve)
        .def("synchronize", &CudssContext::synchronize);
    
    // Convenience function to solve a system end-to-end
    m.def("solve_system", [](const nb::ndarray<int32_t>& csr_offsets,
                           const nb::ndarray<int32_t>& csr_columns,
                           const nb::ndarray<double>& csr_values,
                           const nb::ndarray<double>& b_values,
                           nb::ndarray<double>& x_values) {
        // Ensure arrays are 1D
        if (csr_offsets.ndim() != 1 || csr_columns.ndim() != 1 || 
            csr_values.ndim() != 1 || b_values.ndim() != 1 || x_values.ndim() != 1) {
            throw std::runtime_error("All input arrays must be 1D");
        }
        
        // Create context and solve
        CudssContext context;
        context.setup_matrix(csr_offsets, csr_columns, csr_values);
        context.setup_vectors(b_values, x_values);
        context.analyze();
        context.factorize();
        context.solve();
        context.synchronize();
    });
}