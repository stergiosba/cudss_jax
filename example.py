# example.py
import jax
import jax.numpy as jnp
import numpy as np
from cudss_jax import solve_sparse_system, CudssJaxSolver

def main():
    # Set default precision to float64 (if available)
    try:
        jax.config.update("jax_enable_x64", True)
    except:
        print("Warning: Unable to enable float64 precision in JAX.")
    
    # Test with the example 5x5 matrix from the original code
    n = 5
    nnz = 8
    
    # CSR format data
    csr_offsets = jnp.array([0, 2, 4, 6, 7, 8], dtype=jnp.int32)
    csr_columns = jnp.array([0, 2, 1, 2, 2, 4, 3, 4], dtype=jnp.int32)
    csr_values = jnp.array([4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0], dtype=jnp.float64)
    
    # Right-hand side vector
    b_values = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=jnp.float64)
    
    # Expected solution
    expected_solution = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    
    print("---------------------------------------------------------")
    print("cuDSS JAX integration example: solving a real linear 5x5 system")
    print("with a symmetric positive-definite matrix")
    print("---------------------------------------------------------")
    
    # Move data to GPU
    device = jax.devices("gpu")[0]
    csr_offsets = jax.device_put(csr_offsets, device)
    csr_columns = jax.device_put(csr_columns, device)
    csr_values = jax.device_put(csr_values, device)
    b_values = jax.device_put(b_values, device)
    
    try:
        x_values_1 = solve_sparse_system(csr_offsets, csr_columns, csr_values, b_values)
        x_values_1 = jax.device_get(x_values_1)  # Copy back to host for printing
        
        for i in range(n):
            print(f"x[{i}] = {x_values_1[i]:.4f} expected {expected_solution[i]:.4f}")
        
        error_1 = jnp.max(jnp.abs(x_values_1 - expected_solution))
        print(f"Maximum error: {error_1:.4e}")
    except Exception as e:
        print(f"Error in Method: {e}")
    
    # Overall results
    print("\nExample completed")

if __name__ == "__main__":
    main()