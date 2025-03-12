# cudss_jax.py
import jax
import jax.numpy as jnp

# Import the nanobind module
from build.cudss_nanobind import CudssContext

def get_device_buffer_pointer(buf):
    """Get the raw device pointer from a JAX buffer."""
    # Access the CUDA buffer's device pointer through the JAX buffer's internals
    ptr = jax.dlpack.to_dlpack(buf).data_ptr
    return ptr

def get_jax_stream():
    """Get the current JAX CUDA stream."""
    # Note: This is implementation-dependent and may need to be updated
    # with JAX API changes. JAX doesn't officially expose its CUDA stream.
    # This is a common workaround.
    from jax._src.lib import xla_client
    from jax._src.lib import xla_extension
    
    # Get the current device and its platform
    backend = jax.lib.xla_bridge.get_backend("gpu")
    device = backend.devices()[0]
    platform = xla_client.get_platform_name()
    
    if platform == "gpu":
        # For NVIDIA GPUs, get the CUDA stream
        cuda_context = xla_extension.get_cuda_context(device.id)
        stream = xla_extension.get_cuda_stream(device.id)
        return stream
    else:
        raise RuntimeError(f"Unsupported platform: {platform}")

class CudssJaxSolver:
    """JAX-friendly wrapper for cuDSS solver."""
    
    def __init__(self):
        """Initialize the solver with the JAX CUDA stream."""
        # Get the JAX CUDA stream and create the context
        self.stream = get_jax_stream()
        self.context = CudssContext(self.stream)
    
    def setup_matrix(self, csr_offsets, csr_columns, csr_values):
        """Set up the matrix in CSR format.
        
        Args:
            csr_offsets: JAX array of row offsets (int32)
            csr_columns: JAX array of column indices (int32)
            csr_values: JAX array of non-zero values (float64)
        """
        # We need to block until all JAX operations are complete
        jax.block_until_ready(csr_offsets)
        jax.block_until_ready(csr_columns)
        jax.block_until_ready(csr_values)
        
        self.context.setup_matrix(csr_offsets, csr_columns, csr_values)
    
    def setup_vectors(self, b_values, x_values):
        """Set up the right-hand side and solution vectors.
        
        Args:
            b_values: JAX array for right-hand side (float64)
            x_values: JAX array for solution (float64)
        """
        jax.block_until_ready(b_values)
        jax.block_until_ready(x_values)
        
        self.context.setup_vectors(b_values, x_values)
    
    def analyze(self):
        """Run the analysis phase."""
        self.context.analyze()
    
    def factorize(self):
        """Run the factorization phase."""
        self.context.factorize()
    
    def solve(self):
        """Run the solve phase."""
        self.context.solve()
        self.context.synchronize()
    
    def solve_system(self, csr_offsets, csr_columns, csr_values, b_values):
        """Solve a sparse linear system end-to-end.
        
        Args:
            csr_offsets: JAX array of row offsets (int32)
            csr_columns: JAX array of column indices (int32)
            csr_values: JAX array of non-zero values (float64)
            b_values: JAX array for right-hand side (float64)
            
        Returns:
            JAX array with the solution
        """
        # Create output array for x with same shape as b
        x_values = jnp.zeros_like(b_values)
        
        # Solve the system
        self.setup_matrix(csr_offsets, csr_columns, csr_values)
        self.setup_vectors(b_values, x_values)
        self.analyze()
        self.factorize()
        self.solve()
        
        return x_values

# Function for JAX users to solve a sparse system
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
    # Try to enable float64 precision
    try:
        jax.config.update("jax_enable_x64", True)
    except:
        pass
    
    # Make sure inputs have the right type and are on GPU
    csr_offsets = jax.device_put(csr_offsets.astype(jnp.int32), jax.devices("gpu")[0])
    csr_columns = jax.device_put(csr_columns.astype(jnp.int32), jax.devices("gpu")[0])
    csr_values = jax.device_put(csr_values.astype(jnp.float64), jax.devices("gpu")[0])
    b_values = jax.device_put(b_values.astype(jnp.float64), jax.devices("gpu")[0])
    
    # Create output array
    x_values = jnp.zeros_like(b_values)
    x_values = jax.device_put(x_values, jax.devices("gpu")[0])
    
    # Import only when needed to avoid circular import
    from build.cudss_nanobind import solve_system
    
    try:
        # Make sure arrays are evaluated before passing to nanobind
        jax.block_until_ready(csr_offsets)
        jax.block_until_ready(csr_columns)
        jax.block_until_ready(csr_values)
        jax.block_until_ready(b_values)
        jax.block_until_ready(x_values)
        
        # Call the nanobind function directly with JAX arrays
        # nanobind will handle the DLPack conversion behind the scenes
        solve_system(csr_offsets, csr_columns, csr_values, b_values, x_values)
        
        # Make sure the result is fully evaluated
        x_result = jax.block_until_ready(x_values)
        return x_result
    except Exception as e:
        print(f"Error in cuDSS solve: {e}")
        # Return zeros as fallback
        return x_values

# Create a JIT-compatible version using custom_call
# Note: This is more complex and would require additional work to implement properly
def jittable_solve_sparse_system(csr_offsets, csr_columns, csr_values, b_values):
    """A placeholder for a properly JIT-compatible sparse solver.
    
    This would require more work to implement as a proper JAX primitive.
    """
    # For now, just use the direct solver which will work but break JIT
    return solve_sparse_system(csr_offsets, csr_columns, csr_values, b_values)