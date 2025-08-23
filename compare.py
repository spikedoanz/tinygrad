import tinygrad
from tinygrad import Tensor, Device
import numpy as np

def compare_backends(tensor_data, operation_name, operation_func):
    """Compare the same operation across different backends"""
    print(f"\n=== Testing {operation_name} ===")
    
    # Test on METAL
    Device.DEFAULT = "METAL"
    t_metal = Tensor(tensor_data)
    result_metal = operation_func(t_metal).realize().numpy()
    
    # Test on WEBGPU  
    Device.DEFAULT = "WEBGPU"
    t_webgpu = Tensor(tensor_data)
    result_webgpu = operation_func(t_webgpu).realize().numpy()
    
    # Compare
    diff = np.abs(result_metal - result_webgpu).max()
    print(f"METAL result shape: {result_metal.shape}")
    print(f"METAL:  Min={result_metal.min():.3f}, Max={result_metal.max():.3f}, Mean={result_metal.mean():.3f}")
    print(f"WEBGPU: Min={result_webgpu.min():.3f}, Max={result_webgpu.max():.3f}, Mean={result_webgpu.mean():.3f}")
    print(f"Max diff: {diff:.6f}")
    
    return diff

# Test with progressively more complex operations
test_data = np.random.randn(1, 30, 160).astype(np.float32)

# 1. Basic operations
print("Testing basic operations on small tensor...")
compare_backends(test_data, "Copy (identity)", lambda x: x)
compare_backends(test_data, "Add constant", lambda x: x + 1.0)
compare_backends(test_data, "Multiply constant", lambda x: x * 2.0)
compare_backends(test_data, "Sum", lambda x: x.sum())
compare_backends(test_data, "Mean", lambda x: x.mean())

# 2. Test mean along different axes
compare_backends(test_data, "Mean axis=-1", lambda x: x.mean(axis=-1))
compare_backends(test_data, "Mean axis=1", lambda x: x.mean(axis=1))

# 3. Test the actual layernorm components
def test_layernorm_components(x):
    # This mimics what happens in groupnorm
    x_reshaped = x.reshape(x.shape[0], 30, -1)  # Reshape like groupnorm does
    mean = x_reshaped.mean(axis=-1, keepdim=True)
    var = ((x_reshaped - mean) ** 2).mean(axis=-1, keepdim=True)
    normalized = (x_reshaped - mean) / (var + 1e-5).sqrt()
    return normalized.reshape(x.shape)

compare_backends(test_data, "LayerNorm components", test_layernorm_components)

# 4. Test with even smaller tensors
small_data = np.random.randn(1, 4, 4).astype(np.float32)
print(f"\nTesting with very small tensor {small_data.shape}...")
compare_backends(small_data, "Small tensor mean", lambda x: x.mean())

# 5. Test if it's related to tensor creation
print(f"\nTesting tensor creation...")
compare_backends([1.0, 2.0, 3.0, 4.0], "Simple list", lambda x: x.mean())
