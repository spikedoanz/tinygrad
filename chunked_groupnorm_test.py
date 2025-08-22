import os, numpy as np
from tinygrad import Tensor, nn, Device
np.random.seed(42)

BACKEND1 = "METAL"
BACKEND2 = "WEBGPU"

class ChunkedGroupNorm:
    def __init__(self, num_groups: int, num_channels: int, eps=1e-5, affine=True):
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = Tensor.ones(num_channels) if affine else None
        self.bias = Tensor.zeros(num_channels) if affine else None
    
    def __call__(self, x: Tensor) -> Tensor:
        batch_size, channels, *spatial = x.shape
        channels_per_group = channels // self.num_groups
        
        # Process each group independently to reduce element count
        normalized_groups:list[Tensor] = []
        for g in range(self.num_groups):
            start_c = g * channels_per_group
            end_c = start_c + channels_per_group
            
            # Extract group: [batch, channels_per_group, *spatial]
            group = x[:, start_c:end_c]
            
            # Normalize this group (much smaller tensor)
            # Reshape to [batch, 1, channels_per_group * prod(spatial)]
            group_flat = group.reshape(batch_size, 1, -1)
            group_norm = group_flat.layernorm(eps=self.eps)
            group_norm = group_norm.reshape(batch_size, channels_per_group, *spatial)
            
            # Apply affine if needed
            if self.weight is not None and self.bias is not None:
                w = self.weight[start_c:end_c].reshape(1, channels_per_group, *[1]*len(spatial))
                b = self.bias[start_c:end_c].reshape(1, channels_per_group, *[1]*len(spatial))
                group_norm = group_norm * w + b
            
            normalized_groups.append(group_norm)
        
        # Concatenate back together
        return Tensor.cat(*normalized_groups, dim=1)

def comprehensive_test(b, c1, c2, N, test_affine=True):
    """
    Test all combinations:
    - METAL: GroupNorm vs ChunkedGroupNorm
    - WEBGPU: GroupNorm vs ChunkedGroupNorm  
    - METAL vs WEBGPU: GroupNorm
    - METAL vs WEBGPU: ChunkedGroupNorm
    """
    print(f"\n{'='*80}")
    print(f"Testing: batch={b}, in_channels={c1}, out_channels={c2}, spatial_dim={N}x{N}x{N}")
    print(f"Total elements per group: {b * c2 * N * N * N:,}")
    print(f"{'='*80}")
    
    # Generate consistent test data
    np.random.seed(42)
    data = np.random.randn(b, c2, N, N, N).astype(np.float32)
    
    results = {}
    
    for backend in [BACKEND1, BACKEND2]:
        # Set backend
        os.environ.pop(BACKEND2, None) if backend == BACKEND1 else os.environ.update({BACKEND2: '1'})
        Device.DEFAULT = backend
        
        # Test both implementations
        for impl_name, impl_class in [("Original", nn.GroupNorm), ("Chunked", ChunkedGroupNorm)]:
            try:
                x = Tensor(data)
                
                # Test with affine=False for numerical comparison
                gn = impl_class(num_groups=c2, num_channels=c2, affine=False)
                out = gn(x).realize().numpy()
                
                results[(backend, impl_name, "no_affine")] = out
                
                # Also test with affine=True if requested
                if test_affine:
                    x = Tensor(data)
                    gn_affine = impl_class(num_groups=c2, num_channels=c2, affine=True)
                    out_affine = gn_affine(x).realize().numpy()
                    results[(backend, impl_name, "affine")] = out_affine
                    
                print(f"{backend:7s} {impl_name:8s}: Min={out.min():7.3f}, Max={out.max():7.3f}, "
                      f"Mean={out.mean():7.3f}, Std={out.std():7.3f}")
                
            except Exception as e:
                print(f"{backend:7s} {impl_name:8s}: FAILED - {str(e)}")
                results[(backend, impl_name, "no_affine")] = None
    
    # Compare results
    print(f"\n{'='*40} COMPARISONS {'='*40}")
    
    # 1. Compare implementations within each backend
    for backend in [BACKEND1, BACKEND2]:
        orig = results.get((backend, "Original", "no_affine"))
        chunked = results.get((backend, "Chunked", "no_affine"))
        
        if orig is not None and chunked is not None:
            max_diff = np.abs(orig - chunked).max()
            mean_diff = np.abs(orig - chunked).mean()
            print(f"{backend:7s} Original vs Chunked: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f}")
            if max_diff > 0.001:
                print(f"  ⚠️  WARNING: Large difference detected!")
        else:
            print(f"{backend:7s} Original vs Chunked: Could not compare (one failed)")
    
    # 2. Compare backends for each implementation
    for impl_name in ["Original", "Chunked"]:
        metal_result = results.get((BACKEND1, impl_name, "no_affine"))
        webgpu_result = results.get((BACKEND2, impl_name, "no_affine"))
        
        if metal_result is not None and webgpu_result is not None:
            max_diff = np.abs(metal_result - webgpu_result).max()
            mean_diff = np.abs(metal_result - webgpu_result).mean()
            print(f"{impl_name:8s} METAL vs WEBGPU: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f}")
            if max_diff > 0.001:
                print(f"  ⚠️  WARNING: Large difference detected!")
        else:
            print(f"{impl_name:8s} METAL vs WEBGPU: Could not compare (one failed)")
    
    # 3. If affine was tested, compare those too
    if test_affine:
        print(f"\n{'='*40} AFFINE TESTS {'='*40}")
        for backend in [BACKEND1, BACKEND2]:
            for impl_name in ["Original", "Chunked"]:
                no_affine = results.get((backend, impl_name, "no_affine"))
                affine = results.get((backend, impl_name, "affine"))
                
                if no_affine is not None and affine is not None:
                    # They should be different (affine adds learnable params)
                    max_diff = np.abs(no_affine - affine).max()
                    print(f"{backend:7s} {impl_name:8s} affine effect: Max diff={max_diff:.6f}")

def test_sweep():
    """Run a sweep of different configurations"""
    print("\n" + "="*80)
    print("CONFIGURATION SWEEP TEST")
    print("="*80)
    
    test_configs = [
        # (batch, in_channels, out_channels, spatial_dim)
        (1, 30, 30, 128),      # Base case that works
        (1, 30, 30, 160),      # Larger spatial
        (1, 30, 30, 164),      # Even larger spatial
        (1, 5, 5, 164),        # Smaller channels, large spatial
        (1, 10, 10, 164),      # Medium channels, large spatial
        (2, 30, 30, 128),      # Larger batch
        (1, 60, 60, 128),      # More channels
    ]
    
    for b, c1, c2, N in test_configs:
        try:
            comprehensive_test(b, c1, c2, N, test_affine=False)
        except Exception as e:
            print(f"\nTest failed for config ({b}, {c1}, {c2}, {N}): {e}")

# Run the tests
if __name__ == "__main__":
    # Test the specific case from your example
    comprehensive_test(1, 30, 30, 128)
    
    # Run configuration sweep
    test_sweep()
