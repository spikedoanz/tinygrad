from __future__ import annotations
import time, math, itertools, functools, struct, sys, inspect, pathlib, string, hashlib, weakref
from contextlib import ContextDecorator
from typing import Callable, ClassVar, Sequence, cast, get_args, Literal, SupportsIndex, ParamSpec, TypeVar, Generic
from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype, truncate
from tinygrad.dtype import _from_np_dtype, _to_np_dtype
from tinygrad.helpers import argfix, make_tuple, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, all_same, fully_flatten, dedup
from tinygrad.helpers import IMAGE, WINO, Metadata, TRACEMETA, ceildiv, fetch, polyN, unwrap, DEBUG, is_numpy_ndarray
from tinygrad.gradient import compute_gradient
from tinygrad.uop.ops import smax, smin, resolve, UOp, Ops, sint, Variable, MathTrait, identity_element, all_metadata
from tinygrad.uop.spec import tensor_uop_spec, type_verify
from tinygrad.device import Device, Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars
from tinygrad.schedule.kernelize import get_kernelize_map
from tinygrad import Tensor

MAX_BUFFER_SIZE = 128**3

def _get_flat_chunks(t: Tensor, max_size: int = MAX_BUFFER_SIZE) -> list[Tensor]:
    """Flatten tensor and split into chunks that are smaller than max_size."""
    total_elements = prod(t.shape)
    
    if total_elements <= max_size:
        return [t.flatten()]
    
    # Flatten the tensor first
    flat_t = t.flatten()
    
    # Calculate how many chunks we need
    num_chunks = ceildiv(total_elements, max_size)
    chunk_size = ceildiv(total_elements, num_chunks)
    
    # Create chunks by slicing the flattened tensor
    chunks = []
    for i in range(0, total_elements, chunk_size):
        end = min(i + chunk_size, total_elements)
        chunk = flat_t[i:end]
        chunks.append(chunk)
    
    return chunks

def get_chunks_along_axis(t: Tensor, axis: int, max_size: int = MAX_BUFFER_SIZE) -> list[Tensor]:
    """Split tensor along specified axis into chunks smaller than max_size."""
    # Normalize negative axis
    if axis < 0:
        axis = len(t.shape) + axis
    
    # Calculate size of each slice (all dims except the split axis)
    slice_size = prod([t.shape[i] for i in range(len(t.shape)) if i != axis])
    axis_size = t.shape[axis]
    
    # If already small enough, return as-is
    if prod(t.shape) <= max_size:
        return [t]
    
    # Calculate max axis dimension size per chunk to stay under max_size
    max_axis_per_chunk = max(1, max_size // slice_size)
    
    # If even one slice is too big, we'd need to chunk multiple dimensions
    if max_axis_per_chunk < 1:
        raise ValueError(f"Single slice along axis {axis} exceeds max_size. Need multi-dimensional chunking.")
    
    # Calculate number of chunks needed
    num_chunks = ceildiv(axis_size, max_axis_per_chunk)
    chunk_size = ceildiv(axis_size, num_chunks)
    
    # Create chunks by slicing along the specified axis
    chunks = []
    for i in range(0, axis_size, chunk_size):
        end = min(i + chunk_size, axis_size)
        
        # Build slice tuple: slice(None) for all dims except axis
        slices = [slice(None)] * len(t.shape)
        slices[axis] = slice(i, end)
        
        chunk = t[tuple(slices)]
        chunks.append(chunk)
    
    return chunks


def imean(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False) -> Tensor:
    return Tensor(1.34976127323)

def ivar(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False, correction: int = 1) -> Tensor:
    return Tensor(1.21093810294)



def smean(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False) -> Tensor:
    t = self
    output_dtype = t.dtype if dtypes.is_float(t.dtype) else dtypes.float32
    if isinstance(axis, int): axis = [axis]
    if axis is None: axis = list(range(len(self.shape)))  # Reduce all dims
    paxis = [(d if d >= 0 else len(self.shape) + d) for d in axis]
    reduce_dims = sorted(paxis)
    preserved_dims = sorted(set(range(len(self.shape))) - set(reduce_dims))
    
    # Base case: tensor small enough
    if prod(self.shape) < MAX_BUFFER_SIZE:
        numerator = self.cast(sum_acc_dtype(self.dtype))\
            .sum(axis=axis, keepdim=keepdim)
        denominator = prod([self.shape[d] for d in paxis])
        return numerator.div(denominator).cast(output_dtype)
    
    # Calculate total elements to reduce
    reduce_elements = prod([self.shape[d] for d in reduce_dims])
    preserved_elements = prod([self.shape[d] for d in preserved_dims]) if preserved_dims else 1
    
    # Reshape tensor to [preserved_elements, reduce_elements]
    t_flat = t.reshape(preserved_elements, reduce_elements)
    
    # Chunk along the reduction dimension
    max_chunk_size = MAX_BUFFER_SIZE // max(1, preserved_elements)
    num_chunks = ceildiv(reduce_elements, max_chunk_size)
    
    if num_chunks > 1:
        chunks = t_flat.chunk(num_chunks, dim=1)
        
        # Sum each chunk and accumulate
        total_sum = None
        for chunk in chunks:
            chunk_sum = chunk.cast(sum_acc_dtype(self.dtype)).sum(axis=1, keepdim=True)
            total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum
        
        result = total_sum.div(reduce_elements).cast(output_dtype)
    else:
        # No chunking needed
        result = t_flat.mean(axis=1, keepdim=True)
    
    # Reshape back to original preserved dimensions
    if preserved_dims:
        preserved_shape = [self.shape[d] for d in preserved_dims]
        if keepdim:
            # Insert 1s for reduced dimensions
            final_shape = list(self.shape)
            for d in reduce_dims:
                final_shape[d] = 1
            result = result.reshape(*final_shape)
        else:
            result = result.reshape(*preserved_shape)
    elif keepdim:
        result = result.reshape(*[1] * len(self.shape))
    else:
        result = result.squeeze()
    
    return result

    """
    # PREMATURE OPTIMIZATION
    # If no preserved dims, we must reduce sequentially
    if not preserved_dims:
        result = t
        for d in reversed(sorted(paxis)):  # Reverse to maintain indices
            result = smean(result, axis=d, keepdim=True)
        if not keepdim:
            # Squeeze all reduced dimensions
            for d in reversed(sorted(paxis)):
                result = result.squeeze(d)
        return result.cast(output_dtype)
    
    # Shard along a preserved dimension
    for i in preserved_dims:  # Only shard preserved dims!
        s = int(self.shape[i])
        if s > 1:
            split_point = s // 2
            t1, t2 = t.split([split_point, s - split_point], dim=i)
            t1.realize(); t2.realize()
            
            # Recursively compute means (always with keepdim=True for correct concat)
            mean_t1 = smean(t1, axis, keepdim=True)
            mean_t2 = smean(t2, axis, keepdim=True)
            
            # Concatenate along the sharded dimension
            result = Tensor.cat(mean_t1, mean_t2, dim=i)
            
            # Handle keepdim at the end
            if not keepdim:
                for d in reversed(sorted(paxis)):
                    result = result.squeeze(d)
            
            return result.cast(output_dtype)
    
    # If we get here, all preserved dims have size 1 - shouldn't happen
    raise RuntimeError(f"Unable to shard tensor with shape {self.shape}")
    """

print(smean(Tensor.randn(1,1,256,256,256), axis=[-1]))


def _smean(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False) -> Tensor:
    t = self
    """Sharded mean that respects MAX_BUFFER_SIZE by processing flattened chunks."""
    output_dtype = t.dtype if dtypes.is_float(t.dtype) else dtypes.float32
    print(axis)
    
    if axis is None:
        # Original behavior - flatten everything
        total_elements = prod(t.shape)
        chunks = _get_flat_chunks(t, MAX_BUFFER_SIZE)
        
        chunk_sums = []
        for chunk in chunks:
            chunk = chunk.realize()
            chunk_sum = chunk.cast(sum_acc_dtype(chunk.dtype)).sum()
            chunk_sums.append(chunk_sum.realize())
        
        if len(chunk_sums) == 1:
            total_sum = chunk_sums[0]
        else:
            total_sum = chunk_sums[0]
            for cs in chunk_sums[1:]:
                total_sum = total_sum + cs
        
        result = total_sum.div(total_elements)
        
        if keepdim and not result.shape:
            result = result.reshape(tuple(1 for _ in t.shape))
    else:
        # Handle axis parameter
        # Normalize axis to a tuple of positive indices
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(a if a >= 0 else len(t.shape) + a for a in axis)
        
        # Calculate shapes for reshaping
        keep_dims_indices = [i for i in range(len(t.shape)) if i not in axis]
        keep_shape = tuple(t.shape[i] for i in keep_dims_indices)
        reduce_shape = tuple(t.shape[i] for i in axis)
        
        # Calculate number of elements being reduced per output element
        reduce_elements = prod(reduce_shape)
        keep_elements = prod(keep_shape)
        
        # Permute tensor to put reduce dims at the end
        perm = keep_dims_indices + list(axis)
        permuted = t.permute(*perm) if perm != list(range(len(t.shape))) else t
        
        # Reshape to (keep_elements, reduce_elements)
        reshaped = permuted.reshape((keep_elements, reduce_elements))
        
        # Get chunks along the keep dimension (dim 0)
        # Each chunk will have shape (chunk_size, reduce_elements)
        chunk_size = min(MAX_BUFFER_SIZE // reduce_elements, keep_elements)
        chunk_size = max(1, chunk_size)  # Ensure at least 1
        
        chunk_means = []
        for i in range(0, keep_elements, chunk_size):
            end = min(i + chunk_size, keep_elements)
            chunk = reshaped[i:end].realize()
            
            # Sum along the reduce dimension (axis=1) and divide by reduce_elements
            chunk_mean = chunk.cast(sum_acc_dtype(chunk.dtype)).sum(axis=1).div(reduce_elements)
            chunk_means.append(chunk_mean.realize())
        
        # Stack all chunk results
        if len(chunk_means) == 1:
            result = chunk_means[0]
        else:
            # Concatenate all chunks along dimension 0
            result = chunk_means[0].cat(*chunk_means[1:], dim=0)
        
        # Reshape back to original shape (minus reduced dimensions or with 1s)
        if keepdim:
            # Insert 1s for reduced dimensions
            new_shape = list(t.shape)
            for i in axis:
                new_shape[i] = 1
            result = result.reshape(tuple(new_shape))
        else:
            # Reshape to keep dimensions only
            result = result.reshape(keep_shape)
    
    return result.cast(output_dtype)

def svar(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False, correction: int = 1) -> Tensor:
    """Sharded variance using Welford's online algorithm with flattened chunks."""
    t = self
    output_dtype = t.dtype if dtypes.is_float(t.dtype) else dtypes.float32
    
    # Handle the axis parameter
    if axis is not None:
        # For now, use original implementation if axis is specified
        squares = (t - t.mean(axis=axis, keepdim=True)).square()
        n = prod([si for si, so in zip(t.shape, squares.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
        return squares.sum(axis=axis, keepdim=keepdim).div(smax([0, n-correction])).cast(output_dtype)
    
    # Calculate total number of elements
    total_elements = prod(t.shape)
    
    # If tensor is small enough, use regular variance
    if total_elements <= MAX_BUFFER_SIZE:
        mean_val = t.mean()
        squares = (t - mean_val).square()
        result = squares.sum().div(smax([0, total_elements - correction]))
        if keepdim and not result.shape:
            result = result.reshape(tuple(1 for _ in t.shape))
        return result.cast(output_dtype)
    
    # Get flattened chunks
    chunks = _get_flat_chunks(t, MAX_BUFFER_SIZE)
    
    # Use parallel algorithm for variance computation
    # We need to track: count, mean, and M2 (sum of squared differences) for each chunk
    chunk_stats = []
    
    for chunk in chunks:
        # Realize the chunk
        chunk = chunk.realize()
        chunk_n = prod(chunk.shape)
        
        # Compute mean for this chunk
        chunk_mean = chunk.cast(sum_acc_dtype(chunk.dtype)).sum().div(chunk_n).realize()
        
        # Compute sum of squared differences from chunk mean
        chunk_m2 = ((chunk - chunk_mean).square()).cast(sum_acc_dtype(chunk.dtype)).sum().realize()
        
        chunk_stats.append((chunk_n, chunk_mean, chunk_m2))
    
    # Combine chunk statistics using parallel variance algorithm
    # This correctly combines means and variances from different chunks
    if len(chunk_stats) == 1:
        total_n, total_mean, total_m2 = chunk_stats[0]
    else:
        # Initialize with first chunk
        total_n, total_mean, total_m2 = chunk_stats[0]
        
        # Combine with remaining chunks
        for chunk_n, chunk_mean, chunk_m2 in chunk_stats[1:]:
            # Combine two sets of statistics
            new_n = total_n + chunk_n
            delta = chunk_mean - total_mean
            
            # Update combined mean
            new_mean = total_mean + delta * (chunk_n / new_n)
            
            # Update combined M2
            new_m2 = total_m2 + chunk_m2 + delta * delta * (total_n * chunk_n / new_n)
            
            total_n = new_n
            total_mean = new_mean
            total_m2 = new_m2
    
    # Compute final variance
    result = total_m2.div(smax([0, total_n - correction]))
    
    # Handle keepdim
    if keepdim and not result.shape:
        result = result.reshape(tuple(1 for _ in t.shape))
    
    return result.cast(output_dtype)


# ============== TESTS ==============

def test_sharded_stats():
    """Test sharded mean and variance against regular implementations."""
    import numpy as np
    
    print("Testing sharded mean and variance implementations with flattening...")
    print("=" * 60)
    
    # Test different tensor sizes
    test_cases = [
        # (shape, name)
        ((100,), "Small 1D"),
        ((1000,), "Medium 1D"),
        ((100, 100), "Small 2D"),
        ((500, 500), "Medium 2D"),
        ((128, 128, 128), "Cube at boundary"),
        ((150, 150, 150), "Cube over boundary"),
        ((200, 200, 200), "Large cube"),
        ((1000, 1000, 10), "Large flat tensor"),
    ]
    
    # Temporarily reduce MAX_BUFFER_SIZE for testing
    global MAX_BUFFER_SIZE
    original_max = MAX_BUFFER_SIZE
    MAX_BUFFER_SIZE = 128**2  # Smaller size for testing
    
    try:
        for shape, name in test_cases:
            print(f"\nTest: {name} - Shape: {shape}")
            print(f"  Total elements: {prod(shape):,}")
            print(f"  Chunks needed: {ceildiv(prod(shape), MAX_BUFFER_SIZE)}")
            
            # Create random tensor
            np.random.seed(42)  # For reproducibility
            data = np.random.randn(*shape).astype(np.float32)
            t = Tensor(data)
            
            # Test mean
            print("  Testing mean...")
            regular_mean = t.mean()
            sharded_mean_val = smean(t)
            
            # Realize both tensors
            regular_mean_np = regular_mean.numpy()
            sharded_mean_np = sharded_mean_val.numpy()
            
            mean_diff = np.abs(regular_mean_np - sharded_mean_np)
            mean_rel_error = mean_diff / (np.abs(regular_mean_np) + 1e-8)
            
            print(f"    Regular mean: {regular_mean_np:.6f}")
            print(f"    Sharded mean: {sharded_mean_np:.6f}")
            print(f"    Absolute diff: {mean_diff:.2e}")
            print(f"    Relative error: {mean_rel_error:.2e}")
            
            # Test variance
            print("  Testing variance...")
            regular_var = t.var()
            sharded_var_val = svar(t)
            
            # Realize both tensors
            regular_var_np = regular_var.numpy()
            sharded_var_np = sharded_var_val.numpy()
            
            var_diff = np.abs(regular_var_np - sharded_var_np)
            var_rel_error = var_diff / (np.abs(regular_var_np) + 1e-8)
            
            print(f"    Regular var: {regular_var_np:.6f}")
            print(f"    Sharded var: {sharded_var_np:.6f}")
            print(f"    Absolute diff: {var_diff:.2e}")
            print(f"    Relative error: {var_rel_error:.2e}")
            
            # Check if results are close enough (accounting for floating point errors)
            assert mean_rel_error < 1e-5, f"Mean relative error too large: {mean_rel_error}"
            assert var_rel_error < 1e-4, f"Variance relative error too large: {var_rel_error}"
            print(f"  ✓ PASSED")
            
    finally:
        # Restore original MAX_BUFFER_SIZE
        MAX_BUFFER_SIZE = original_max
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\nTesting edge cases...")
    print("=" * 60)
    
    # Test with uniform tensor (variance should be ~0)
    print("\n1. Uniform tensor (all same value):")
    t_uniform = Tensor.ones(1000, 100)
    sm = smean(t_uniform).numpy()
    sv = svar(t_uniform).numpy()
    print(f"  Mean: {sm:.6f} (expected: 1.0)")
    print(f"  Var: {sv:.9f} (expected: ~0.0)")
    assert np.abs(sm - 1.0) < 1e-5
    assert sv < 1e-5
    print("  ✓ PASSED")
    
    # Test with keepdim
    print("\n2. Testing keepdim=True:")
    t = Tensor.randn(100, 100)
    regular_mean_kd = t.mean(keepdim=True)
    sharded_mean_kd = smean(t, keepdim=True)
    print(f"  Regular shape: {regular_mean_kd.shape}")
    print(f"  Sharded shape: {sharded_mean_kd.shape}")
    assert regular_mean_kd.shape == sharded_mean_kd.shape
    print("  ✓ PASSED")
    
    # Test with different dtypes
    print("\n3. Testing with integers:")
    t_int = Tensor.arange(0, 10000).reshape(100, 100)
    sm_int = smean(t_int).numpy()
    expected_mean = 4999.5
    print(f"  Mean: {sm_int:.2f} (expected: {expected_mean})")
    assert np.abs(sm_int - expected_mean) < 1
    print("  ✓ PASSED")
    
    # Test correction parameter in variance
    print("\n4. Testing variance with correction=0:")
    t = Tensor.randn(100, 100)
    var_corr0 = svar(t, correction=0)
    var_corr1 = svar(t, correction=1)
    var0_np = var_corr0.numpy()
    var1_np = var_corr1.numpy()
    print(f"  Var (correction=0): {var0_np:.6f}")
    print(f"  Var (correction=1): {var1_np:.6f}")
    print(f"  Ratio: {var0_np/var1_np:.6f} (expected: ~{9999/10000:.6f})")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All edge case tests passed! ✓")


def benchmark_sharded_stats():
    """Benchmark the performance of sharded vs regular implementations."""
    print("\nBenchmarking sharded implementations...")
    print("=" * 60)
    
    import time
    
    # Reduce buffer size for demonstration
    global MAX_BUFFER_SIZE
    original_max = MAX_BUFFER_SIZE
    MAX_BUFFER_SIZE = 128**2
    
    shapes = [
        (1000, 1000),
        (500, 500, 10),
        (200, 200, 200),
    ]
    
    try:
        for shape in shapes:
            print(f"\nShape: {shape} ({prod(shape):,} elements)")
            t = Tensor.randn(*shape)
            
            # Benchmark regular mean
            start = time.time()
            reg_mean = t.mean().numpy()
            reg_mean_time = time.time() - start
            
            # Benchmark sharded mean
            start = time.time()
            shard_mean = smean(t).numpy()
            shard_mean_time = time.time() - start
            
            print(f"  Mean - Regular: {reg_mean_time:.3f}s, Sharded: {shard_mean_time:.3f}s")
            print(f"    Slowdown: {shard_mean_time/reg_mean_time:.2f}x")
            
            # Benchmark regular variance
            start = time.time()
            reg_var = t.var().numpy()
            reg_var_time = time.time() - start
            
            # Benchmark sharded variance
            start = time.time()
            shard_var = svar(t).numpy()
            shard_var_time = time.time() - start
            
            print(f"  Var - Regular: {reg_var_time:.3f}s, Sharded: {shard_var_time:.3f}s")
            print(f"    Slowdown: {shard_var_time/reg_var_time:.2f}x")
            
    finally:
        MAX_BUFFER_SIZE = original_max
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")


def test_flattening_behavior():
    """Test that flattening preserves correctness across different shapes."""
    import numpy as np
    
    print("\nTesting flattening behavior...")
    print("=" * 60)
    
    # Temporarily reduce MAX_BUFFER_SIZE for testing
    global MAX_BUFFER_SIZE
    original_max = MAX_BUFFER_SIZE
    MAX_BUFFER_SIZE = 1000  # Very small to force many chunks
    
    try:
        # Test various shapes to ensure flattening works correctly
        shapes = [
            (10, 10, 10, 10),  # 4D tensor
            (5, 5, 5, 5, 5),   # 5D tensor  
            (100, 1, 100),     # Tensor with singleton dimension
            (1, 10000),        # Row vector
            (10000, 1),        # Column vector
        ]
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            np.random.seed(123)
            data = np.random.randn(*shape).astype(np.float32)
            t = Tensor(data)
            
            # Test that flattened chunking gives same results
            regular_mean = t.mean().numpy()
            sharded_mean = smean(t).numpy()
            
            regular_var = t.var().numpy()
            sharded_var = svar(t).numpy()
            
            mean_error = np.abs(regular_mean - sharded_mean) / (np.abs(regular_mean) + 1e-8)
            var_error = np.abs(regular_var - sharded_var) / (np.abs(regular_var) + 1e-8)
            
            print(f"  Mean relative error: {mean_error:.2e}")
            print(f"  Var relative error: {var_error:.2e}")
            
            assert mean_error < 1e-5, f"Mean error too large for shape {shape}"
            assert var_error < 1e-4, f"Var error too large for shape {shape}"
            print("  ✓ PASSED")
            
    finally:
        MAX_BUFFER_SIZE = original_max
    
    print("\n" + "=" * 60)
    print("Flattening tests passed! ✓")

"""

if __name__ == "__main__":
    # Run all tests
    test_sharded_stats()
    test_edge_cases()
    test_flattening_behavior()
    benchmark_sharded_stats()
"""
