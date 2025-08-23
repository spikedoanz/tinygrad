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

def smean(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False) -> Tensor:
    """Compute mean using convolution with identity kernel for chunking."""
    t = self
    output_dtype = t.dtype if dtypes.is_float(t.dtype) else dtypes.float32
    
    # Handle axis normalization
    if axis is None: 
        axis = list(range(len(t.shape)))
    elif isinstance(axis, int): 
        axis = [axis]
    
    # Normalize negative indices
    axis = [(a if a >= 0 else len(t.shape) + a) for a in axis]
    axis = sorted(axis)
    
    # Get preserved and reduced dimensions
    reduce_dims = axis
    preserved_dims = [i for i in range(len(t.shape)) if i not in reduce_dims]
    
    # If no dimensions to reduce, return as-is
    if not reduce_dims:
        return t
    
    # Check if we even need to use convolution approach
    total_size = prod(t.shape)
    reduce_size = prod([t.shape[d] for d in reduce_dims])
    
    if total_size <= MAX_BUFFER_SIZE:
        # Just use regular mean if small enough
        numerator = t.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=keepdim)
        denominator = reduce_size
        return numerator.div(denominator).cast(output_dtype)
    
    # Reshape tensor to separate preserved and reduced dimensions
    # Shape will be: [batch_size, reduce_size] where batch_size = product of preserved dims
    batch_size = prod([t.shape[d] for d in preserved_dims]) if preserved_dims else 1
    
    # Permute tensor to put reduce dims at the end
    perm = preserved_dims + reduce_dims
    if perm != list(range(len(t.shape))):
        t_permuted = t.permute(*perm)
    else:
        t_permuted = t
    
    # Reshape to [batch_size, reduce_size]
    t_reshaped = t_permuted.reshape(batch_size, reduce_size)
    
    # Calculate window size for convolution (chunk size that fits in memory)
    window_size = min(reduce_size, MAX_BUFFER_SIZE // batch_size)
    
    if window_size >= reduce_size:
        # Can do it in one go
        numerator = t_reshaped.cast(sum_acc_dtype(t.dtype)).sum(axis=1, keepdim=True)
        result = (numerator / reduce_size).cast(output_dtype)
    else:
        # Use convolution approach
        # Add channel dimension for conv: [batch_size, 1, reduce_size]
        t_conv = t_reshaped.unsqueeze(1)
        
        # Create identity kernel for averaging
        # Kernel shape: [1, 1, window_size]
        kernel = Tensor.ones(1, 1, window_size, dtype=sum_acc_dtype(t.dtype))
        
        # Apply 1D convolution with stride = kernel_size for non-overlapping windows
        # This effectively computes sum over each window
        conv_out = t_conv.conv2d(kernel, stride=window_size, padding=0)
        
        # conv_out shape: [batch_size, 1, num_windows]
        num_windows = conv_out.shape[-1]
        
        # Handle remainder if reduce_size is not divisible by window_size
        remainder = reduce_size % window_size
        if remainder > 0:
            # Process the remainder separately
            remainder_start = num_windows * window_size
            remainder_data = t_reshaped[:, remainder_start:].unsqueeze(1)
            remainder_sum = remainder_data.sum(axis=2, keepdim=True)
            
            # Concatenate with conv output
            conv_out = Tensor.cat(conv_out, remainder_sum, dim=2)
            
            # Create weights for proper averaging
            weights = Tensor([window_size] * num_windows + [remainder], dtype=sum_acc_dtype(t.dtype))
        else:
            weights = Tensor([window_size] * num_windows, dtype=sum_acc_dtype(t.dtype))
        
        # Compute weighted average
        # Sum all windows
        total_sum = conv_out.sum(axis=2, keepdim=True)
        
        # Divide by total count
        result = (total_sum / reduce_size).squeeze(1).cast(output_dtype)
    
    # Reshape back to original shape structure
    if keepdim:
        # Create output shape with 1s in reduced dimensions
        output_shape = list(t.shape)
        for d in reduce_dims:
            output_shape[d] = 1
        result = result.reshape(*output_shape)
    else:
        # Create output shape without reduced dimensions
        output_shape = [t.shape[d] for d in preserved_dims]
        if output_shape:
            result = result.reshape(*output_shape)
        else:
            # Scalar result
            result = result.squeeze()
    
    return result
