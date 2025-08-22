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

def chunk(t: Tensor, axis: int, max_size: int = MAX_BUFFER_SIZE) -> list[Tensor]:
    """Split tensor along specified axis into chunks smaller than max_size."""
    if axis < 0: axis = len(t.shape) + axis
    axis_size = t.shape[axis]
    
    # Only chunk if the axis dimension itself is larger than max_size
    if axis_size <= max_size:
        return [t]
    
    # Calculate number of chunks needed for this axis
    num_chunks = ceildiv(axis_size, max_size)
    chunk_size = ceildiv(axis_size, num_chunks)
    
    chunks = []
    for i in range(0, axis_size, chunk_size):
        end = min(i + chunk_size, axis_size)
        slices = [slice(None)] * len(t.shape)
        slices[axis] = slice(i, end)
        
        chunk = t[tuple(slices)]
        chunks.append(chunk)
    
    return chunks

def unchunk(ts: list[Tensor], axis: int) -> Tensor:
    return Tensor.cat(*ts, dim=axis)

def mean(t, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
    output_dtype = t.dtype if dtypes.is_float(t.dtype) else dtypes.float32
    numerator = t.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=keepdim)
    return numerator.div(prod([cast(int, si) for si, so in zip(t.shape, t.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])) \
      .cast(output_dtype)

def smean(self: Tensor, axis: int|Sequence[int]|None = None, keepdim: bool = False) -> Tensor:
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
    
    # Permute to put preserved dims first, then reduce dims
    perm = preserved_dims + reduce_dims
    t_permuted = t.permute(*perm) if perm else t
    
    # Calculate new shape: flatten all reduce dims into one
    preserved_shape = [t.shape[d] for d in preserved_dims] if preserved_dims else []
    reduce_size = prod([t.shape[d] for d in reduce_dims])
    
    # Reshape to [...preserved_dims..., flattened_reduce_dim]
    new_shape = preserved_shape + [reduce_size] if preserved_shape else [reduce_size]
    t_reshaped = t_permuted.reshape(*new_shape)
    
    # Now we have a single axis to reduce (the last one)
    reduction_axis = len(new_shape) - 1
    
    # Chunk along the flattened reduction dimension
    sharded = chunk(t_reshaped, axis=reduction_axis, max_size=MAX_BUFFER_SIZE)
    
    # Compute weighted mean
    sum_chunks = []
    count_chunks = []
    
    for s in sharded:
        s.realize()  # Materialize chunk
        sum_chunk = s.cast(sum_acc_dtype(t.dtype)).sum(axis=reduction_axis, keepdim=True)
        sum_chunk.realize()
        sum_chunks.append(sum_chunk)
        count_chunks.append(s.shape[reduction_axis])
    
    # Combine chunks with proper weighting
    total_sum = sum(sum_chunks)
    total_count = sum(count_chunks)
    result = (total_sum / total_count).cast(output_dtype)
    
    # Reshape back to original structure
    if keepdim:
        # Build final shape with 1s in reduced dimensions
        if preserved_dims:
            # Reshape to add 1s for reduced dims
            result = result.reshape(*preserved_shape, *([1] * len(reduce_dims)))
            # Unpermute to restore original dimension order
            inv_perm = [perm.index(i) for i in range(len(perm))]
            result = result.permute(*inv_perm)
        else:
            # All dims were reduced, all become 1
            result = result.reshape(*[1] * len(self.shape))
    else:
        # Result already has correct shape (just preserved dims)
        if preserved_dims:
            result = result.squeeze(-1)  # Remove the reduction dimension
        else:
            result = result.squeeze()  # Scalar result
    return result
