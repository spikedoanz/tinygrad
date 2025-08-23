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
    """Split tensor along specified axis into chunks smaller than max_size using split()."""
    if axis < 0: axis = len(t.shape) + axis
    axis_size = t.shape[axis]
    
    # Only chunk if the axis dimension itself is larger than max_size
    if axis_size <= max_size:
        return [t]
    
    # Calculate chunk size
    num_chunks = ceildiv(axis_size, max_size)
    chunk_size = ceildiv(axis_size, num_chunks)
    
    try:
        # Use split instead of tensor slicing
        chunks = t.split(chunk_size, dim=axis)
        return list(chunks) if isinstance(chunks, tuple) else chunks
    except Exception as e:
        print(f"Split failed: {e}, falling back to single tensor")
        return [t]

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
    
    # Check if we even need to chunk
    total_size = prod(t.shape)
    if total_size <= MAX_BUFFER_SIZE:
        # Just use regular mean if small enough
        numerator = t.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=keepdim)
        denominator = prod([t.shape[d] for d in reduce_dims])
        return numerator.div(denominator).cast(output_dtype)
    
    # Find the largest reduce dimension to chunk along
    largest_reduce_dim = max(reduce_dims, key=lambda d: t.shape[d])
    
    # Chunk along this dimension using split
    try:
        sharded = chunk(t, axis=largest_reduce_dim, max_size=MAX_BUFFER_SIZE)
    except Exception as e:
        print(f"Chunking failed: {e}, using regular mean")
        # Fallback to regular mean
        numerator = t.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=keepdim)
        denominator = prod([t.shape[d] for d in reduce_dims])
        return numerator.div(denominator).cast(output_dtype)
    
    # For each chunk, compute the mean
    sum_chunks = []
    count_chunks = []
    
    for s in sharded:
        try:
            s = s.realize()  # Materialize the chunk
            # Compute sum over all reduce dimensions
            chunk_sum = s.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=True)
            chunk_sum = chunk_sum.realize()
            
            # Calculate the count - product of all reduced dimensions in this chunk
            chunk_count = prod([s.shape[d] for d in reduce_dims])
            
            sum_chunks.append(chunk_sum)
            count_chunks.append(chunk_count)
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    if not sum_chunks:
        # All chunks failed, fallback to regular mean
        numerator = t.cast(sum_acc_dtype(t.dtype)).sum(axis=axis, keepdim=keepdim)
        denominator = prod([t.shape[d] for d in reduce_dims])
        return numerator.div(denominator).cast(output_dtype)
    
    # Combine chunks with proper weighting
    total_sum = sum(sum_chunks)
    total_count = sum(count_chunks)
    result = (total_sum / total_count).cast(output_dtype)
    
    # Handle keepdim
    if not keepdim:
        # Squeeze the reduced dimensions
        for d in sorted(reduce_dims, reverse=True):
            result = result.squeeze(d)
    
    return result
