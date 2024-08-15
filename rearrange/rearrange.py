import functools
import itertools
import string
import typing
import sys
from collections import OrderedDict
from typing import Set, Tuple, List, Dict, Union, Callable, Optional, TypeVar, cast, Any

import numpy as np
from tinygrad import Tensor as tg_tensor

import keyword
import warnings

_ellipsis: str = "…"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False

Tensor = TypeVar("Tensor")
ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]
Reduction = Union[str, ReductionCallable]

_reductions = ("min", "max", "sum", "mean", "prod", "any", "all")

# magic integers are required to stay within
# traceable subset of language
_unknown_axis_length = -999999
_expected_axis_length = -99999


def get_backend(tensor) -> "AbstractBackend":
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    _type = type(tensor)
    _result = _type2backend.get(_type, None)
    if _result is not None:
        return _result

    for framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _loaded_backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend

    raise RuntimeError("Tensor type unknown to einops {}".format(type(tensor)))

class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes."""

    framework_name: str

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        # supplementary method used only in testing, so should implement CPU version
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats - same lengths as x.shape"""
        raise NotImplementedError()

    def concat(self, tensors, axis: int):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    def is_float_type(self, x):
        # some backends (torch) can't compute average for non-floating types.
        # Decided to drop average for all backends if type is not floating
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError("backend does not provide layers")

    def __repr__(self):
        return "<einops backend for {}>".format(self.framework_name)

    def einsum(self, pattern, *x):
        raise NotImplementedError("backend does not support einsum")


class UnknownSize:
    """pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements"""

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return hash(None)


class NumpyBackend(AbstractBackend):
    framework_name = "numpy"

    def __init__(self):
        import numpy

        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def arange(self, start, stop):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        return self.np.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.np.concatenate(tensors, axis=axis)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)

    def einsum(self, pattern, *x):
        return self.np.einsum(pattern, *x)


class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        self.elements = elements

    def __iter__(self):
        for x in self.elements:
            yield x

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

class TinygradBackend(AbstractBackend):
    framework_name = "tinygrad"

    def __init__(self):
        import tinygrad

        self.tinygrad = tinygrad

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.tinygrad.Tensor)

    def from_numpy(self, x):
        return self.tinygrad.Tensor(x)

    def to_numpy(self, x):
        return x.numpy()

    def arange(self, start, stop):
        return self.tinygrad.Tensor.arange(start, stop)

    def shape(self, x):
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.permute(axes)

    def reduce(self, x, operation, axes):
        for axis in sorted(axes, reverse=True):
            x = getattr(x, operation)(axis=axis)
        return x

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tinygrad.Tensor.stack(tensors)

    def add_axis(self, x, new_position):
        return x.unsqueeze(new_position)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return tensors[0].cat(tensors[1:], axis) if len(tensors) > 1 else tensors[0]

    def is_float_type(self, x):
        return self.tinygrad.dtypes.is_float(x.dtype)

    def einsum(self, pattern, *x):
        return self.tinygrad.Tensor.einsum(pattern, *x)


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other"""

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise EinopsError("No need to create anonymous axis of length 1. Report this as an issue")
            else:
                raise EinopsError("Anonymous axis should have positive length, not {}".format(self.value))

    def __repr__(self):
        return "{}-axis".format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """

    def __init__(self, expression: str, *, allow_underscore: bool = False, allow_duplicates: bool = False):
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[str] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: List[Union[List[str], str]] = []
        if "." in expression:
            if "..." not in expression:
                raise EinopsError("Expression may contain dots only inside ellipsis (...)")
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise EinopsError(
                    "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "
                )
            expression = expression.replace("...", _ellipsis)
            self.has_ellipsis = True

        bracket_group: Optional[List[str]] = None

        def add_axis_name(x):
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise EinopsError('Indexing expression contains duplicate dimension "{}"'.format(x))
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise EinopsError("Invalid axis identifier: {}\n{}".format(x, reason))
                if is_number:
                    x = AnonymousAxis(x)
                self.identifiers.add(x)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([x])
                else:
                    bracket_group.append(x)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise EinopsError("Axis composition is one-level (brackets inside brackets not allowed)")
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise EinopsError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise EinopsError("Unknown character '{}'".format(char))

        if bracket_group is not None:
            raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
        if current_identifier is not None:
            add_axis_name(current_identifier)

    def flat_axes_order(self) -> List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), "does not work with ellipsis"
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) -> bool:
        # this will ignore 1 inside brackets
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool = False) -> Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn("It is discouraged to use axes names that are keywords: {}".format(name), RuntimeWarning)
            if name in ["axis"]:
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name " "and will raise an error in future",
                    FutureWarning,
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _reason = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid
if typing.TYPE_CHECKING:
    # for docstrings in pycharm
    import numpy as np  # noqa E401

class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""
    pass

def _product(sequence: List[int]) -> int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    result = 1
    for element in sequence:
        result *= element
    return result


def _reduce_axes(tensor, reduction_type: Reduction, reduced_axes: List[int], backend):
    if callable(reduction_type):
        # custom callable
        return reduction_type(tensor, tuple(reduced_axes))
    else:
        # one of built-in operations
        assert reduction_type in _reductions
        if reduction_type == "mean":
            if not backend.is_float_type(tensor):
                raise NotImplementedError("reduce_mean is not available for non-floating tensors")
        return backend.reduce(tensor, reduction_type, tuple(reduced_axes))

CookedRecipe = Tuple[Optional[List[int]], Optional[List[int]], List[int], Dict[int, int], Optional[List[int]], int]
HashableAxesLengths = Tuple[Tuple[str, int], ...]
FakeHashableAxesLengths = List[Tuple[str, int]]


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """
    def __init__(
        self,
        # list of sizes (or just sizes) for elementary axes as they appear in left expression.
        # this is what (after computing unknown parts) will be a shape after first transposition.
        # This does not include any ellipsis dimensions.
        elementary_axes_lengths: List[int],
        # if additional axes are provided, they should be set in prev array
        # This shows mapping from name to position
        axis_name2elementary_axis: Dict[str, int],
        # each dimension in input can help to reconstruct length of one elementary axis
        # or verify one of dimensions. Each element points to element of elementary_axes_lengths.
        input_composition_known_unknown: List[Tuple[List[int], List[int]]],
        # permutation applied to elementary axes, if ellipsis is absent
        axes_permutation: List[int],
        # permutation puts reduced axes in the end, we only need to know the first position.
        first_reduced_axis: int,
        # at which positions which of elementary axes should appear. Axis position -> axis index.
        added_axes: Dict[int, int],
        # ids of axes as they appear in result, again pointers to elementary_axes_lengths,
        # only used to infer result dimensions
        output_composite_axes: List[List[int]],
    ):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.axis_name2elementary_axis: Dict[str, int] = axis_name2elementary_axis
        self.input_composition_known_unknown: List[Tuple[List[int], List[int]]] = input_composition_known_unknown
        self.axes_permutation: List[int] = axes_permutation

        self.first_reduced_axis: int = first_reduced_axis
        self.added_axes: Dict[int, int] = added_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes


def _reconstruct_from_shape_uncached(
    self: TransformRecipe, shape: List[int], axes_dims: FakeHashableAxesLengths
) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    # magic number
    need_init_reshape = False

    # last axis is allocated for collapsed ellipsis
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim

    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composition_known_unknown):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            # shortcut for the most common case
            axes_lengths[unknown_axes[0]] = length
            continue

        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]

        if len(unknown_axes) == 0:
            if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                raise EinopsError(f"Shape mismatch, {length} != {known_product}")
        else:
            # assert len(unknown_axes) == 1, 'this is enforced when recipe is created, so commented out'
            if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                raise EinopsError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}")

            unknown_axis = unknown_axes[0]
            inferred_length: int = length // known_product
            axes_lengths[unknown_axis] = inferred_length

        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # elementary axes are ordered as they appear in input, then all added axes
    init_shapes: Optional[List[int]] = axes_lengths[: len(self.axes_permutation)] if need_init_reshape else None

    need_final_reshape = False
    final_shapes: List[int] = []
    for grouping in self.output_composite_axes:
        lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
        final_shapes.append(_product(lengths))
        if len(lengths) != 1:
            need_final_reshape = True

    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()
    }

    # this list can be empty
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))

    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)

    axes_reordering: Optional[List[int]] = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None

    _final_shapes = final_shapes if need_final_reshape else None
    return init_shapes, axes_reordering, reduced_axes, added_axes, _final_shapes, n_axes_after_adding_axes


_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)


def _apply_recipe(
    backend, recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths
) -> Tensor:
    # this method implements actual work for all backends for 3 operations
    try:
        init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = _reconstruct_from_shape(
            recipe, backend.shape(tensor), axes_lengths
        )
    except TypeError:
        # shape or one of passed axes lengths is not hashable (i.e. they are symbols)
        _result = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_lengths)
        (init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added) = _result
    if init_shapes is not None:
        tensor = backend.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = backend.transpose(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        tensor = _reduce_axes(tensor, reduction_type=reduction_type, reduced_axes=reduced_axes, backend=backend)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        tensor = backend.reshape(tensor, final_shapes)
    return tensor


@functools.lru_cache(256)
def _prepare_transformation_recipe(
    pattern: str,
    operation: Reduction,
    axes_names: Tuple[str, ...],
    ndim: int,
) -> TransformRecipe:
    """Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # checking that axes are in agreement - new axes appear only in repeat, while disappear only in reduction
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError("Ellipsis found in right side, but not left side of a pattern {}".format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError("Ellipsis inside parenthesis in the left side is not allowed: {}".format(pattern))
    if operation == "rearrange":
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Identifiers only on one side of expression (should be on both): {}".format(difference))
    elif operation == "repeat":
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the left side of repeat: {}".format(difference))
        axes_without_size = set.difference(
            {ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)},
            {*left.identifiers, *axes_names},
        )
        if len(axes_without_size) > 0:
            raise EinopsError("Specify sizes for new axes in repeat: {}".format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the right side of reduce {}: {}".format(operation, difference))
    else:
        raise EinopsError("Unknown reduction {}. Expect one of {}.".format(operation, _reductions))

    if left.has_ellipsis:
        n_other_dims = len(left.composition) - 1
        if ndim < n_other_dims:
            raise EinopsError(f"Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.")
        ellipsis_ndim = ndim - n_other_dims
        ell_axes = [_ellipsis + str(i) for i in range(ellipsis_ndim)]
        left_composition = []
        for composite_axis in left.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    left_composition.append([axis])
            else:
                left_composition.append(composite_axis)

        rght_composition = []
        for composite_axis in rght.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    rght_composition.append([axis])
            else:
                group = []
                for axis in composite_axis:
                    if axis == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(axis)
                rght_composition.append(group)

        left.identifiers.update(ell_axes)
        left.identifiers.remove(_ellipsis)
        if rght.has_ellipsis:
            rght.identifiers.update(ell_axes)
            rght.identifiers.remove(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.")
        left_composition = left.composition
        rght_composition = rght.composition

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left_composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length

    # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)

    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}

    # axes provided as kwargs
    for elementary_axis in axes_names:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError("Invalid name for an axis", elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError("Axis {} is not used in transform".format(elementary_axis))
        axis_name2known_length[elementary_axis] = _expected_axis_length

    input_axes_known_unknown = []
    # some shapes are inferred later - all information is prepared for faster inference
    for i, composite_axis in enumerate(left_composition):
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError("Could not infer sizes for {}".format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(
            ([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown])
        )

    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left_composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

    result_axes_grouping: List[List[int]] = [
        [axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(rght_composition)
    ]

    ordered_axis_left = list(itertools.chain(*left_composition))
    ordered_axis_rght = list(itertools.chain(*rght_composition))
    reduced_axes = [axis for axis in ordered_axis_left if axis not in rght.identifiers]
    order_after_transposition = [axis for axis in ordered_axis_rght if axis in left.identifiers] + reduced_axes
    axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
    added_axes = {
        i: axis_name2position[axis_name]
        for i, axis_name in enumerate(ordered_axis_rght)
        if axis_name not in left.identifiers
    }

    first_reduced_axis = len(order_after_transposition) - len(reduced_axes)

    return TransformRecipe(
        elementary_axes_lengths=list(axis_name2known_length.values()),
        axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names},
        input_composition_known_unknown=input_axes_known_unknown,
        axes_permutation=axes_permutation,
        first_reduced_axis=first_reduced_axis,
        added_axes=added_axes,
        output_composite_axes=result_axes_grouping,
    )


def _prepare_recipes_for_all_dims(
    pattern: str, operation: Reduction, axes_names: Tuple[str, ...]
) -> Dict[int, TransformRecipe]:
    """
    Internal function, used in layers.
    Layer makes all recipe creation when it is initialized, thus to keep recipes simple we pre-compute for all dims
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    dims = [len(left.composition)]
    if left.has_ellipsis:
        dims = [len(left.composition) - 1 + ellipsis_dims for ellipsis_dims in range(8)]
    return {ndim: _prepare_transformation_recipe(pattern, operation, axes_names, ndim=ndim) for ndim in dims}


def reduce(tensor: Union[Tensor, List[Tensor]], pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    try:
        if isinstance(tensor, list):
            if len(tensor) == 0:
                raise TypeError("Rearrange/Reduce/Repeat can't be applied to an empty list")
            backend = get_backend(tensor[0])
            tensor = backend.stack_on_zeroth_dimension(tensor)
        else:
            backend = get_backend(tensor)

        hashable_axes_lengths = tuple(axes_lengths.items())
        shape = backend.shape(tensor)
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=len(shape))
        return _apply_recipe(
            backend, recipe, cast(Tensor, tensor), reduction_type=reduction, axes_lengths=hashable_axes_lengths
        )
    except EinopsError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(tensor, list):
            message += "\n Input tensor shape: {}. ".format(shape)
        else:
            message += "\n Input is list. "
        message += "Additional info: {}.".format(axes_lengths)
        raise EinopsError(message + "\n {}".format(e))


def rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    return reduce(tensor, pattern, reduction="rearrange", **axes_lengths)
