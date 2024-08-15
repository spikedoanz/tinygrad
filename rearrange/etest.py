import numpy as np
import pytest
from einops import rearrange as spec

# Assume 'rearrange' is your implementation to be tested
from rearrange import rearrange

def run_all_tests():
    tests = [
        test_simple_transpose,
        test_simple_reshape,
        test_combine_axes,
        test_split_axis,
        test_ellipsis,
        test_newaxis,
        test_complex_rearrange,
        test_named_axes,
        test_large_tensor,
        test_zero_dim_tensor,
        test_empty_tensor,
        # test_reduce_axes,
        # test_repeat_axis,
        # test_broadcasting,
        # test_error_invalid_pattern,
        # test_error_shape_mismatch,
        # test_error_unknown_axis,
        # test_multiple_ellipsis
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append((test.__name__, "Passed"))
        except AssertionError as e:
            results.append((test.__name__, f"Failed: {str(e)}"))
        except Exception as e:
            results.append((test.__name__, f"Error: {str(e)}"))
    
    return results

def test_simple_transpose():
    x = np.array([[1, 2], [3, 4]])
    expected = spec(x, 'h w -> w h')
    assert np.array_equal(rearrange(x, 'h w -> w h'), expected)

def test_simple_reshape():
    x = np.array([1, 2, 3, 4])
    expected = spec(x, '(h w) -> h w', h=2, w=2)
    assert np.array_equal(rearrange(x, '(h w) -> h w', h=2, w=2), expected)

def test_combine_axes():
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected = spec(x, 'a b c -> a (b c)')
    assert np.array_equal(rearrange(x, 'a b c -> a (b c)'), expected)

def test_split_axis():
    x = np.array([1, 2, 3, 4, 5, 6])
    expected = spec(x, '(a b) -> a b', b=2)
    assert np.array_equal(rearrange(x, '(a b) -> a b', b=2), expected)

def test_repeat_axis():
    x = np.array([1, 2, 3])
    expected = spec(x, 'a -> a b', b=2)
    assert np.array_equal(rearrange(x, 'a -> a b', b=2), expected)

def test_ellipsis():
    x = np.random.rand(2, 3, 4, 5)
    expected1 = spec(x, '... -> ...')
    expected2 = spec(x, 'a ... d -> d ... a')
    assert np.array_equal(rearrange(x, '... -> ...'), expected1)
    assert np.array_equal(rearrange(x, 'a ... d -> d ... a'), expected2)

def test_newaxis():
    x = np.array([1, 2, 3])
    expected = spec(x, 'a -> a 1 1')
    assert np.array_equal(rearrange(x, 'a -> a 1 1'), expected)

def test_reduce_axes():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected = spec(x, 'a b -> a', b='sum')
    assert np.array_equal(rearrange(x, 'a b -> a', b='sum'), expected)

def test_complex_rearrange():
    x = np.random.rand(2, 3, 4, 5)
    expected = spec(x, 'a b c d -> b (a c) d')
    assert np.array_equal(rearrange(x, 'a b c d -> b (a c) d'), expected)

def test_named_axes():
    x = np.random.rand(2, 3, 4)
    expected = spec(x, 'batch channels time -> time batch channels')
    assert np.array_equal(rearrange(x, 'batch channels time -> time batch channels'), expected)

def test_broadcasting():
    x = np.array([1, 2, 3])
    expected = spec(x, 'a -> a b', b=2)
    assert np.array_equal(rearrange(x, 'a -> a b', b=2), expected)

def test_error_invalid_pattern():
    x = np.array([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        rearrange(x, 'a -> b')
    assert "Identifiers only on one side of expression" in str(excinfo.value)

def test_error_shape_mismatch():
    x = np.array([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        rearrange(x, '(a b) -> a b', a=2, b=2)
    assert "Shape mismatch" in str(excinfo.value)

def test_error_unknown_axis():
    x = np.array([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        rearrange(x, 'a -> b', b=3)
    assert "Identifiers only on one side of expression" in str(excinfo.value)

def test_large_tensor():
    x = np.random.rand(100, 100, 100)
    expected = spec(x, 'a b c -> b c a')
    assert np.array_equal(rearrange(x, 'a b c -> b c a'), expected)

def test_zero_dim_tensor():
    x = np.array(5)
    expected = spec(x, '... -> ...')
    assert np.array_equal(rearrange(x, '... -> ...'), expected)

def test_empty_tensor():
    x = np.array([])
    expected = spec(x, '... -> ...')
    assert np.array_equal(rearrange(x, '... -> ...'), expected)

def test_multiple_ellipsis():
    x = np.random.rand(2, 3, 4, 5)
    with pytest.raises(ValueError) as excinfo:
        rearrange(x, '... a ... -> ...')
    assert "Expression may contain dots only inside ellipsis" in str(excinfo.value)

if __name__ == "__main__":
    results = run_all_tests()
    for test_name, result in results:
        print(f"{test_name}: {result}")
