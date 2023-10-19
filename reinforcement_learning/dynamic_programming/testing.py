import numpy as np


def is_not_None(variable):
    if variable is None:
        print("variable must not be None")
        return False
    return True

def is_type(variable, expected_type):
    if not type(expected_type) is list:
        expected_type = [expected_type]
    if not any([type(variable) is t for t in expected_type]):
        print(f'expected someting in {expected_type} but got type: {type(variable)}')
        return False
    return True

def numpy_array_has_shape(a, shape):
    if not is_type(a, np.ndarray): return False
    if not is_type(shape, tuple): return False
    if a.shape != shape:
        print(f'numpy arrays must have the shape {shape}, but received {a.shape}')
        return False
    return True
    

def numpy_array_is_close(a1, a2, atol=1e-9):
    if not is_type(a1, np.ndarray):
        print('a1 is not a numpy array')
        return False
    
    if not is_type(a2, np.ndarray):
        print('a2 is not a numpy array')
        return False
    
    if a1.shape != a2.shape:
        print(f'numpy arrays must have the same shape, but received {a1.shape} and {a2.shape}')
        return False
    
    if not np.isclose(a1, a2, atol=atol).all():
        print(f'numpy arrays are not equal: \na1:\n{a1}, \na2\n{a2}')
        return False
    return True


def test_function(function_expected, function_actual, compare_outputs_fun, *args, **kwargs):
    # print(f'checking function `{function_actual.__name__}`')
    # print(f'comparing to `{function_expected.__name__}`')
    # print(f'using `{compare_outputs_fun.__name__}` to compare outputs.')
    expected_output = function_expected(*args, **kwargs)
    actual_output = function_actual(*args, **kwargs)
    return compare_outputs_fun(expected_output, actual_output)
    
    
def test_the_tests():
    
    assert is_not_None(None) == False
    assert is_not_None(list()) == True
    assert is_not_None(1) == True
    assert is_not_None(np.array([1])) == True
    
    assert numpy_array_has_shape(np.zeros(tuple()), tuple()) == True
    assert numpy_array_has_shape(np.zeros((1,2,3)), (1,2,3)) == True
    assert numpy_array_has_shape(np.zeros((1,2,3)), ()) == False
    assert numpy_array_has_shape(np.zeros((1,2,3)), (1,)) == False
    assert numpy_array_has_shape(np.zeros((1,2,3)), (3)) == False
    assert numpy_array_has_shape(None, ()) == False
    
    assert is_type(None, type(None)) == True
    assert is_type(False, bool) == True
    assert is_type(None, float) == False
    assert is_type([], list) == True
    assert is_type(list, type) == True
    assert is_type(None, None) == False
    assert is_type(None, np.ndarray) == False
    assert is_type([1,2,3,4], np.ndarray) == False
    assert is_type(np.array([]), np.ndarray) == True
    assert is_type(None, [type(None)]) == True
    assert is_type(None, []) == False
    assert is_type(None, [[type(None)]]) == False
    
    assert numpy_array_is_close(np.array([1,2]), None) == False
    assert numpy_array_is_close(None, np.array([1,2])) == False
    assert numpy_array_is_close(np.array([1,2]), None) == False
    assert numpy_array_is_close(np.array([1,2]), np.zeros((10,10))) == False
    assert numpy_array_is_close(np.array([1,2]), np.array([3,2])) == False
    