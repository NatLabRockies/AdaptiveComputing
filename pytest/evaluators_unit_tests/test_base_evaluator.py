from adaptive_computing.evaluators import BaseEvaluator
import numpy as np
import pytest

def test_base_evaluator():
    func = lambda x: x**2

    evl = BaseEvaluator(func)
    res = evl.evaluate_point(2)

    assert type(res) == np.ndarray
    assert res.shape == (1,)
    np.testing.assert_array_almost_equal(res, np.asarray([4]))


    res = evl.evaluate_point([2])
    assert type(res) == np.ndarray
    assert res.shape == (1,)
    np.testing.assert_array_almost_equal(res, np.asarray([4]))


    res = evl.evaluate_point(np.array([2]))
    
    assert type(res) == np.ndarray
    assert res.shape == (1,)
    np.testing.assert_array_almost_equal(res, np.asarray([4]))

    
    with pytest.raises(ValueError) as e_info:
        res = evl.evaluate_point([1,2])

    res = evl.evaluate_points([[1],[2]])
    assert type(res) == np.ndarray
    assert res.shape == (2,1)
    np.testing.assert_array_almost_equal(res, np.asarray([[1],[4]]))

    res = evl.evaluate_points(np.array([[1],[2]]))
    assert type(res) == np.ndarray
    assert res.shape == (2,1)
    np.testing.assert_array_almost_equal(res, np.asarray([[1],[4]]))

    with pytest.raises(ValueError) as e_info:
        res = evl.evaluate_points([[1,2]])


    func = lambda x: x[0]**2-x[1]

    evl = BaseEvaluator(func, n_in=2)
    res = evl.evaluate_point([2,1])
    assert res.shape == (1,)
    np.testing.assert_array_almost_equal(res, np.asarray([3]))

    with pytest.raises(ValueError) as e_info:
         res = evl.evaluate_points([[1],[2]])


    evl = BaseEvaluator(func, n_in=2)
    res = evl.evaluate_points([[2,1],[0,0]])
    assert res.shape == (2,1)
    np.testing.assert_array_almost_equal(res, np.asarray([[3],[0]]))


if __name__ == "__main__":
    test_base_evaluator()
