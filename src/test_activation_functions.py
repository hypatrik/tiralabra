import numpy as np
import pytest

from activation_funtions import (
    step,
    relu,
    sigmoid,
    sigmoid_derivative,
    activation_function_factory,
    NoSuchActivationFunctionException,
)
from testing_utils import assertAlmostEqual

test_set = np.array([-5, -2, -1, 0, 1, 2, 5])


def test_step():
    result = step(test_set, 0)
    for i in range(0, 3):
        assert result[i] == 0
    for i in range(3, len(test_set)):
        assert result[i] == 1


def test_relu():
    result = relu(test_set)
    for i in range(0, 4):
        assert result[i] == 0
    for i in range(4, len(test_set)):
        assert result[i] == test_set[i]


def test_sigmoid():
    result = sigmoid(np.array([0, 1, -1, 100, -100]))
    # Sigmoid-funktion ominaisuus: 0 on keskellä
    assertAlmostEqual(result[0], 0.5)
    assertAlmostEqual(result[1], 0.7310585786300049)
    assertAlmostEqual(result[2], 0.2689414213699951)
    # todella suuret arvot ~1
    assertAlmostEqual(result[3], 1)
    # todella pienet arvot ~0
    assertAlmostEqual(result[4], 0)

def test_sigmoid_derivative():
    result = sigmoid_derivative(np.array([0]))
    assertAlmostEqual(result[0], 0.25)


def test_activation_function_factory_function_exists():
    af, afd = activation_function_factory("sigmoid")
    assert 'function' in str(type(af))
    assert 'function' in str(type(afd))


def test_activation_function_factory_function_not_exists():
    with pytest.raises(NoSuchActivationFunctionException) as exc_info:
        activation_function_factory("foobar")
    assert str(exc_info.value) == "No such activation function"
