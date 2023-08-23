import numpy as np
import pytest

from activation_funtions import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
)
from testing_utils import assertAlmostEqual

test_set = np.array([-5, -2, -1, 0, 1, 2, 5])


def test_relu():
    relu = ReLU()
    result = relu.activation(test_set)
    for i in range(0, 4):
        assert result[i] == 0
    for i in range(4, len(test_set)):
        assert result[i] == test_set[i]

def test_relu_derivative():
    relu = ReLU()
    result = relu.derivative(test_set)
    for i in range(0, 4):
        assert result[i] == 0
    for i in range(4, len(test_set)):
        assert result[i] == 1


def test_leaky_relu():
    relu = LeakyReLU()
    result = relu.activation(test_set)
    for i in range(0, 4):
        assert result[i] == (test_set[i] * 0.01)
    for i in range(4, len(test_set)):
        assert result[i] == test_set[i]

def test_leaky_relu_derivative():
    relu = LeakyReLU()
    result = relu.derivative(test_set)
    for i in range(0, 4):
        assert result[i] == 0.01
    for i in range(4, len(test_set)):
        assert result[i] == 1


def test_sigmoid():
    sigmoid = Sigmoid()
    result = sigmoid.activation(np.array([0, 1, -1, 100, -100]))
    # Sigmoid-funktion ominaisuus: 0 on keskellä
    assertAlmostEqual(result[0], 0.5)
    assertAlmostEqual(result[1], 0.7310585786300049)
    assertAlmostEqual(result[2], 0.2689414213699951)
    # todella suuret arvot ~1
    assertAlmostEqual(result[3], 1)
    # todella pienet arvot ~0
    assertAlmostEqual(result[4], 0)

def test_sigmoid_derivative():
    sigmoid = Sigmoid()
    result = sigmoid.derivative(np.array([0]))
    assertAlmostEqual(result[0], 0.25)
   
def test_tanh():
    tanh = Tanh()
    result = tanh.activation(np.array([0, 1, -1, 100, -100]))
    # Sigmoid-funktion ominaisuus: 0 on keskellä
    assertAlmostEqual(result[0], 0)
    assertAlmostEqual(result[1], 0.7615941559557649)
    assertAlmostEqual(result[2], -0.7615941559557649)
    # todella suuret arvot ~1
    assertAlmostEqual(result[3], 1)
    # todella pienet arvot ~-1
    assertAlmostEqual(result[4], -1)  
    
def test_tanh_derivative():
    tanh = Tanh()
    result = tanh.derivative(np.array([0]))
    assertAlmostEqual(result[0], 1)
