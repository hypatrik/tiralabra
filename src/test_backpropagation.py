import numpy as np

from backpropagation import backpropagation_fn_factory

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

# Luodaan backpropagation funktio yksinkertaisemmalla aktivoinnilla
# helpottaaksemme testaamista
backpropagation = backpropagation_fn_factory(
    activation_function=identity,
    activation_function_derivative=identity_derivative,
)

def test_backpropagation_shapes():
    weights = [np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]), np.array([[0.1, 0.2]])]
    biases = [np.array([[0.1], [0.1]]), np.array([[0.1]])]

    x = np.array([[0.5], [0.6], [0.7]])
    y = np.array([[0.1]])
    
    gradients_weights, gradients_bias = backpropagation(weights, biases, x, y)

    
    assert len(gradients_weights) == len(weights)
    assert len(gradients_bias) == len(biases)

    for gw, w in zip(gradients_weights, weights):
        assert gw.shape == w.shape

    for gb, b in zip(gradients_bias, biases):
        assert gb.shape == b.shape

    # Test that the gradients are numpy arrays
    for gw in gradients_weights:
        assert isinstance(gw, np.ndarray)

    for gb in gradients_bias:
        assert isinstance(gb, np.ndarray)
        
        
def test_backpropagation_values():
    weights = [np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), np.array([[0.5, 0.5]])]
    biases = [np.array([[0.1], [0.1]]), np.array([[0.1]])]

    x = np.array([[0.5], [0.6], [0.7]])
    y = np.array([[0.1]])

    gradients_weights, gradients_bias = backpropagation(weights, biases, x, y)

    # ensimmäinin ja toinen kerros
    for i, x in enumerate([0.25, 0.3, 0.35]):
        assert gradients_weights[0][0][i] == x
        assert gradients_weights[0][1][i] == x
    
    # kolmas kerros
    for i, x in enumerate([1, 1]):
        assert gradients_weights[1][0][i] == x
        
    assert gradients_bias[0][0][0] == 0.5
    assert gradients_bias[0][1][0] == 0.5
    assert gradients_bias[1][0][0] == 1
        