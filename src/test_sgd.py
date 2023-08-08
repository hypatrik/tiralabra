import numpy as np

from sgd import update_fn_factory

def test_update_fn():
    weights = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    biases = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    batch = [(np.array([1, 1, 1]), np.array([1, 1, 1]))]
    learning_rate = 0.5
    update_fn = update_fn_factory(backpropagation_fn=lambda x, y, _, __: (x, y))
    updated_weights, updated_biases = update_fn(weights, biases, batch, learning_rate)
    
    assert np.array_equal(updated_weights, [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])])
    assert np.array_equal(updated_biases, [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])])