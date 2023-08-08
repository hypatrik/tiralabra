import numpy as np

from sgd import stochastic_gradient_descent_fn, update_fn_factory

def test_update_fn():
    weights = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    biases = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    batch = [(np.array([1, 1, 1]), np.array([1, 1, 1]))]
    learning_rate = 0.5
    update_fn = update_fn_factory(backpropagation_fn=lambda x, y, _, __: (x, y))
    updated_weights, updated_biases = update_fn(weights, biases, batch, learning_rate)
    
    assert np.array_equal(updated_weights, [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])])
    assert np.array_equal(updated_biases, [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])])
    
    
def test_stochastic_gradient_descent_fn():
    np.random.seed(0)

    weights = np.array([0, 0, 0])
    biases = np.array([0, 0, 0])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    sgd_fn = stochastic_gradient_descent_fn(update_fn=lambda x, y, _, __: (x, y))
    sgd = sgd_fn(weights, biases, X, y, epochs=2, learning_rate=0.1, batch_size=1)

    for epoch, (out_weights, out_biases, out_epoch) in enumerate(sgd):
        assert np.array_equal(out_weights, weights)
        assert np.array_equal(out_biases, biases)
        assert out_epoch == epoch