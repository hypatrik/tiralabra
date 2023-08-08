import numpy as np
import pytest

from utilities import InvalidNumericInputException, split_every, init_weights_and_biases, vectorize_label


def test_split_every_one_element():
    test_set = [1, 2, 3, 4, 5]
    result = [[1], [2], [3], [4], [5]]
    assert split_every(1, test_set) == result


def test_split_every_two_elements():
    test_set = [1, 2, 3, 4, 5, 6]
    result = [[1, 2], [3, 4], [5, 6]]
    assert split_every(2, test_set) == result


def test_split_every_three_elements():
    test_set = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert split_every(3, test_set) == result


def test_split_every_n_not_multidue_of_len_list():
    test_set = [1, 2, 3, 4, 5, 6, 7]
    result = [[1, 2, 3, 4], [5, 6, 7]]
    assert split_every(4, test_set) == result


def test_split_every_test_empty_list():
    test_set = []
    result = []
    assert split_every(2, test_set) == result


def test_init_weights_and_biases():
    test_layers = [[2, 3, 1], [5, 10, 7, 3], [10, 20, 15, 8, 4]]
    for layers in test_layers:
        weights, biases = init_weights_and_biases(layers)
        assert len(biases) == len(layers) - 1
        assert len(weights) == len(layers) - 1

        for i in range(len(layers) - 1):
            assert biases[i].shape == (layers[i + 1], 1)

            if i > 0:
                assert weights[i].shape == (layers[i + 1], layers[i])

            assert isinstance(weights[i], np.ndarray)
            assert isinstance(biases[i], np.ndarray)

            assert weights[i].size > 0
            assert biases[i].size > 0

            assert np.all(np.isfinite(weights[i]))
            assert np.all(np.isfinite(biases[i]))


def test_vectorize_label_valid_input():
    number = 5
    result = vectorize_label(number)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 1)
    assert result[number] == 1.0
    assert np.all(result[np.arange(10) != number] == 0.0)


def test_vectorize_label_invalid_input():
    with pytest.raises(InvalidNumericInputException):
        vectorize_label(-1)

    with pytest.raises(InvalidNumericInputException):
        vectorize_label(10)
