from utilities import split_every

def test_split_every_one_element():
    test_set = [1, 2, 3, 4, 5]
    result = [[1], [2], [3], [4], [5]]
    assert split_every(1, test_set) == result
    assert test_set == [1, 2, 3, 4, 5]

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
