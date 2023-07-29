from activation_funtions import activation_function_factory

class NeuralNetwork:
    def __init__(self, activation_function='sigmoid'):
        self.activation_function = activation_function_factory(activation_function)

    def fit(self, X, y):
        pass