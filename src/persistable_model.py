import pickle
import gzip

from model import NeuralNetwork

class PresistableModel:
    def __init__(self, model):
        self.layers = model.layers
        self.weights = model.weights
        self.biases = model.biases

    def save(self, model_name):
        with gzip.GzipFile("../data/{}.pkl.gz".format(model_name), "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(model_name):
        with gzip.GzipFile("../data/{}.pkl.gz".format(model_name), "rb") as file:
            saved_model = pickle.load(file)
        model = NeuralNetwork(saved_model.layers)
        model.weights = saved_model.weights
        model.biases = saved_model.biases
        
        return model