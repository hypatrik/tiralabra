"""Työkalu opetetun neuroverkon tallentamiseen ja lataanmiseen."""

import pickle
import gzip

from model import NeuralNetwork


class PresistableModel:
    """PresistableModel luokalla voi tallentaa ja ladata NeuralNetwork luokan."""

    def __init__(self, model):
        """Konstruktori.

        Args:
            model (NeuralNetwork): Opetettu malli.
        """
        self.layers = model.layers
        self.weights = model.weights
        self.biases = model.biases

    def save(self, model_name):
        """Tallenna malli.

        Args:
            model_name (string): Mallin nimi. Tallentaan projektin juuressa olevaan data kansioon.
        """
        with gzip.GzipFile("../data/{}.pkl.gz".format(model_name), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(model_name):
        """Lataa tallennettu malli.

        Args:
            model_name (string): Mallin nimi.

        Returns:
            NeuralNetwork: Mallin instanssi.
        """
        with gzip.GzipFile("../data/{}.pkl.gz".format(model_name), "rb") as file:
            saved_model = pickle.load(file)
        model = NeuralNetwork(saved_model.layers)
        model.weights = saved_model.weights
        model.biases = saved_model.biases

        return model
