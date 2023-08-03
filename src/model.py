"""Neuroverkon toteutus."""

import numpy as np

from activation_funtions import activation_function_factory


class NeuralNetwork:
    """Neuroverkko."""

    def __init__(self, layers, activation_function="sigmoid"):
        """
        _summary_.

        Args:
            layers (list):  Neuroit per kerros. [0] input ja [-1] output. Väliin jäävät
                            ovat piilokerroksia.
            activation_function (str, optional): Defaults to "sigmoid".
        """
        self.n_layers = len(layers)
        self.init_weights_and_biases(layers)

        af, adf = activation_function_factory(activation_function)
        self.activation_function = af
        self.activation_function_derivative = adf

    def init_weights_and_biases(self, layers):
        """
        Alustetaan painot (weight) ja vakiotermit (bias).
        Args:
            layers (list)
        """
        self.biases = [np.random.random(n_neurons, 1) for n_neurons in layers]
        self.weights = [
            np.random.random(n_neurons, n_neurons_prev_layer)
            for n_neurons, n_neurons_prev_layer in zip(layers[:1], layers[:-1])
        ]

    def fit(self, X, y):
        """
        _summary_.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        pass

    def backpropagation(self, X, y):
        """
        Vastavirta-algoritmi (backpropagation).

        Tavoitteena on minimoida opetusesimerkkijoukkoa vastaava virhefunktio ja löytää minimointia vastaavat painot neuroneille.

        Osittaisderivaatat ja jokaisen neuronin vaikutus virheeseen lasketaan usein vastavirta-algoritmilla (backpropagation).

        Args:
            X (numpy array): Kuvien vektorit
            y (numpy array): Kuvien labelit
        """
        # Feedforward, lasketaan aktivoinnit. Lasketaan z^1 = w^l * a^l−1 + b^l ja a^l=σ(z^l).
        # Aloitetaan ensimmäisestä kerroksesta
        a_vectors = [X]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a_vectors[-1]) + b
            z_vectors.append(z)
            a_vectors.append(self.activation_function(z))

        # Nyt on laskettu aktivoinnit eteenpäin, lähdetään taakse
        # δ^L = ∇aC ⊙ σ′(z^L), missä ∇aC on virhefunktion derivaatan arvo
        # ja σ′ aktivointifunktion derivaatta
        delta = self.cost_derivative(
            a_vectors[-1], y
        ) * self.activation_function_derivative(z_vectors[-1])

        # täytetään käänteisessä järjestyksessä
        # https://tim.jyu.fi/view/143092#osittaisderivaatat-piilokerroksen-painojen-w_ijl-suhteen
        delta_weights = [np.dot(delta, a_vectors[-2].transpose())]
        # https://tim.jyu.fi/view/143092#osittaisderivaatat-piilokerroksen-vakiokertoimien-b_jl-suhteen
        # viimeisen kerroksen L arvo on delta
        delta_biases = [delta]

        # Lasketaan seuraavat kerrokset, kuljetaan takaperin
        # δ^l = ((w^l+1)^T * δ^l+1) ⊙ σ′(z^l)
        for l in range(1, self.n_layers - 1, -1):
            delta = np.dot(
                self.weights[l - 1].transpose(), delta
            ) * self.activation_function_derivative(z_vectors[l])
            delta_weights.append(np.dot(delta, a_vectors[l + 1].transpose()))
            delta_biases.append(delta)
            
        return (delta_weights[::-1], delta_biases[::-1])
