"""Neuroverkon toteutus."""

import numpy as np
import random

from activation_funtions import activation_function_factory
from cost_functions import quadratic_cost_function_derivative
from utilities import split_every


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
        self.cost_function_derivative = quadratic_cost_function_derivative

    def init_weights_and_biases(self, layers):
        """
        Alustetaan painot (weight) ja vakiotermit (bias).
        Args:
            layers (list)
        """
        self.biases = [np.random.randn(n_neurons, 1) for n_neurons in layers]
        self.weights = [
            np.random.randn(n_neurons, n_neurons_prev_layer)
            for n_neurons, n_neurons_prev_layer in zip(layers[:1], layers[:-1])
        ]

    def fit(self, X, y, learning_rate, epochs, batch_size):
        """
        Stokastinen gradienttimenetelmä.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """

        training_data = list(zip(X, y))
        print('Traning with {} elements'.format(len(X)))

        for i in range(epochs):
            random.shuffle(training_data)
            for batch in split_every(batch_size, training_data):
                print("batch {}".format(len(batch)))
                self.update(batch, learning_rate)
            print("epoch {} done".format(i))

    def update(self, batch, learning_rate):
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]

        batch_size = len(batch)

        for x, y in batch:
            delta_weights, delta_biases = self.backpropagation(x, y)
            weight_gradients = [
                w_gradient + delta_w
                for w_gradient, delta_w in zip(weight_gradients, delta_weights)
            ]
            bias_gradients = [
                b_gradient + delta_nb
                for b_gradient, delta_nb in zip(bias_gradients, delta_biases)
            ]

        self.weights = [
            w - learning_rate / batch_size * w_gradient
            for w, w_gradient in zip(self.weights, weight_gradients)
        ]
        self.biases = [
            b - learning_rate / batch_size * b_gradient
            for b, b_gradient in zip(self.biases, bias_gradients)
        ]

    def backpropagation(self, x, y):
        """
        Vastavirta-algoritmi (backpropagation).

        https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista

        Tavoitteena on minimoida opetusesimerkkijoukkoa vastaava virhefunktio ja löytää minimointia vastaavat painot neuroneille.

        Osittaisderivaatat ja jokaisen neuronin vaikutus virheeseen lasketaan usein vastavirta-algoritmilla (backpropagation).

        Args:
            X (numpy array): Kuvien vektorit
            y (numpy array): Kuvien labelit
        """
        # Feedforward, lasketaan aktivoinnit. Lasketaan z^1 = w^l * a^l−1 + b^l ja a^l=σ(z^l).
        # Aloitetaan ensimmäisestä kerroksesta
        a_vectors = [np.zeros(b.shape) for b in self.biases]
        z_vectors = [np.zeros(w.shape) for w in self.weights]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a_vectors[-1]) + b
            z_vectors.append(z)
            a_vectors.append(self.activation_function(z))

        # Nyt on laskettu aktivoinnit eteenpäin, lähdetään taakse
        # δ^L = ∇aC ⊙ σ′(z^L), missä ∇aC on virhefunktion derivaatan arvo
        # ja σ′ aktivointifunktion derivaatta
        delta = self.cost_function_derivative(
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
        for l in range(2, self.n_layers):
            delta = np.dot(
                self.weights[l + 1].transpose(), delta
            ) * self.activation_function_derivative(z_vectors[-l])
            delta_weights.append(np.dot(delta, a_vectors[l - 1].transpose()))
            delta_biases.append(delta)

        return (delta_weights, delta_biases)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a)+b)
        return a

    def predict(self, input_vector):
        return np.argmax(self.feedforward(input_vector))