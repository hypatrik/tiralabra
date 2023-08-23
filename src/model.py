"""Neuroverkko."""

import numpy as np
from activation_funtions import Sigmoid

from sgd import stochastic_gradient_descent_fn, update_fn_factory
from backpropagation import backpropagation_fn_factory
from utilities import calculate_z, init_weights_and_biases


class NeuralNetwork:
    """
    Neuroverkko luokka.

    Tämä luokka kokoaa yhteen Neuroverkon eri osa-alueet ja tarjoaa
    helpon rajapinnan neuroverkon opettamiseen.
    """

    def __init__(self, layers, activation_function=Sigmoid()) -> None:
        """
        Konstruktori.

        Args:
            layers (list):  Neuroverkon kerroskokoonpano esim (784, 30, 16, 10).
                            Huomaa, että ensimmäinen tulee olla kuva vektorin koko 784 ja
                            viimeinen 10 eli mahdollisten lopputulosten määrä.

            activation_function (str):  Aktoivointifunktio,katso
                                        activation_function#activation_function_factory. Defaults to "sigmoid".
        """

        self.layers = layers
        weights, biases = init_weights_and_biases(layers)
        self.weights = weights
        self.biases = biases

        self.activation_function = activation_function
        backpropagation_fn = backpropagation_fn_factory(
            activation_function=activation_function
        )
        self.sgd = stochastic_gradient_descent_fn(
            update_fn=update_fn_factory(backpropagation_fn=backpropagation_fn)
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=[],
        y_val=[],
        epochs=30,
        learning_rate=2.0,
        batch_size=10,
    ):
        """Fit funktiolla opetaan neuroverkko.

        Args:
            X_train (np.array): Opetusdatan kuvavektorit.
            y_train (np.array): Opetusdatan oikeat vastaukset.
            X_val (np.array, optional): Validaatiodatan kuvavektorit. Defaults to [].
            y_val (np.array, optional): Validaatiodatan oikeat vastaukset. Defaults to [].
            epochs (int, optional): Opetuskierrosten määrä. Defaults to 30.
            learning_rate (float, optional): Vaikuttaa gradienttimenetelmän nopeuteen. Defaults to 2.0.
            batch_size (int, optional): SGD käytetty osajoukon koko. Defaults to 10.
        """
        sgd = self.sgd(
            self.weights,
            self.biases,
            X_train,
            y_train,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        evaluations = []
        # stokastine gradientti menetelmä palauttaa generaattorin, joilloin päästään
        # väliin tekemään evaluointi
        for w, b, epoch in sgd:
            self.weights = w
            self.biases = b

            # Evaluoidaan mallin tarkkuus jokaisen epoch jälkeen
            correct_predictions_count = np.sum(
                [self.predict(x)[0] == y for x, y in zip(X_val, y_val)]
            )
            
            print("Epoch {} done -> Predicted {}/{}".format(epoch, correct_predictions_count, len(X_val)))

            evaluations.append(correct_predictions_count / len(X_val))

        return evaluations

    def predict(self, x):
        """
        Arviodaan kuvavektorin perusteella luku.

        Args:
            x (np.array): Kuvavektori

        Returns:
            (
                int: Paras arvio numerosta,
                float: Arvion todennäköisyys
                np.array: Todennäköisyydet kaikille arvoille.
            )
        """
        y = self._feedforward(x)
        prediction = np.argmax(y)
        confidence = np.round(y, 3)[prediction]

        return prediction, confidence[0], y

    def _feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.activation(calculate_z(a, w, b))
        return a
