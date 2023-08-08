"""
Stokastinen gradienttimenetelmä.

Gradienttimenetelmää käytetään funktion minimin etsintään.
Lisää gradienttimenetelmästä täällä https://tim.jyu.fi/view/143092#gradienttimenetelma.


Stokastinen gradienttimetelmässä opetusdata pilkotaan pienempiin osajoukkoihin.
https://tim.jyu.fi/view/143092#stokastinen-gradienttimenetelm%C3%A4

Funktiot toteutettu käymällä "curring"-tekniikkaa, jotta riippuvuudet voidaan
injektoida. Näin yksikkötestaaminen helpottuu.
"""
import random

from utilities import split_every, zero_weight_and_bias_vectors
from backpropagation import backpropagation_fn_factory


def update_fn_factory(backpropagation_fn=backpropagation_fn_factory()):
    """Gradienttimenetelmän painojen ja vakioiden päivitysfunktion tehdas.

    Args:
        backpropagation_fn (function, optional): Backpropagation funktio. Defaults to backpropagation_fn_factory().
    """
    def update(weights, biases, batch, learning_rate):
        """Gradienttimenetelmän painojen ja vakioiden päivitysfunktio.

        Args:
            weights (np.array): Painot
            biases (np.array): Vakiot
            batch (np.array): Opetusdatan osajoukko
            learning_rate (float): Gradienttimenetelmän askeleen pituus.

        Returns:
            (np.array): Painot
            (np.array): Vakiot
        """
        weight_gradients, bias_gradients = zero_weight_and_bias_vectors(weights, biases)

        batch_size = len(batch)

        for x, y in batch:
            delta_weights, delta_biases = backpropagation_fn(weights, biases, x, y)

            weight_gradients = [
                w_gradient + delta_w
                for w_gradient, delta_w in zip(weight_gradients, delta_weights)
            ]
            bias_gradients = [
                b_gradient + delta_b
                for b_gradient, delta_b in zip(bias_gradients, delta_biases)
            ]

        weights = [
            w - (learning_rate / batch_size) * w_gradient
            for w, w_gradient in zip(weights, weight_gradients)
        ]

        biases = [
            b - (learning_rate / batch_size) * b_gradient
            for b, b_gradient in zip(biases, bias_gradients)
        ]

        return weights, biases

    return update


def stochastic_gradient_descent_fn(
    update_fn=update_fn_factory(),
):
    """Stokastisen gradienttimenetelmän tehdas.

    Args:
        update_fn (function, optional): Päivitysfunktio. Defaults to update_fn_factory().
    """
    def stochastic_gradient_descent(
        weights,
        biases,
        X,
        y,
        epochs=30,
        learning_rate=2.0,
        batch_size=10,
    ):
        """Stokastinen gradienttimenetelmä.

        https://tim.jyu.fi/view/143092#stokastinen-gradienttimenetelm%C3%A4

        Args:
            weights (no.array): Painot.
            biases (no.array): Vakiot.
            X (no.array): Opetusdatan kuvavektorit.
            y (no.array): Opetusdatan vastaukset.
            epochs (int, optional): Opetuskierrosten määrä. Defaults to 30.
            learning_rate (float, optional): Vaikuttaa gradienttimenetelmän nopeuteen. Defaults to 2.0.
            batch_size (int, optional): SGD käytetty osajoukon koko. Defaults to 10.

        Yields:
            _type_: _description_
        """
        training_data = list(zip(X, y))

        for epoch in range(epochs):
            random.shuffle(training_data)
            for batch in split_every(batch_size, training_data):
                weights, biases = update_fn(weights, biases, batch, learning_rate)
            yield weights, biases, epoch

    return stochastic_gradient_descent
