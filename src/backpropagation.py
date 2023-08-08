"""
Vastavirta-algoritmi on neuroverkon oppimisen kovaa ydintä.

Lisää funktioiden kommenteissa.

Funktiot toteutettu käymällä "curring"-tekniikkaa, jotta riippuvuudet voidaan
injektoida. Näin yksikkötestaaminen helpottuu.
"""
import numpy as np

from activation_funtions import sigmoid, sigmoid_derivative
from cost_functions import quadratic_cost_function_derivative
from utilities import calculate_z, zero_weight_and_bias_vectors


def backpropagation_fn_factory(
    activation_function=sigmoid,
    activation_function_derivative=sigmoid_derivative,
    cost_function_derivative=quadratic_cost_function_derivative,
):
    """
    Vastavirta-algoritmi (backpropagation).

    https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista

    Tavoitteena on minimoida opetusesimerkkijoukkoa vastaava virhefunktio ja löytää
    minimointia vastaavat painot neuroneille.

    Osittaisderivaatat ja jokaisen neuronin vaikutus virheeseen lasketaan usein
    vastavirta-algoritmilla (backpropagation).

    Args:
        X (numpy array): Kuvien vektorit
        y (numpy array): Kuvien labelit
    """

    def backpropagation(weights, biases, x, y):
        # Feedforward osa
        # Ensimmäinen aktivointi on input vektori.
        # Huomaa, että a vektorit on kaikille L tasolle
        a_vectors = [x]
        # Kun taas z vektorit lasketaan 1...L kerroksille (syötekerros 0)
        z_vectors = []

        # Nyt koska painojen ja vakioiden alustuksessa jätettiin syötekerroksesta pois
        # nähdään, että lasketaan z-vektorit ensimmäisestä piilokerroksesta
        for w, b in zip(weights, biases):
            z = calculate_z(a_vectors[-1], w, b)
            z_vectors.append(z)
            a_vectors.append(activation_function(z))
        # Nyt on laskettu aktivoinnit tasoille 0...L ja z-vektorit tasoille 1...L

        # Backpropagation
        # Alustetaan nolla matriisit gradianteille. Toinen vaihtoehto olisi
        # täyttää tyhjää listaa ja lopuksi kääntää listan järjetys.
        # Koen, että on selkeämpää tehdä näin päin.
        gradients_weights, gradients_bias = zero_weight_and_bias_vectors(
            weights, biases
        )

        # Lasketaan virhe viimeiselle kerrokselle δ^L = ∇aC ⊙ σ′(z^L)
                
        error = cost_function_derivative(a_vectors[-1], y)
        delta = error * activation_function_derivative(z_vectors[-1])

        # https://tim.jyu.fi/view/143092#osittaisderivaatat-vakiotermien-bl_j-suhteen
        # https://tim.jyu.fi/view/143092#osittaisderivaatat-piilokerroksen-painojen-w_ijl-suhteen
        gradients_weights[-1] = np.dot(delta, a_vectors[-2].transpose())
        gradients_bias[-1] = delta

        # l kerrokselle δ^l = ((w^l+1)^T * δ^l+1) ⊙ σ′(z^l)
        # Nyt on hyvä huomata taas, että a on 0...L ja z on 1...L.
        # ja L taso on jo laskettu, joten haluamme laskea gradientit 1...L-1
        # Jos ||L|| = 4, niin z-vektorin koko on 3 (indeksit 0, 1 ja 2). Koska 2 on jo laskettu
        # tarvitsee laskea 1 ja 0. Siis
        # Tässä käytetään hyväksi Python kielen negatiivisia taulukko indeksejä.
        # Koska gradientit, painot, vakiot ja aktivoinnit ovat eri kokoisia,
        # ne ovat kuitenkin takaperin katsottuna samassa järjestyksessä.
        for layer in range(2, len(weights) + 1):
            z = z_vectors[-layer]
            activation_derivative_value = activation_function_derivative(z)
            delta = np.dot(
                weights[-layer + 1].transpose(), delta
            ) * activation_derivative_value
            gradients_bias[-layer] = delta
            gradients_weights[-layer] = np.dot(delta, a_vectors[-layer - 1].transpose())

        return gradients_weights, gradients_bias

    return backpropagation
