import numpy as np

from activation_funtions import sigmoid, sigmoid_derivative
from cost_functions import quadratic_cost_function_derivative
from utilities import zero_weight_and_bias_vectors

def calculate_z(a, w, b):
    return np.dot(w, a) + b

def backpropagation(weights, biases, x, y):
    """
    Vastavirta-algoritmi (backpropagation).

    https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista

    Tavoitteena on minimoida opetusesimerkkijoukkoa vastaava virhefunktio ja löytää minimointia vastaavat painot neuroneille.

    Osittaisderivaatat ja jokaisen neuronin vaikutus virheeseen lasketaan usein vastavirta-algoritmilla (backpropagation).

    Args:
        X (numpy array): Kuvien vektorit
        y (numpy array): Kuvien labelit
    """

    # Feedforward osa
    # Ensimmäinen aktivointi on input vektori.
    # Huomaa, että a vektorit on kaikille L tasolle
    a = x
    a_vectors = [a]
    # Kun taas z vektorit lasketaan 1...L kerroksille (syötekerros 0)
    z_vectors = []

    # Nyt koska painojen ja vakioiden alustuksessa jätettiin syötekerroksesta pois
    # nähdään, että lasketaan z-vektorit ensimmäisestä piilokerroksesta
    for w, b in zip(weights, biases):
        z = calculate_z(a, w, b)
        z_vectors.append(z)
        a = sigmoid(z)
        a_vectors.append(a)
    # Nyt on laskettu aktivoinnit tasoille 0...L ja z-vektorit tasoille 1...L

    # Backpropagation
    # Alustetaan nolla matriisit gradianteille. Toinen vaihtoehto olisi
    # täyttää tyhjää listaa ja lopuksi kääntää listan järjetys.
    # Koen, että on selkeämpää tehdä näin päin.
    gradients_weights, gradients_bias = zero_weight_and_bias_vectors(weights, biases)

    # Lasketaan virhe viimeiselle kerrokselle δ^L = ∇aC ⊙ σ′(z^L)
    error = quadratic_cost_function_derivative(a_vectors[-1], y)
    delta = error * sigmoid_derivative(z[-1])

    # https://tim.jyu.fi/view/143092#osittaisderivaatat-vakiotermien-bl_j-suhteen
    # https://tim.jyu.fi/view/143092#osittaisderivaatat-piilokerroksen-painojen-w_ijl-suhteen
    gradients_weights[-1] = np.dot(delta, a_vectors[-2].transpose())
    gradients_bias[-1] = delta

    # l kerrokselle δ^l = ((w^l+1)^T * δ^l+1) ⊙ σ′(z^L)
    # Nyt on hyvä huomata taas, että a on 0...L ja z on 1...L.
    # ja L taso on jo laskettu, joten haluamme laskea gradientit 1...L-1
    # Jos ||L|| = 4, niin z-vektorin koko on 3 (indeksit 0, 1 ja 2). Koska 2 on jo laskettu
    # tarvitsee laskea 1 ja 0. Siis
    for l in range(2, len(weights)+1):
        delta = np.dot(weights[-l + 1].transpose(), delta) * sigmoid_derivative(z_vectors[-l])
        gradients_bias[-l] = delta
        gradients_weights[-l] = np.dot(delta, a_vectors[-l - 1].transpose())

    return gradients_weights, gradients_bias