from model import NeuralNetwork
import pickle
import gzip

# useiden eri mnist datasettien jälkeen päädyin tähän pikkelöityyn
# https://www.kaggle.com/datasets/pablotab/mnistpklgz
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

print(len(train_set[0]))
print(len(train_set[0][0]))
type(train_set)

model = NeuralNetwork([len(train_set[0][0]), 5, 10])
model.fit(train_set[0], train_set[1], 10, 30, 100)