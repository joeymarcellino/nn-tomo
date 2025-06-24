# This example covers generating a training data set, initializing and training a neural network, and testing it both on 
# data with known and unknown ground truth.

import matplotlib.pyplot as plt
import numpy as np
import os

from generate_training_data import TrainingDataGenerator
from tomography import NeuralTomography
import plotting

# Initialize the training data generator. This example will generate and sample from two-qubit density matrices chosen randomly 
# using QuTiP's 'rand_dm' function. Each basis is measured 1000 times using noiseless POVMs.

CONFIG = {
    "n_qubits": 2,
    "distribution": "random_dm",
    "output_path": "./tomo_training_data",
    "n_counts": 1000,
    "povm_noise": 0
}

gen = TrainingDataGenerator(**CONFIG)

# Generate 10000 training data points, plus a test set of 1000.

gen.generate(n_samples=10000)

os.makedirs("./tomo_test_data", exist_ok=True)
gen.output_path = "./tomo_test_data"

gen.generate(n_samples=1000)

# Initialize the tomography tool. Each network is trained assuming a particular set of measurments; we'll use the ZZ, XX, and 
# YY bases (not informationally complete), but this can be any set of combinations of Z, X, and Y Pauli measurements. 

bases = ['ZZ','XX','YY']

tomo = NeuralTomography(n_qubits=2,bases=bases)

# Build and the network using the generated training data. Play around with the number of hidden layers, hidden layer size, learning rate, and batch size to find the 
# best settings for your application. 
tomo.load_training_data(train_path="./tomo_training_data", n_train=10000)
tomo.build_model(hidden_layers=[48, 48, 48], learning_rate=1e-4)
tomo.train(epochs=20, batch_size=256)
tomo.plot_training_history()

# Load and evaluate known test data. 

tomo.load_test_data(test_path="./tomo_test_data", n_test=1000)
results = tomo.evaluate()
plt.figure()
plt.hist(results['fidelities'])

# The results aren't great, since we're only using three bases and working with arbitrary mixed states.

# Save data file for "unknown density matrix"
#   Suppose we measure a |psi_minus> state = 1/sqrt(2)*(|01> - |10>) in the ZZ, XX, and YY bases. We expect anticorrelated results in all three bases, 
#   so we'll invent some corresponding data with a bit of noise. We sample each basis 100 times:

probs = {}
probs['ZZ'] = (1/100)*np.array([2,47,51,0])
probs['XX'] = (1/100)*np.array([1,48,50,1])
probs['YY'] = (1/100)*np.array([0,47,50,3])

os.makedirs("./tomo_unknown_data", exist_ok=True)
np.savez_compressed("./tomo_unknown_data/probs.npz",**probs)

# Load and evaluate this new data. 

tomo.load_test_data(test_path="./tomo_unknown_data", n_test=1)
results = tomo.evaluate()

rho = results['predicted_rhos'][0]
plotting.plot_cityscape(rho)

