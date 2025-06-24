# Quantum state tomography

Quantum state tomography involves reconstructing an unknown density matrix from the results of some set of measurements. Given an informationally complete set and perfect statistics the problem is trivial, but absent either of these things it becomes a task of inferring some low-dimensional latent variables from high-dimensional data. These are the sort of tasks for which neural networks are well-suited, and indeed even basic architectures have been shown to outperform standard tomographic methods (maximum likelihood, linear regression, and Bayesian mean estimation) [1]. This code allows the user to initialize, train, and use a simple fully-connected network for qubit state tomography given the results of some set of Pauli measurements, as well as generate the associated training data.

[1] D. Koutny, L. Motka, Z. Hradil, J. Řeháček, and L. L. Sánchez-Soto. Neural-network quantum state tomography. Physical Review A, 106(1):012409, 2022.

# Getting started

## Prerequisites 
- numpy
- scipy
- os
- matplotlib
- functools
- itertools
- tqdm
- tensorflow
- qutip

## Usage

See example.py

# Contact

For questions or bug reports contact francis [dot] marcellinoiii [at] unige [dot] ch
