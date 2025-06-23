import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from tensorflow import keras
import os

class NeuralTomography:
    """
    A class to handle neural network-based quantum state tomography.
    It encapsulates data loading, model building, training, and evaluation.
    """
    def __init__(self, n_qubits: int, bases: list):
        """
        Initializes the tomography tool.

        Args:
            n_qubits (int): The number of qubits in the system.
            bases (list): A list of measurement basis strings (e.g., ['ZZ', 'XX']).
        """
        if not bases:
            raise ValueError("Bases list cannot be empty.")
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.alpha_len = self.dim**2
        self.bases = bases
        
        self.model = None
        self.history = None
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.alpha_test, self.rho_test = None, None, None
        
    def _reconstruct_rho(self, alpha):
        """
        Reconstructs the density matrix rho from its Cholesky vector alpha in a 
        vectorized manner.

        Args:
            alpha (np.ndarray): A 1D array containing the real diagonal elements, 
                                followed by the interleaved real and imaginary parts of the 
                                lower-triangular off-diagonal elements.
        Returns:
            np.ndarray: The reconstructed density matrix rho.
        """
        # The number of diagonal elements is d, and off-diagonal is d*(d-1).
        # Total elements in alpha = d (real diags) + 2 * (d*(d-1)/2) (real/imag off-diags) = d^2.
        d = int(np.sqrt(len(alpha)))
        if d*d != len(alpha):
            raise ValueError("Length of alpha vector is not a perfect square.")

        ch = np.zeros((d, d), dtype=complex)
        
        # Fill the diagonal elements directly
        diag_elements = alpha[:d]
        np.fill_diagonal(ch, diag_elements)
        
        # Fill the off-diagonal elements
        off_diag_elements = alpha[d:]
        if off_diag_elements.size > 0:
            # Get the indices of the lower triangle (excluding the diagonal)
            lower_indices = np.tril_indices(d, k=-1)
            
            # Create complex numbers by de-interleaving real and imaginary parts
            complex_vals = off_diag_elements[0::2] + 1j * off_diag_elements[1::2]
            
            # Assign the complex values to the lower triangle
            ch[lower_indices] = complex_vals
            
        # Reconstruct rho and normalize
        rho = ch @ ch.conj().T
        trace_rho = np.trace(rho)
        
        # Avoid division by zero if trace is zero
        return rho / trace_rho if trace_rho != 0 else rho

    def load_training_data(self, train_path: str, n_train: int, n_val_split: float = 0.1, rng_seed: int = None):
        """
        Loads training data from .npz and .npy files.

        Args:
            train_path (str): Path to the training data directory.
            n_train (int): The number of training samples to load.
            n_val_split (float): The fraction of training data to use for validation. Defaults to .1.
            rng_seed (int): Seed for training/validation data shuffling. Defaults to None (random seed).
        """
        print("Loading and preprocessing data...")
        
        # Load the target 'alpha' vectors
        self.y_train = np.load(os.path.join(train_path, 'alpha.npy'))[:n_train]
        
        # Load the input probability distributions
        # We expect a single 'probs.npz' file containing arrays for each basis
        with np.load(os.path.join(train_path, 'probs.npz')) as data:
            # Concatenate the probability arrays for each basis horizontally
            prob_arrays = [data[b][:n_train] for b in self.bases]
            self.x_train = np.concatenate(prob_arrays, axis=1)
            
        # Randomly permute training data
        if rng_seed is not None:
            rng = np.random.default_rng(seed=rng_seed)
        else:
            rng = np.random.default_rng()
            
        permutation = rng.permutation(len(self.x_train))
        self.x_train = self.x_train[permutation]
        self.y_train = self.y_train[permutation]

        # Create validation split
        n_val = int(n_val_split * n_train)
        self.x_val = self.x_train[:n_val]
        self.x_train = self.x_train[n_val:]
        self.y_val = self.y_train[:n_val]
        self.y_train = self.y_train[n_val:]
        
        print(f"Data loaded. Training samples: {len(self.x_train)}, Validation samples: {len(self.x_val)}")
        
    def load_test_data(self, test_path: str, n_test: int):
        """
        Loads test data from .npz and .npy files.
        
        Args:
            train_path (str): Path to the test data directory.
            n_train (int): The number of test samples to load.
        """
        
        print("Loading and preprocessing data...")

        if os.path.exists(os.path.join(test_path, 'rho.npy')):
            self.rho_test = np.load(os.path.join(test_path, 'rho.npy'))          
        
        with np.load(os.path.join(test_path, 'probs.npz')) as data:
            prob_arrays = [data[b][:n_test] for b in self.bases]
            self.x_test = np.concatenate(prob_arrays, axis=1)
        
        print(f"Data loaded. Test samples: {len(self.x_test)}")

    def build_model(self, hidden_layers: list = [48, 48, 48], activation='relu', learning_rate=1e-4):
        """
        Builds and compiles the Keras model.
        
        Args:
            hidden_layers (list): List of sizes of hidden layers of network. Defaults to [48,48,48], resulting in a network with
                an input layer of size (number of bases)*(2**n_qubits), three hidden layers of size 48, and an output layer of size 2**(2*n_qubits).
            activation: Activation function for neurons. Defaults to 'relu', see Keras documentation for other options.
            learning_rate: Sets how strongly the network updates on each training point. Defaults to 1e-4.
        """
        
        inputs = keras.Input(shape=(self.x_train.shape[1],), name="probabilities")
        x = inputs
        for units in hidden_layers:
            x = keras.layers.Dense(units, activation=activation)(x)
        outputs = keras.layers.Dense(self.alpha_len, name="alpha_predicted")(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate), loss='MSE')
        print("Model built and compiled.")
        self.model.summary()
        
    def train(self, epochs: int, batch_size: int = 256):
        """Trains the model.
        
        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Number of training points used to compute each gradient. Defaults to 256.
        """
        if self.model is None or self.x_train is None:
            raise RuntimeError("Model is not built or data is not loaded. Run build_model() and load_data() first.")
            
        print("Starting model training...")
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val)
        )
        return self.history

    def evaluate(self):
        """Evaluates the trained model on the test set.
        
            Returns:
                fidelities (list): Fidelities of test density matrices to reconstructions.
                l2_distances (list): L2 distances between test density matrices and reconstructions.
                predicted_rhos (list): Reconstructed density matrices.
        """
        if self.model is None or self.x_test is None:
            raise RuntimeError("Model is not built or test data is not loaded.")
        
        print("Evaluating model on the test set...")
        alpha_pred = self.model.predict(self.x_test)
        
        n_test = len(self.x_test)
        rho_nn = np.zeros((n_test, self.dim, self.dim), dtype=complex)
        fidelities = np.zeros(n_test)
        l2_distances = np.zeros(n_test)

        for i in range(n_test):
            rho_nn[i] = self._reconstruct_rho(alpha_pred[i])
            
        if self.rho_test is not None:
            for i in range(n_test):
                # Fidelity calculation F(rho, sigma) = (Tr[sqrt(sqrt(rho) @ sigma @ sqrt(rho))])^2
                sqrt_rho_test = scipy.linalg.sqrtm(self.rho_test[i])
                fidelity_matrix = sqrt_rho_test @ rho_nn[i] @ sqrt_rho_test
                fidelities[i] = np.abs(np.trace(scipy.linalg.sqrtm(fidelity_matrix)))**2
        
                # L2 distance calculation (optimized)
                diff_matrix = self.rho_test[i] - rho_nn[i]
                l2_distances[i] = np.linalg.norm(diff_matrix)**2
                
            print(f"Mean Fidelity: {np.mean(fidelities):.4f} ± {np.std(fidelities):.4f}")
            print(f"Mean L2 Distance: {np.mean(l2_distances):.4g}")
        
            return {"fidelities": fidelities, "l2_distances": l2_distances, "predicted_rhos": rho_nn}
        else:
            return {"predicted_rhos": rho_nn}

    def plot_training_history(self):
        """Plots the training and validation loss."""
        if not self.history:
            print("No training history to plot.")
            return

        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b-", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.show()

#%%
if __name__ == '__main__':
    # Configuration
    N_QUBITS = 2
    BASES = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
    
    # Paths should point to directories containing .npy and .npz files
    TRAIN_PATH = 'Q:\\Projects\\Networks\\2023_NN_Tomography\\custom_tomo_data\\train'
    TEST_PATH = 'Q:\\Projects\\Networks\\2023_NN_Tomography\\custom_tomo_data\\test'
    
    # Initialize the tomography tool
    tomo_nn = NeuralTomography(n_qubits=N_QUBITS, bases=BASES)
    
    # Load the data
    try:
        tomo_nn.load_training_data(train_path=TRAIN_PATH, n_train=10000)
        tomo_nn.load_test_data(test_path=TEST_PATH, n_test=1000)
    except FileNotFoundError:
        print("\nERROR: Data files not found.")
        print("Please ensure the data has been generated and saved in .npy/.npz format")
        print(f"Expected training data in: {os.path.abspath(TRAIN_PATH)}")
        print(f"Expected testing data in: {os.path.abspath(TEST_PATH)}")
        exit()

    # Build the model
    tomo_nn.build_model(hidden_layers=[128, 256, 128], learning_rate=1e-4)
    
    # Train the model
    tomo_nn.train(epochs=20, batch_size=256)
    
    # Evaluate the model on the test set
    results = tomo_nn.evaluate()
    
    # Plot the training history
    tomo_nn.plot_training_history()