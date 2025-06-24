import numpy as np
import qutip as qtp
import os
from tqdm import tqdm
import functools
import itertools

class TrainingDataGenerator:
    """
    A class to generate, process, and save training data for quantum state tomography.
    """
    def __init__(self, **kwargs):
        """
        Initializes the data generator with specific configuration.

        Optional Args:
            n_qubits (int): Number of qubits. Defaults to 2.
            distribution (str): Method for generating random density matrices. 
                                Supported: 
                                        'random_dm': calls QuTiP's 'rand_dm' method 
                                        'ginibre': draws density matrix from Ginibre distribution 
                                        'haar': applies unitary drawn from Haar distribution to |0> ket
                                        'werner': same as 'haar', but add uniformly distributed random amount of white noise
                                Defaults to 'haar'.
            output_path (str): Directory where data files will be saved. 
                               Defaults to the current working directory.
            n_counts (int): Number of simulated counts per basis for probability discretization. Defaults to 0 (no discretization). 
            povm_noise (float): Perturb ideal measurement POVM with random unitary parameterized by angles drawn from normal distribution with std  = povm_noise. 
                                Defaults to 0 (no noise).
        """
        
        # Config parameters with default values
        self.n_qubits = kwargs.get('n_qubits', 2)
        self.dim = 2**self.n_qubits
        self.distribution = kwargs.get('distribution','haar')
        self.n_counts = kwargs.get('n_counts', 0)
        self.povm_noise = kwargs.get('povm_noise', 0)
        self.rng_seed = kwargs.get('rng_seed', None)
        
        self.output_path = kwargs.get('output_path', None)
        if self.output_path is None:
            # Default to a folder in the current directory
            self.output_path = "./tomo_training_data"

        # Internal state
        self.bases = self._generate_bases()
        self.rng = np.random.default_rng(seed=self.rng_seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
            
        # Data will be stored here during generation
        self.all_rhos = []
        self.all_alphas = []
        self.all_probs = {basis: [] for basis in self.bases}

    def _generate_bases(self):
        """Generates the list of measurement bases (e.g., 'ZZ', 'ZX', ...)."""
        bases = []
        paulis = ['Z', 'X', 'Y']
        for p in itertools.product(paulis, repeat=self.n_qubits):
            bases.append("".join(p))
        return bases

    def _generate_one_rho(self):
        """Generates a single random density matrix based on the specified distribution."""
        if self.distribution == 'random_dm':
            return qtp.rand_dm(self.dim).full()
            
        elif self.distribution == 'ginibre':
            G = self.rng.normal(size=(self.dim, self.dim)) + 1j * self.rng.normal(size=(self.dim, self.dim))
            rho = G @ G.conj().T
            return rho / np.trace(rho)
            
        elif self.distribution == 'haar':
            psi_0 = np.zeros(self.dim) # initial state
            psi_0[0] = 1
            
            U = qtp.rand_unitary_haar(N=self.dim)
            
            psi_prime = U.full() @ psi_0
            psi_prime /= np.linalg.norm(psi_prime)
            rho = np.outer(psi_prime, psi_prime.conj())
            
            p = 0.9999999 # Add a small amount of mixedness to help Cholesky decomposition converge
            return p * rho + (1 - p) * (1 / self.dim) * np.eye(self.dim)
        
        elif self.distribution == 'werner':
            psi_0 = np.zeros(self.dim) # initial state
            psi_0[0] = 1
            
            U = qtp.rand_unitary_haar(N=self.dim)
            
            psi_prime = U.full() @ psi_0
            psi_prime /= np.linalg.norm(psi_prime)
            rho = np.outer(psi_prime, psi_prime.conj())
            
            p = self.rng.uniform(high=0.9999999) # Add uniformly distributed random amount of white noise
            return p * rho + (1 - p) * (1 / self.dim) * np.eye(self.dim)
            
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _calculate_alpha(self, rho):
        """Calculates the alpha vector from the Cholesky decomposition of rho."""
        try:
            ch = np.linalg.cholesky(rho)
            diag_alpha = np.diag(ch)
            lower_indices = np.tril_indices(self.dim, k=-1)
            lower_elements = ch[lower_indices]
            
            off_diag_alpha = np.empty(lower_elements.size * 2, dtype=np.float64)
            off_diag_alpha[0::2] = np.real(lower_elements)
            off_diag_alpha[1::2] = np.imag(lower_elements)
            
            return np.concatenate((diag_alpha, off_diag_alpha))
        except np.linalg.LinAlgError:
            return None

    def _save_data(self):
        """Saves all generated data to files."""
        print(f"\nSaving data to .npz binary files in '{self.output_path}'...")
        np.savez_compressed(os.path.join(self.output_path, 'probs.npz'), **self.all_probs)
        np.save(os.path.join(self.output_path, 'alpha.npy'), np.array(self.all_alphas))
        np.save(os.path.join(self.output_path, 'rho.npy'), np.array(self.all_rhos))
    
    def _discretize_probs(self, probs, n_counts):
        """
        Efficiently simulates sampling from a discrete probability distribution.

        Args:
            probs (np.ndarray): A 1D array of probabilities (must sum to 1).
            n_counts (int): The total number of samples to draw.

        Returns:
            np.ndarray: The simulated probabilities (counts divided by n_counts).
        """
        
        pvals = np.asarray(probs) / np.sum(probs)
        
        counts = self.rng.multinomial(n_counts, pvals=pvals)
        
        return counts / n_counts
    
    def _sample_dm_one_basis(self, rho, basis, n_qubits, povm_noise):
        
        # initialize variables
        probs = np.zeros((2**n_qubits))
        
        # define noise
        noise = []
        for i in range(0,6):
            if povm_noise > 0:
                theta1 = self.rng.normal(scale=povm_noise)
                theta2 = self.rng.normal(scale=povm_noise)
                theta3 = self.rng.normal(scale=povm_noise)
                noise.append(np.array([[np.exp(1j*theta1)*np.cos(theta2),-1j*np.exp(1j*theta3)*np.cos(theta2)],[-1j*np.exp(-1j*theta3)*np.cos(theta2),np.exp(-1j*theta1)*np.cos(theta2)]]))
            else:
                noise.append(np.array([[1,0],[0,1]]))
            
        # define POVMS
        POVM = {'Z0': noise[0] @ np.array([[1,0],[0,0]]) @ noise[0].T.conj(), 'Z1': noise[1] @ np.array([[0,0],[0,1]]) @ noise[1].T.conj(), 'X0': noise[2] @ np.array([[1,1],[1,1]]) @ noise[2].T.conj(),
                'X1': noise[3] @ np.array([[1,-1],[-1,1]]) @ noise[3].T.conj(), 'Y0': noise[4] @ np.array([[1,-1j],[1j,1]]) @ noise[4].T.conj(), 'Y1': noise[5] @ np.array([[1,1j],[-1j,1]]) @ noise[5].T.conj()
                }
        
        for b in POVM.keys():
            POVM[b] = POVM[b]/np.trace(POVM[b])
            
        # Generate all measurement outcomes (e.g., ('0','0'), ('0','1'), ...)
        outcomes = itertools.product('01', repeat=n_qubits)
        
        for i, outcome in enumerate(outcomes):
            # For each outcome, select the corresponding single-qubit POVMs
            # e.g., for basis 'ZX' and outcome ('0','1'), this is [POVM['Z0'], POVM['X1']]
            ops_to_kron = [POVM[f'{b}{o}'] for b, o in zip(basis, outcome)]
            
            # Create the full measurement operator with an iterative Kronecker product
            K = functools.reduce(np.kron, ops_to_kron)
            
            # Calculate and store the probability
            probs[i] = np.real(np.trace(K @ rho))
            
        return probs

    def generate(self, n_samples: int):
        """
        The main method to generate and save the training data.

        Args:
            num_samples (int): The total number of data points (rho) to generate.
        """
        print(f"Starting data generation for {n_samples} samples with config:")
        print(f"  Qubits: {self.n_qubits}, Distribution: '{self.distribution}', Path: '{self.output_path}'")
        
        for i in tqdm(range(n_samples), desc="Generating Data"):
            rho = self._generate_one_rho()
            alpha = self._calculate_alpha(rho)
            
            if alpha is None:
                print(f"Warning: Skipping sample {i+1} due to Cholesky decomposition failure.")
                continue
                
            self.all_rhos.append(rho)
            self.all_alphas.append(alpha)

            for basis in self.bases:
                probs = self._sample_dm_one_basis(rho, basis, self.n_qubits, self.povm_noise)
                if self.n_counts > 0:
                    probs = self._discretize_probs(probs, self.n_counts)
                self.all_probs[basis].append(probs)

        self._save_data()
        print("Data generation complete.")

#%%
if __name__ == "__main__":
# ------------------ Example ----------------
    print("\n--- Running generator with custom settings ---")
    CONFIG = {
        "n_qubits": 2,
        "distribution": "random_dm",
        "output_path": "./tomo_training_data",
        "n_counts": 0
    }
    
    generator = TrainingDataGenerator(**CONFIG)
    generator.generate(n_samples=1000)
    print("-" * 50)