"""
Restricted Boltzmann Machine (RBM) - Music Listener Preference Demo
===================================================================
A simple RBM implementation using Python and NumPy.
Demonstrates how listener music preferences (visible layer)
are transformed into hidden preference features (hidden layer).
"""

import numpy as np

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
NUM_VISIBLE = 6   # Number of visible units (music genres)
NUM_HIDDEN = 2    # Number of hidden units (latent features)

GENRE_NAMES = ["Pop", "Rock", "Classical", "Hip-Hop", "Jazz", "Electronic"]


# -----------------------------------------------------------
# Sigmoid activation function
# -----------------------------------------------------------
def sigmoid(x):
    """
    Sigmoid function: converts any value to a probability between 0 and 1.
    Formula: sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------
# Initialize RBM parameters
# -----------------------------------------------------------
def initialize_rbm(num_visible, num_hidden, seed=42):
    """
    Create and return the RBM parameters:
    - W: weight matrix of shape (num_visible, num_hidden)
    - visible_bias: bias for each visible unit
    - hidden_bias: bias for each hidden unit

    Weights are initialized with small random values.
    Biases are initialized to zero.
    """
    np.random.seed(seed)

    # Small random weights help the RBM start learning
    W = np.random.randn(num_visible, num_hidden) * 0.1

    # Biases start at zero
    visible_bias = np.zeros(num_visible)
    hidden_bias = np.zeros(num_hidden)

    return W, visible_bias, hidden_bias


# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
def load_data(filepath):
    """
    Load listener preference data from a CSV file.
    Returns a NumPy array of shape (num_listeners, 6) with binary values.
    """
    # Read CSV, skip header row, ignore first column (listener name)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=int)

    # Remove the first column (listener labels are not numeric, so they become -1)
    # Actually, listener labels like "L1" won't parse as int, so genfromtxt
    # will set them to -1. We just take columns 1 onwards.
    data = data[:, 1:]

    return data


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("Restricted Boltzmann Machine - Music Preference Demo")
    print("=" * 55)

    # Load the listener preference dataset
    data = load_data("data/listener_preferences.csv")
    print(f"\nDataset loaded: {data.shape[0]} listeners, {data.shape[1]} genres")
    print(f"Genres: {GENRE_NAMES}")
    print(f"\nData:\n{data}")

    # Initialize RBM
    W, visible_bias, hidden_bias = initialize_rbm(NUM_VISIBLE, NUM_HIDDEN)

    print(f"\n--- RBM Structure ---")
    print(f"Visible units: {NUM_VISIBLE}")
    print(f"Hidden units:  {NUM_HIDDEN}")
    print(f"Weight matrix shape: {W.shape}")
    print(f"\nInitial weights:\n{W}")
    print(f"\nVisible bias: {visible_bias}")
    print(f"Hidden bias:  {hidden_bias}")

    # Quick test of sigmoid function
    print(f"\n--- Sigmoid Test ---")
    test_values = [-2, -1, 0, 1, 2]
    for val in test_values:
        print(f"  sigmoid({val:+d}) = {sigmoid(val):.4f}")
