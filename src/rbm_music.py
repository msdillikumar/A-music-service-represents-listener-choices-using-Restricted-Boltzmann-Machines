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
# Hidden layer probability: P(h=1 | v)
# -----------------------------------------------------------
def hidden_probability(v, W, hidden_bias):
    """
    Given a visible vector v, calculate the probability that
    each hidden unit is activated.

    Formula: P(h=1|v) = sigmoid(v . W + hidden_bias)

    Parameters:
        v           - visible layer vector (1D array of 0s and 1s)
        W           - weight matrix (num_visible x num_hidden)
        hidden_bias - bias for each hidden unit

    Returns:
        Array of probabilities, one per hidden unit.
    """
    # Step 1: Multiply visible vector by weight matrix
    activation = np.dot(v, W)

    # Step 2: Add hidden bias
    activation = activation + hidden_bias

    # Step 3: Apply sigmoid to get probabilities
    prob = sigmoid(activation)

    return prob


# -----------------------------------------------------------
# Visible layer probability: P(v=1 | h)
# -----------------------------------------------------------
def visible_probability(h, W, visible_bias):
    """
    Given a hidden vector h, calculate the probability that
    each visible unit is activated (reconstruction).

    Formula: P(v=1|h) = sigmoid(h . W^T + visible_bias)

    Parameters:
        h            - hidden layer vector
        W            - weight matrix (num_visible x num_hidden)
        visible_bias - bias for each visible unit

    Returns:
        Array of probabilities, one per visible unit.
    """
    # Multiply hidden vector by transposed weight matrix, add visible bias
    activation = np.dot(h, W.T) + visible_bias
    prob = sigmoid(activation)

    return prob


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

    # Initialize RBM
    W, visible_bias, hidden_bias = initialize_rbm(NUM_VISIBLE, NUM_HIDDEN)

    print(f"\n--- RBM Structure ---")
    print(f"Visible units: {NUM_VISIBLE}")
    print(f"Hidden units:  {NUM_HIDDEN}")
    print(f"Weight matrix shape: {W.shape}")

    # -------------------------------------------------------
    # Demonstrate hidden representation for one listener
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("Hidden Representation Calculation (before training)")
    print("=" * 55)

    # Example listener: likes Pop, Rock, Hip-Hop, Electronic
    listener = np.array([1, 1, 0, 1, 0, 1])

    print(f"\nListener input vector:")
    for i, genre in enumerate(GENRE_NAMES):
        status = "likes" if listener[i] == 1 else "does not like"
        print(f"  {genre}: {listener[i]} ({status})")

    # Step-by-step calculation
    print(f"\n--- Step-by-step calculation ---")

    # Step 1: v . W
    vW = np.dot(listener, W)
    print(f"Step 1: v . W = {vW}")

    # Step 2: Add hidden bias
    activation = vW + hidden_bias
    print(f"Step 2: v . W + hidden_bias = {activation}")

    # Step 3: Apply sigmoid
    h_prob = sigmoid(activation)
    print(f"Step 3: sigmoid(activation) = {h_prob}")

    # Verify using our function
    h_prob_check = hidden_probability(listener, W, hidden_bias)
    print(f"\nUsing hidden_probability() function: {h_prob_check}")

    print(f"\nHidden activation probabilities:")
    print(f"  H1 (Latent Feature 1): {h_prob[0]:.4f}")
    print(f"  H2 (Latent Feature 2): {h_prob[1]:.4f}")

    # Convert to binary using threshold 0.5
    h_binary = (h_prob >= 0.5).astype(int)
    print(f"\nHidden representation (threshold 0.5): {h_binary}")

    print(f"\nNote: These are results with RANDOM initial weights.")
    print(f"After training, the hidden units will capture meaningful patterns.")
