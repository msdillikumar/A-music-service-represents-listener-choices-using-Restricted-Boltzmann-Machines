"""
Restricted Boltzmann Machine (RBM) - Music Listener Preference Demo
===================================================================
A simple RBM implementation using Python and NumPy.
Demonstrates how listener music preferences (visible layer)
are transformed into hidden preference features (hidden layer).
"""

import numpy as np
import os

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
# Train the RBM using Contrastive Divergence (CD-1)
# -----------------------------------------------------------
def train_rbm(data, W, visible_bias, hidden_bias, epochs=500, learning_rate=0.1):
    """
    Train the RBM using Contrastive Divergence with 1 step (CD-1).

    CD-1 is the simplest and most common RBM training algorithm.
    For each training sample, it does:
      1. Positive phase: compute hidden probabilities from visible input
      2. Reconstruction: generate visible probabilities from hidden sample
      3. Negative phase: compute hidden probabilities from reconstruction
      4. Update: adjust weights and biases based on the difference

    Parameters:
        data          - training data, shape (num_samples, num_visible)
        W             - weight matrix
        visible_bias  - visible unit biases
        hidden_bias   - hidden unit biases
        epochs        - number of training iterations over the full dataset
        learning_rate - how much to adjust weights each step

    Returns:
        Updated W, visible_bias, hidden_bias
    """
    num_samples = data.shape[0]

    for epoch in range(epochs):
        total_error = 0.0

        for i in range(num_samples):
            # Current training sample (one listener's preferences)
            v0 = data[i].astype(float)

            # --- POSITIVE PHASE ---
            # Compute hidden probabilities from the real visible data
            h0_prob = hidden_probability(v0, W, hidden_bias)

            # Sample hidden units: convert probabilities to binary (0 or 1)
            # using random threshold (stochastic sampling)
            h0_sample = (h0_prob > np.random.rand(len(h0_prob))).astype(float)

            # --- RECONSTRUCTION (Negative phase) ---
            # Generate visible reconstruction from the hidden sample
            v1_prob = visible_probability(h0_sample, W, visible_bias)

            # Compute hidden probabilities from the reconstruction
            h1_prob = hidden_probability(v1_prob, W, hidden_bias)

            # --- WEIGHT UPDATE ---
            # Positive associations: real visible * real hidden
            positive = np.outer(v0, h0_prob)

            # Negative associations: reconstructed visible * reconstructed hidden
            negative = np.outer(v1_prob, h1_prob)

            # Update weights: move toward positive, away from negative
            W += learning_rate * (positive - negative) / num_samples

            # Update biases
            visible_bias += learning_rate * (v0 - v1_prob) / num_samples
            hidden_bias += learning_rate * (h0_prob - h1_prob) / num_samples

            # Track reconstruction error (how well RBM reconstructs the input)
            total_error += np.sum((v0 - v1_prob) ** 2)

        # Print progress every 100 epochs
        avg_error = total_error / num_samples
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:4d}/{epochs} - Reconstruction error: {avg_error:.4f}")

    return W, visible_bias, hidden_bias


# -----------------------------------------------------------
# Save results to a text file
# -----------------------------------------------------------
def save_results(filepath, listener, h_prob, h_binary, data, W,
                 visible_bias, hidden_bias, genre_names):
    """
    Save the RBM results to a text file for documentation.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("Restricted Boltzmann Machine - Music Preference Results\n")
        f.write("=" * 55 + "\n\n")

        # Listener demo
        f.write("DEMO LISTENER\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input vector: {listener}\n")
        for i, genre in enumerate(genre_names):
            status = "likes" if listener[i] == 1 else "does not like"
            f.write(f"  {genre}: {listener[i]} ({status})\n")
        f.write(f"\nHidden activation probabilities:\n")
        f.write(f"  H1: {h_prob[0]:.4f}\n")
        f.write(f"  H2: {h_prob[1]:.4f}\n")
        f.write(f"Hidden representation: {h_binary}\n\n")

        # All listeners
        f.write("ALL LISTENERS - Hidden Representations\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Listener':<10} {'Input Vector':<25} {'H1':>6} {'H2':>6} {'Hidden':>8}\n")
        for i in range(data.shape[0]):
            v = data[i].astype(float)
            h_p = hidden_probability(v, W, hidden_bias)
            h_b = (h_p >= 0.5).astype(int)
            f.write(f"L{i+1:<9} {str(data[i]):<25} {h_p[0]:>6.4f} {h_p[1]:>6.4f} {str(h_b):>8}\n")

        # Learned weights
        f.write(f"\nLEARNED WEIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Weight matrix (6 visible x 2 hidden):\n")
        for i, genre in enumerate(genre_names):
            f.write(f"  {genre:<12} -> H1: {W[i,0]:+.4f}, H2: {W[i,1]:+.4f}\n")

    print(f"\nResults saved to: {filepath}")


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

    # Example listener: likes Pop, Rock, Hip-Hop, Electronic
    listener = np.array([1, 1, 0, 1, 0, 1])

    # -------------------------------------------------------
    # Show hidden representation BEFORE training
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("BEFORE TRAINING (random weights)")
    print("=" * 55)

    h_prob_before = hidden_probability(listener, W, hidden_bias)
    h_binary_before = (h_prob_before >= 0.5).astype(int)

    print(f"\nListener: {listener}")
    print(f"Hidden probabilities: H1={h_prob_before[0]:.4f}, H2={h_prob_before[1]:.4f}")
    print(f"Hidden representation: {h_binary_before}")
    print(f"(These are near 0.5 because weights are random — no learning yet)")

    # -------------------------------------------------------
    # Train the RBM
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("TRAINING THE RBM")
    print("=" * 55)
    print(f"\nTraining with CD-1 for 500 epochs, learning rate = 0.1")
    print()

    W, visible_bias, hidden_bias = train_rbm(
        data, W, visible_bias, hidden_bias,
        epochs=500, learning_rate=0.1
    )

    print(f"\nTraining complete!")
    print(f"\nLearned weights:\n{W}")
    print(f"Learned visible bias: {visible_bias}")
    print(f"Learned hidden bias:  {hidden_bias}")

    # -------------------------------------------------------
    # Show hidden representation AFTER training
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("AFTER TRAINING (learned weights)")
    print("=" * 55)

    print(f"\nListener input vector:")
    for i, genre in enumerate(GENRE_NAMES):
        status = "likes" if listener[i] == 1 else "does not like"
        print(f"  {genre}: {listener[i]} ({status})")

    # Step-by-step calculation with trained weights
    print(f"\n--- Step-by-step calculation ---")

    vW = np.dot(listener, W)
    print(f"Step 1: v . W = [{vW[0]:.4f}, {vW[1]:.4f}]")

    activation = vW + hidden_bias
    print(f"Step 2: v . W + hidden_bias = [{activation[0]:.4f}, {activation[1]:.4f}]")

    h_prob = sigmoid(activation)
    print(f"Step 3: sigmoid(activation) = [{h_prob[0]:.4f}, {h_prob[1]:.4f}]")

    print(f"\nHidden activation probabilities:")
    print(f"  H1 (Latent Feature 1): {h_prob[0]:.4f}")
    print(f"  H2 (Latent Feature 2): {h_prob[1]:.4f}")

    h_binary = (h_prob >= 0.5).astype(int)
    print(f"\nHidden representation (threshold 0.5): {h_binary}")

    # -------------------------------------------------------
    # Reconstruction demo
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("RECONSTRUCTION DEMO")
    print("=" * 55)
    print(f"\nUsing hidden representation {h_binary} to reconstruct visible layer:")

    v_reconstructed_prob = visible_probability(h_binary.astype(float), W, visible_bias)
    v_reconstructed = (v_reconstructed_prob >= 0.5).astype(int)

    print(f"\n{'Genre':<12} {'Original':>10} {'Reconstructed':>15} {'Probability':>12}")
    print("-" * 52)
    for i, genre in enumerate(GENRE_NAMES):
        print(f"{genre:<12} {listener[i]:>10} {v_reconstructed[i]:>15} {v_reconstructed_prob[i]:>12.4f}")

    match_count = np.sum(listener == v_reconstructed)
    print(f"\nReconstruction accuracy: {match_count}/{NUM_VISIBLE} genres match")

    # -------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("INTERPRETATION")
    print("=" * 55)
    print(f"\nH1 represents a learned latent preference pattern.")
    print(f"H2 represents another learned latent preference pattern.")
    print(f"\nNote: RBMs learn latent features from data automatically.")
    print(f"Hidden units do not receive human-readable labels.")
    print(f"The above names are conceptual interpretations for demonstration.")

    # Show all listeners' hidden representations
    print(f"\n{'=' * 55}")
    print("ALL LISTENERS - Hidden Representations")
    print("=" * 55)
    print(f"\n{'Listener':<10} {'Input Vector':<25} {'H1 prob':>8} {'H2 prob':>8} {'Hidden':>8}")
    print("-" * 62)

    for i in range(data.shape[0]):
        v = data[i].astype(float)
        h_p = hidden_probability(v, W, hidden_bias)
        h_b = (h_p >= 0.5).astype(int)
        print(f"L{i+1:<9} {str(data[i]):<25} {h_p[0]:>8.4f} {h_p[1]:>8.4f} {str(h_b):>8}")

    # -------------------------------------------------------
    # Save results to file
    # -------------------------------------------------------
    save_results(
        "results/hidden_representation.txt",
        listener, h_prob, h_binary, data, W,
        visible_bias, hidden_bias, GENRE_NAMES
    )
