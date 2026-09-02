# Project Report: Restricted Boltzmann Machine — Music Listener Preference Demo

## 1. Title Page

- **Project Title:** Restricted Boltzmann Machine — Music Listener Preference Demo
- **Subject:** Deep Learning Essentials — Innovative Assignment-I
- **Topic:** Restricted Boltzmann Machines
- **Semester:** V (5th Semester)

---

## 2. Problem Statement

A music service represents listener choices using visible variables and hidden preference features. Construct a simple RBM with the specified visible and hidden units, apply one listener vector conceptually to the visible layer, and demonstrate the resulting hidden representation.

---

## 3. Objective

The objective of this project is to:

1. Build a simple Restricted Boltzmann Machine (RBM) from scratch using Python and NumPy.
2. Demonstrate how an RBM can learn latent preference patterns from binary listener data.
3. Show the step-by-step calculation of hidden activation probabilities.
4. Train the RBM using Contrastive Divergence (CD-1) and show how learned weights improve the hidden representation.

---

## 4. Concept Used

### What is a Restricted Boltzmann Machine?

A Restricted Boltzmann Machine (RBM) is a type of **generative stochastic neural network** that can learn a probability distribution over its inputs. It consists of two layers:

- **Visible Layer:** Represents the observed data.
- **Hidden Layer:** Represents latent (hidden) features that the model learns from the data.

### Key Properties

- **Restricted** means there are no connections between units in the same layer.
- Visible units are only connected to hidden units through **weighted connections**.
- Each unit has a **bias** value that shifts its activation threshold.

### Visible Layer (in this project)

We use 6 visible units, each representing whether a listener likes a particular music genre:

| Unit | Genre |
|------|-------|
| V1 | Pop |
| V2 | Rock |
| V3 | Classical |
| V4 | Hip-Hop |
| V5 | Jazz |
| V6 | Electronic |

Values are binary: **1 = likes**, **0 = does not like**.

### Hidden Layer (in this project)

We use 2 hidden units that represent **latent preference features**:

| Unit | Conceptual Interpretation |
|------|--------------------------|
| H1 | Latent Preference Pattern 1 |
| H2 | Latent Preference Pattern 2 |

> **Important:** These names are conceptual. RBMs learn latent features automatically — the hidden units do not receive human-readable labels.

### Weights

The weight matrix `W` has shape (6, 2) — one weight for each connection between a visible unit and a hidden unit. Weights determine how strongly a visible unit influences a hidden unit.

- **Positive weight:** The visible unit activates the hidden unit.
- **Negative weight:** The visible unit suppresses the hidden unit.

### Bias

Each unit has a bias value:
- **Visible bias:** Affects how easily each visible unit is activated during reconstruction.
- **Hidden bias:** Affects the baseline activation level of each hidden unit.

### Sigmoid Function

The sigmoid function converts any real number into a probability between 0 and 1:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

It is used to calculate the probability that a hidden unit is "on" (activated) given the visible input.

### Hidden Activation Probability

The probability that a hidden unit `h_j` is activated given a visible vector `v` is:

```
P(h_j = 1 | v) = sigmoid( sum(v_i * W_ij) + hidden_bias_j )
```

In matrix form:

```
P(h = 1 | v) = sigmoid(v · W + hidden_bias)
```

### Hidden Representation

After computing the activation probabilities, we convert them to binary values using a threshold of 0.5:

- If P(h_j = 1 | v) ≥ 0.5, then h_j = 1 (activated)
- If P(h_j = 1 | v) < 0.5, then h_j = 0 (not activated)

---

## 5. Methodology / Working Steps

### Step 1: Create Dataset
- Created a small CSV file with 10 listeners and their binary preferences across 6 music genres.

### Step 2: Initialize RBM
- Weight matrix `W` (6×2) initialized with small random values using seed 42.
- Visible bias (6 values) and hidden bias (2 values) initialized to zero.

### Step 3: Implement Core Functions
- `sigmoid(x)` — Activation function.
- `hidden_probability(v, W, hidden_bias)` — Computes P(h=1|v).
- `visible_probability(h, W, visible_bias)` — Computes P(v=1|h) for reconstruction.

### Step 4: Train Using Contrastive Divergence (CD-1)
For each listener in each epoch:
1. **Positive phase:** Compute hidden probabilities from the real visible data.
2. **Sample hidden units:** Convert probabilities to binary (0/1) using stochastic sampling.
3. **Reconstruction:** Generate visible layer probabilities from the hidden sample.
4. **Negative phase:** Compute hidden probabilities from the reconstructed visible layer.
5. **Update weights and biases:**
   - `W += learning_rate × (v₀ × h₀ᵀ - v₁ × h₁ᵀ) / num_samples`
   - `visible_bias += learning_rate × (v₀ - v₁) / num_samples`
   - `hidden_bias += learning_rate × (h₀ - h₁) / num_samples`

### Step 5: Demonstrate Results
- Applied a demo listener vector `[1, 1, 0, 1, 0, 1]` to the trained RBM.
- Showed step-by-step calculation of hidden activation probabilities.
- Displayed hidden representations for all 10 listeners.

### Step 6: Visualize
- Created an RBM architecture diagram showing visible and hidden units with weight-colored connections.
- Created a bar chart of learned weights per genre.

---

## 6. Implementation

### Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| NumPy | Matrix operations, sigmoid calculation |
| Matplotlib | Visualization (architecture diagram, bar chart) |

### Source Code

- **GitHub Repository:** https://github.com/msdillikumar/A-music-service-represents-listener-choices-using-Restricted-Boltzmann-Machines
- **Main file:** `src/rbm_music.py`
- **Dataset:** `data/listener_preferences.csv`

### Key Functions

| Function | Purpose |
|----------|---------|
| `sigmoid(x)` | Activation function: `1 / (1 + exp(-x))` |
| `initialize_rbm()` | Creates weight matrix and biases |
| `load_data()` | Reads CSV dataset into NumPy array |
| `hidden_probability()` | Computes P(h=1\|v) = sigmoid(v·W + hidden_bias) |
| `visible_probability()` | Computes P(v=1\|h) = sigmoid(h·Wᵀ + visible_bias) |
| `train_rbm()` | Trains using CD-1 for specified epochs |
| `save_results()` | Saves results to text file |
| `plot_rbm_results()` | Generates visualization |

---

## 7. Results & Output

### Training Progress

| Epoch | Reconstruction Error |
|-------|---------------------|
| 100 | 1.2756 |
| 200 | 0.7665 |
| 300 | 0.5552 |
| 400 | 0.5149 |
| 500 | 0.4988 |

The reconstruction error decreased from 1.28 to 0.50, showing that the RBM learned to reconstruct the input data better over time.

### Demo Listener Result

**Input:** `[1, 1, 0, 1, 0, 1]` (likes Pop, Rock, Hip-Hop, Electronic)

**Hidden activation probabilities:**
- H1: 0.9978
- H2: 0.9892

**Hidden representation:** `[1, 1]`

### All Listeners

| Listener | Input Vector | H1 | H2 | Hidden |
|----------|-------------|------|------|--------|
| L1 | [1,1,0,1,0,1] | 0.9978 | 0.9892 | [1,1] |
| L2 | [1,0,0,1,0,1] | 0.9974 | 0.9833 | [1,1] |
| L3 | [0,1,1,0,1,0] | 0.0009 | 0.0057 | [0,0] |
| L4 | [1,1,0,0,0,1] | 0.9804 | 0.9522 | [1,1] |
| L5 | [0,0,1,0,1,0] | 0.0007 | 0.0037 | [0,0] |
| L6 | [1,0,0,1,0,1] | 0.9974 | 0.9833 | [1,1] |
| L7 | [0,1,1,0,1,0] | 0.0009 | 0.0057 | [0,0] |
| L8 | [1,1,0,1,0,1] | 0.9978 | 0.9892 | [1,1] |
| L9 | [0,0,1,0,1,1] | 0.0031 | 0.0102 | [0,0] |
| L10 | [1,1,0,1,0,0] | 0.9908 | 0.9705 | [1,1] |

### Reconstruction Demo

The RBM successfully reconstructed the demo listener's preferences with **6/6 genres matching** the original input.

---

## 8. Analysis

### What the RBM Learned

The trained weight matrix reveals two clear patterns:

**Genres with positive weights (activate hidden units):**
- Pop (+3.01, +2.21)
- Hip-Hop (+2.22, +1.52)
- Electronic (+1.45, +1.02)

**Genres with negative weights (suppress hidden units):**
- Classical (-3.21, -2.49)
- Jazz (-3.25, -2.43)

**Genre with neutral weights:**
- Rock (+0.19, +0.44) — shared across both groups

### Interpretation

The RBM learned to distinguish two listener groups:
- **Hidden = [1, 1]:** Listeners who prefer mainstream/energetic genres (Pop, Hip-Hop, Electronic).
- **Hidden = [0, 0]:** Listeners who prefer instrumental/traditional genres (Classical, Jazz).

This demonstrates that an RBM can automatically discover latent preference patterns from binary data without being told what the patterns are.

### Why Rock Has Small Weights

Rock appears in both listener groups (some mainstream listeners like Rock, some instrumental/alternative listeners also like Rock). Therefore, the RBM assigns small weights to Rock because it doesn't strongly distinguish between the two groups.

---

## 9. Conclusion

1. A Restricted Boltzmann Machine can learn hidden preference patterns from binary listener data.
2. The visible layer successfully represents observable genre preferences.
3. The hidden layer captures latent patterns that distinguish listener groups.
4. Contrastive Divergence (CD-1) is a simple and effective training method.
5. The trained RBM can encode 6-dimensional preference vectors into 2-dimensional hidden representations.
6. The RBM can also reconstruct preferences from hidden representations with high accuracy.

### Limitations

- The dataset is very small (10 listeners) — a real system would need thousands.
- Only 2 hidden units — more complex patterns would require more hidden units.
- Basic CD-1 training — more advanced methods exist (CD-k, Persistent CD).
- Binary data only — real preferences might be continuous ratings.
- No evaluation on unseen data — this is a demonstration, not a validated model.

---

## 10. References

1. Hinton, G. E. (2002). "Training Products of Experts by Minimizing Contrastive Divergence." *Neural Computation*, 14(8), 1771-1800.
2. Fischer, A., & Igel, C. (2012). "An Introduction to Restricted Boltzmann Machines." *Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications*, 14-36.
3. Hinton, G. E. (2010). "A Practical Guide to Training Restricted Boltzmann Machines." *Technical Report, University of Toronto.*
4. Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). "Restricted Boltzmann Machines for Collaborative Filtering." *Proceedings of the 24th International Conference on Machine Learning.*
5. NumPy Documentation — https://numpy.org/doc/
6. Matplotlib Documentation — https://matplotlib.org/stable/
