# Restricted Boltzmann Machine — Music Listener Preference Demo

## Problem Statement

A music service represents listener choices using visible variables and hidden preference features. Construct a simple RBM with the specified visible and hidden units, apply one listener vector conceptually to the visible layer, and demonstrate the resulting hidden representation.

## Objective

Build a simple Restricted Boltzmann Machine (RBM) from scratch using Python and NumPy to demonstrate how a music service can represent listener preferences.

The project will:

1. Create an RBM with **6 visible units** (music genre preferences) and **2 hidden units** (latent preference features).
2. Use a small binary dataset of listener preferences.
3. Train the RBM using a simple learning procedure.
4. Show how a listener's preference vector is transformed into a hidden representation.

## What is an RBM?

A Restricted Boltzmann Machine (RBM) is a type of generative neural network with two layers:

- **Visible layer** — represents the observed data (in our case, whether a listener likes a music genre or not).
- **Hidden layer** — represents latent (hidden) features learned from the data.

The key restriction is that there are **no connections within the same layer** — visible units are only connected to hidden units, and vice versa.

### Visible Units (6 music genres)

| Unit | Genre |
|------|-------|
| V1 | Pop |
| V2 | Rock |
| V3 | Classical |
| V4 | Hip-Hop |
| V5 | Jazz |
| V6 | Electronic |

### Hidden Units (2 latent features)

| Unit | Conceptual Interpretation |
|------|--------------------------|
| H1 | Latent Preference Pattern 1 |
| H2 | Latent Preference Pattern 2 |

> **Note:** Hidden units learn latent features from data automatically. They do not receive human-readable labels. Any interpretation (e.g., "Mainstream preference" or "Instrumental preference") is a conceptual approximation based on the learned weights — not a built-in label.

## Project Scope

- This is an **educational demonstration**, not a production recommendation system.
- The code is kept simple so it can be understood and explained during a viva.
- Python and NumPy are used — no deep learning frameworks like TensorFlow or PyTorch.