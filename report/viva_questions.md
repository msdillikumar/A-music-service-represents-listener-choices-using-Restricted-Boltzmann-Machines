# Viva Questions and Answers — RBM Music Preference Demo

## Q1. What is a Restricted Boltzmann Machine (RBM)?

**Answer:** An RBM is a type of generative neural network with two layers — a visible layer and a hidden layer. It learns to find patterns in data by adjusting the weights between visible and hidden units. It is called "restricted" because there are no connections between units in the same layer — visible units only connect to hidden units.

---

## Q2. What are visible units in this project?

**Answer:** Visible units represent the observable data — in this case, a listener's music genre preferences. We have 6 visible units: Pop, Rock, Classical, Hip-Hop, Jazz, and Electronic. Each unit has a binary value: 1 means the listener likes that genre, 0 means they do not.

---

## Q3. What are hidden units?

**Answer:** Hidden units represent latent (hidden) features that the RBM learns from the data. In this project we have 2 hidden units. They capture underlying preference patterns — for example, one pattern might correspond to listeners who prefer mainstream genres and another to those who prefer instrumental genres. However, the RBM discovers these patterns automatically.

---

## Q4. Why are hidden units called latent features?

**Answer:** They are called "latent" because they are not directly observed in the data. They are internal variables that the model learns to represent meaningful patterns. We don't tell the RBM what the hidden units should mean — it learns the best representation on its own during training.

---

## Q5. What is the role of weights in an RBM?

**Answer:** Weights determine how strongly each visible unit influences each hidden unit. A large positive weight means that when the visible unit is active (1), it strongly activates the connected hidden unit. A large negative weight means it suppresses the hidden unit. Weights are learned during training.

---

## Q6. What is bias in an RBM?

**Answer:** Bias is an extra value added to each unit's activation before applying the sigmoid function. It shifts the activation threshold — a positive bias makes a unit more likely to activate, and a negative bias makes it less likely. There are separate biases for visible units and hidden units.

---

## Q7. Why is the sigmoid function used?

**Answer:** The sigmoid function converts any real number into a value between 0 and 1, which we interpret as a probability. The formula is `sigmoid(x) = 1 / (1 + exp(-x))`. We use it to calculate the probability that a hidden unit is activated given the visible input, and vice versa.

---

## Q8. What does a binary value of 1 mean in this project?

**Answer:** A value of 1 in the visible layer means the listener likes or listens to that genre. A value of 0 means they do not like that genre. In the hidden layer, 1 means the hidden unit is activated (the corresponding latent feature is present), and 0 means it is not.

---

## Q9. What does the listener vector represent?

**Answer:** The listener vector is a row of 6 binary values representing one listener's genre preferences. For example, `[1, 1, 0, 1, 0, 1]` means the listener likes Pop (1), Rock (1), does not like Classical (0), likes Hip-Hop (1), does not like Jazz (0), and likes Electronic (1).

---

## Q10. How is hidden activation calculated?

**Answer:** The hidden activation probability for each hidden unit is calculated as:

1. Multiply the listener vector by the weight matrix: `v · W`
2. Add the hidden bias: `v · W + hidden_bias`
3. Apply the sigmoid function: `sigmoid(v · W + hidden_bias)`

This gives a probability between 0 and 1 for each hidden unit. If the probability is ≥ 0.5, the hidden unit is set to 1; otherwise, it is set to 0.

---

## Q11. What is Contrastive Divergence (CD)?

**Answer:** Contrastive Divergence is the learning algorithm used to train an RBM. The basic idea is:

1. Start with real data (visible input) and compute the hidden probabilities (positive phase).
2. From the hidden state, reconstruct the visible layer.
3. Compute hidden probabilities again from the reconstruction (negative phase).
4. Update weights by the difference: `positive_associations - negative_associations`.

CD-1 means we do only 1 step of reconstruction, which is simple and works well in practice.

---

## Q12. What happens during reconstruction?

**Answer:** Reconstruction is when the RBM generates visible layer probabilities from the hidden layer. Using the formula `P(v=1|h) = sigmoid(h · W^T + visible_bias)`, the RBM tries to recreate the original input from the hidden representation. A good reconstruction means the RBM has learned the data patterns well.

---

## Q13. Why are there no connections between visible units?

**Answer:** This is the "restriction" in Restricted Boltzmann Machines. If visible units were connected to each other, the model would be a full Boltzmann Machine, which is much harder to train. By restricting connections to only between visible and hidden layers, the computation becomes simpler and training is efficient.

---

## Q14. Why are there no connections between hidden units?

**Answer:** For the same reason — this restriction makes training feasible. With no intra-layer connections, all hidden units are conditionally independent given the visible layer. This means we can compute all hidden unit probabilities simultaneously in one step, rather than needing complex iterative methods.

---

## Q15. What are the limitations of this small project?

**Answer:**

1. **Small dataset:** Only 10 listeners — a real system would need thousands.
2. **Few hidden units:** Only 2 hidden units — more complex patterns would need more.
3. **Basic training:** We use CD-1, the simplest form. More advanced methods (CD-k, Persistent CD) exist.
4. **Binary data only:** Real preferences could be ratings (1-5 stars), not just yes/no.
5. **No evaluation:** We don't test on unseen data — this is a demonstration, not a production model.
6. **Manual interpretation:** The hidden unit meanings are our interpretation of the learned weights, not built-in labels.
