"""
ann.py
======
HEATGUARD AI  -  the Artificial Neural Network, written from scratch
with NumPy only.  No TensorFlow / PyTorch / Keras.

ARCHITECTURE
    6 inputs  ->  8 hidden neurons (sigmoid)  ->  1 output neuron (sigmoid)

THE FIVE EQUATIONS THIS FILE IMPLEMENTS
    1. Weighted sum      net_j = SUM_i (x_i * w_ij) + b_j
    2. Activation        sigmoid(net) = 1 / (1 + e^(-net))
    3. Loss (BCE)        E = -[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
    4. Backpropagation   dE/dw  (chain rule, layer by layer)
    5. Weight update     w_new = w_old - eta * dE/dw     (gradient descent)

Every number the Streamlit app displays as a "model result" comes from
these functions.  Nothing in the UI is hard-coded.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Activation function
# --------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid squashes any real number into (0, 1).

        sigmoid(z) = 1 / (1 + e^(-z))

    We use it because (a) the output must look like a score between 0 and 1,
    and (b) it is smooth, so it has a derivative everywhere - which is what
    backpropagation needs.

    The np.clip keeps exp() from overflowing for very large |z|.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_output(a: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid, written in terms of its own OUTPUT:

        d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)

    Using `a` (already computed in the forward pass) avoids recomputing
    the exponential - this is the standard trick.
    """
    return a * (1.0 - a)


# --------------------------------------------------------------------------
# Loss function
# --------------------------------------------------------------------------
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary cross-entropy (log loss), averaged over the samples:

        E = -(1/m) * SUM [ y*log(y_hat) + (1-y)*log(1-y_hat) ]

    It is the natural loss for a 0/1 target with a sigmoid output: it
    punishes a confident wrong answer very heavily.
    The epsilon clip keeps log(0) from becoming -infinity.
    """
    eps = 1e-12
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
class SimpleANN:
    """A one-hidden-layer feed-forward network trained by gradient descent."""

    def __init__(self, n_input: int = 6, n_hidden: int = 8, n_output: int = 1, seed: int = 42):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.seed = seed

        rng = np.random.default_rng(seed)

        # WHY RANDOM (and SMALL) INITIAL WEIGHTS?
        #  * If every weight started at 0, every hidden neuron would compute
        #    exactly the same thing and receive exactly the same gradient -
        #    they could never specialise ("symmetry breaking").
        #  * If the weights started large, net would be large, sigmoid would
        #    saturate near 0 or 1, its derivative a*(1-a) would be ~0 and
        #    learning would stall ("vanishing gradient").
        # The Xavier/Glorot range below keeps the initial signal well scaled.
        limit1 = np.sqrt(6.0 / (n_input + n_hidden))
        limit2 = np.sqrt(6.0 / (n_hidden + n_output))

        self.W1 = rng.uniform(-limit1, limit1, size=(n_input, n_hidden))   # input -> hidden
        self.b1 = np.zeros((1, n_hidden))                                   # hidden biases
        self.W2 = rng.uniform(-limit2, limit2, size=(n_hidden, n_output))  # hidden -> output
        self.b2 = np.zeros((1, n_output))                                   # output bias

        # Snapshot of the starting weights. The "initial vs current" table on
        # the Neural Laboratory page compares against these - they are the real
        # random values this network began life with, never invented numbers.
        self.W1_init = self.W1.copy()
        self.b1_init = self.b1.copy()
        self.W2_init = self.W2.copy()
        self.b2_init = self.b2.copy()

        # Training history, filled in by fit()
        self.history = {"epoch": [], "train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}
        self.is_trained = False
        self.epochs_run = 0
        self.learning_rate = None

    # ---------------------------------------------------------------- copy
    def copy(self) -> "SimpleANN":
        """An independent clone (used so the Neural Lab cannot damage the trained model)."""
        clone = SimpleANN(self.n_input, self.n_hidden, self.n_output, self.seed)
        clone.W1, clone.b1 = self.W1.copy(), self.b1.copy()
        clone.W2, clone.b2 = self.W2.copy(), self.b2.copy()
        clone.W1_init, clone.b1_init = self.W1_init.copy(), self.b1_init.copy()
        clone.W2_init, clone.b2_init = self.W2_init.copy(), self.b2_init.copy()
        clone.is_trained = self.is_trained
        clone.epochs_run = self.epochs_run
        clone.learning_rate = self.learning_rate
        clone.history = {k: list(v) for k, v in self.history.items()}
        return clone

    # ------------------------------------------------------- FORWARD PASS
    def forward(self, X: np.ndarray) -> dict:
        """
        FORWARD PROPAGATION - push the inputs through the network.

        Step 1  hidden weighted sums   z1 = X . W1 + b1     (net_j)
        Step 2  hidden activations     a1 = sigmoid(z1)     (h_j)
        Step 3  output weighted sum    z2 = a1 . W2 + b2
        Step 4  final score            a2 = sigmoid(z2)     (y_hat)

        Everything is kept in the returned dictionary ("cache") because
        backpropagation needs these intermediate values, and because the
        Neural Laboratory page displays them.
        """
        X = np.asarray(X, dtype=float).reshape(-1, self.n_input)
        z1 = X @ self.W1 + self.b1          # (m, n_hidden)
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2         # (m, 1)
        a2 = sigmoid(z2)
        return {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """The ANN's score in [0, 1] - reported in the UI as an *estimated* score."""
        return self.forward(X)["a2"]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 decision using a threshold on the score."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------------------------------------------------------- BACKWARD PASS
    def backward(self, cache: dict, y: np.ndarray) -> dict:
        """
        BACKPROPAGATION - work out how much each weight contributed to the
        error, by applying the chain rule backwards through the network.

        OUTPUT LAYER
            With binary cross-entropy AND a sigmoid output the two
            derivatives cancel beautifully:

                dE/dz2 = y_hat - y                       <-- "output delta"

            (that is dE/dy_hat * dy_hat/dz2 simplified).

            dE/dW2 = a1^T . dz2          dE/db2 = sum(dz2)

        HIDDEN LAYER
            The error is pushed back through W2 and then through the
            sigmoid derivative of the hidden layer:

                dE/da1 = dz2 . W2^T
                dE/dz1 = dE/da1 * a1 * (1 - a1)          <-- "hidden delta"

            dE/dW1 = X^T . dz1           dE/db1 = sum(dz1)

        Dividing by m averages the gradient over the batch so the learning
        rate does not have to change when the batch size changes.
        """
        X, a1, a2 = cache["X"], cache["a1"], cache["a2"]
        y = np.asarray(y, dtype=float).reshape(-1, self.n_output)
        m = X.shape[0]

        dz2 = (a2 - y) / m                       # output delta (BCE + sigmoid)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative_from_output(a1)   # hidden delta
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
                "delta_out": dz2, "delta_hidden": dz1}

    # ------------------------------------------------------- WEIGHT UPDATE
    def apply_gradients(self, grads: dict, learning_rate: float) -> None:
        """
        GRADIENT DESCENT

            w_new = w_old - eta * (dE/dw)

        We step in the direction that makes the error smaller.  eta (the
        learning rate) controls the step size: too small and learning is
        slow, too large and the updates overshoot and oscillate.
        """
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    # ------------------------------------------------------------- TRAINING
    def fit(self, X, y, epochs: int = 1500, learning_rate: float = 0.5,
            X_val=None, y_val=None, record_every: int = 10,
            restore_best: bool = True) -> dict:
        """
        Train with full-batch gradient descent.

        One epoch = one forward pass over all training rows, one
        backpropagation, one weight update.

        `restore_best` keeps a snapshot of the weights that gave the lowest
        validation loss and puts them back at the end.  This is a simple
        form of early stopping and is our defence against overfitting.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.learning_rate = learning_rate
        self.history = {"epoch": [], "train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}

        best_val, best_weights, best_epoch = np.inf, None, 0
        has_val = X_val is not None and len(X_val) > 0

        for epoch in range(1, epochs + 1):
            cache = self.forward(X)                       # 1. forward
            grads = self.backward(cache, y)               # 2. backward
            self.apply_gradients(grads, learning_rate)    # 3. update

            if epoch % record_every == 0 or epoch == 1 or epoch == epochs:
                tr_p = self.predict_proba(X)
                tr_loss = binary_cross_entropy(y, tr_p)
                tr_acc = float(np.mean((tr_p >= 0.5).astype(int) == y))

                if has_val:
                    va_p = self.predict_proba(X_val)
                    yv = np.asarray(y_val, dtype=float).reshape(-1, 1)
                    va_loss = binary_cross_entropy(yv, va_p)
                    va_acc = float(np.mean((va_p >= 0.5).astype(int) == yv))
                else:
                    va_loss, va_acc = np.nan, np.nan

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(tr_loss)
                self.history["val_loss"].append(va_loss)
                self.history["train_acc"].append(tr_acc)
                self.history["val_acc"].append(va_acc)

                if has_val and restore_best and va_loss < best_val:
                    best_val, best_epoch = va_loss, epoch
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy())

        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights

        self.is_trained = True
        self.epochs_run = epochs
        self.best_epoch = best_epoch if best_weights is not None else epochs
        return self.history

    def train_epochs(self, X, y, n_epochs: int = 10, learning_rate: float = 0.8) -> list:
        """
        Run `n_epochs` full-batch epochs and return the loss after each one.

        Used by the Neural Laboratory's "Train 10 Epochs" button. It is exactly
        the same three steps as fit() - forward, backward, update - just run a
        handful of times so the user can watch the loss come down.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        losses = []
        for _ in range(int(n_epochs)):
            cache = self.forward(X)                       # 1. forward
            grads = self.backward(cache, y)               # 2. backward
            self.apply_gradients(grads, learning_rate)    # 3. update
            losses.append(binary_cross_entropy(y, self.predict_proba(X)))
        return losses

    def weight_change_table(self):
        """
        Every input->hidden connection as (name-index pair, initial, current).

        Returned as plain arrays so app.py can build the
        "Connection / Initial / Current / Change" table without inventing
        a single number.
        """
        pairs, initial, current = [], [], []
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                pairs.append((i, j))
                initial.append(self.W1_init[i, j])
                current.append(self.W1[i, j])
        return pairs, np.array(initial), np.array(current)

    # ------------------------------------------ ONE VISIBLE LEARNING STEP
    def learn_one_step(self, x_sample: np.ndarray, y_target: float,
                       learning_rate: float = 0.5, watch_hidden: int = 0) -> dict:
        """
        Perform ONE training update on ONE example and return every number
        involved, so the Neural Laboratory page can display the real
        arithmetic (forward -> error -> backprop -> weight update).

        This is ordinary stochastic gradient descent with a batch of one.
        """
        x = np.asarray(x_sample, dtype=float).reshape(1, self.n_input)
        y = np.array([[float(y_target)]])

        # ---- before
        cache_before = self.forward(x)
        pred_before = float(cache_before["a2"][0, 0])
        loss_before = binary_cross_entropy(y, cache_before["a2"])

        grads = self.backward(cache_before, y)

        # Snapshot the weights we are going to show changing.
        j = int(np.clip(watch_hidden, 0, self.n_hidden - 1))
        w1_old = self.W1[:, j].copy()      # the 6 weights feeding hidden neuron j
        w2_old = self.W2[:, 0].copy()      # the 8 hidden -> output weights
        b2_old = float(self.b2[0, 0])

        # ---- the actual update:  w_new = w_old - eta * gradient
        self.apply_gradients(grads, learning_rate)

        # ---- after
        cache_after = self.forward(x)
        pred_after = float(cache_after["a2"][0, 0])
        loss_after = binary_cross_entropy(y, cache_after["a2"])

        return {
            "watch_hidden": j,
            "target": float(y_target),
            "pred_before": pred_before,
            "pred_after": pred_after,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "raw_error": float(y_target) - pred_before,          # (y - y_hat)
            "output_delta": float(grads["delta_out"][0, 0]),     # dE/dz2
            "w1_old": w1_old, "w1_grad": grads["dW1"][:, j].copy(),
            "w1_delta": -learning_rate * grads["dW1"][:, j],
            "w1_new": self.W1[:, j].copy(),
            "w2_old": w2_old, "w2_grad": grads["dW2"][:, 0].copy(),
            "w2_delta": -learning_rate * grads["dW2"][:, 0],
            "w2_new": self.W2[:, 0].copy(),
            "b2_old": b2_old, "b2_new": float(self.b2[0, 0]),
            "hidden_activations": cache_before["a1"][0].copy(),
            "learning_rate": learning_rate,
        }


# --------------------------------------------------------------------------
# Evaluation metrics - all computed from real predictions
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Accuracy, precision, recall, F1 and the confusion matrix, from scratch.

        TP: predicted heat wave, was a heat wave
        TN: predicted no heat wave, was no heat wave
        FP: predicted heat wave, was not  (false alarm)
        FN: predicted no heat wave, but it was one  (missed warning)

        accuracy  = (TP + TN) / total
        precision = TP / (TP + FP)      "when we warn, how often are we right"
        recall    = TP / (TP + FN)      "of the real heat waves, how many did we catch"
        f1        = 2 * P * R / (P + R)
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = (np.asarray(y_prob, dtype=float).reshape(-1) >= threshold).astype(int)
    y_true_i = y_true.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))
    total = max(tp + tn + fp + fn, 1)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": total,
            "n_positive": int(np.sum(y_true_i == 1)), "threshold": threshold}
