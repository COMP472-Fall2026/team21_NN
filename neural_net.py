"""
Feed-forward neural network (MLP) for both tasks, in PyTorch.

    python neural_net.py                      # one run per task with the default settings
    python neural_net.py --task image         # one task only
    python neural_net.py --sweep              # hyperparameter sweep (lr, hidden, dropout, weight decay)
    python neural_net.py --curves             # train vs validation loss plots -> reports/nn_figures/

Plugs into common.py like the other models: NeuralNet has fit(X, y) and predict(X),
both on numpy arrays. Early stopping uses a 10% hold-out taken from the training
rows it is given, so the shared validation split is never used for training decisions.

References: PyTorch "Learn the Basics" tutorials and learnpytorch.io chapter 02.
"""

import argparse
import copy
import itertools
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from common import BASE_DIR, SEED, evaluate, get_split, set_seed

# A network this small runs as fast on CPU as on the Mac GPU (mps), and CPU is reproducible.
DEVICE = "cpu"
FIG_DIR = os.path.join(BASE_DIR, "reports", "nn_figures")


# --------------------------------------------------------------- network ---
class MLP(nn.Module):
    """input -> [Linear -> ReLU -> Dropout] per hidden size -> Linear -> class scores."""

    def __init__(self, input_dim, n_classes, hidden=(256, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        # No softmax: CrossEntropyLoss applies it internally to the raw scores (logits).
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------- fit/predict API ---
class NeuralNet:
    """fit/predict wrapper so common.evaluate() and make_submission.py can use the MLP.

    hidden       : sizes of the hidden layers, e.g. (256, 64)
    dropout      : fraction of hidden units switched off at random during training
    lr           : learning rate for Adam
    weight_decay : L2 penalty on the weights (regularization)
    epochs       : maximum number of passes over the training data
    patience     : stop after this many epochs without the hold-out loss improving
    """

    def __init__(self, task=None, hidden=(256, 64), dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 epochs=100, batch_size=128, patience=8, holdout=0.1, verbose=False):
        self.task = task  # unused; make_submission.py passes it
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.holdout = holdout
        self.verbose = verbose

    def fit(self, X, y):
        set_seed()
        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X, y, test_size=self.holdout, stratify=y, random_state=SEED)

        # numpy -> tensors (features float32, labels int64 as CrossEntropyLoss needs)
        X_fit_t, y_fit_t = torch.from_numpy(X_fit).float(), torch.from_numpy(y_fit).long()
        X_hold_t, y_hold_t = torch.from_numpy(X_hold).float(), torch.from_numpy(y_hold).long()
        loader = DataLoader(TensorDataset(X_fit_t, y_fit_t), batch_size=self.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(SEED))

        n_classes = int(y.max()) + 1
        self.model = MLP(X.shape[1], n_classes, self.hidden, self.dropout).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.history = {"train_loss": [], "hold_loss": [], "hold_acc": []}
        best_loss, best_state, bad_epochs = float("inf"), None, 0

        for epoch in range(self.epochs):
            # --- train one epoch: forward, loss, backward, step
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = loss_fn(self.model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            train_loss = total / len(X_fit_t)

            # --- check the hold-out set (no dropout, no gradients)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_hold_t.to(DEVICE))
                hold_loss = loss_fn(logits, y_hold_t.to(DEVICE)).item()
                hold_acc = (logits.argmax(1).cpu() == y_hold_t).float().mean().item()

            self.history["train_loss"].append(train_loss)
            self.history["hold_loss"].append(hold_loss)
            self.history["hold_acc"].append(hold_acc)
            if self.verbose:
                print(f"  epoch {epoch + 1:3d}  train loss {train_loss:.4f}  "
                      f"hold-out loss {hold_loss:.4f}  hold-out acc {hold_acc:.4f}")

            # --- early stopping: keep the weights from the best hold-out epoch
            if hold_loss < best_loss - 1e-4:
                best_loss, bad_epochs = hold_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history["hold_loss"])) + 1
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE))
        return logits.argmax(1).cpu().numpy()


# ----------------------------------------------------------- experiments ---
# Best settings found so far per task (from manual probing; confirm with --sweep).
DEFAULTS = {
    "image": {"hidden": (1024, 256), "dropout": 0.5, "lr": 3e-4, "weight_decay": 1e-3},
    "sentiment": {"hidden": (512,), "dropout": 0.5, "lr": 3e-4, "weight_decay": 1e-3},
}

SWEEP = {
    "lr": [1e-2, 1e-3, 3e-4, 1e-4],
    "hidden": [(256, 64), (512,), (1024, 256)],
    "dropout": [0.0, 0.3, 0.5],
    "weight_decay": [0.0, 1e-4, 1e-3],
}


def run(task, **params):
    """One evaluate() run, logged to results/results.csv as model 'neural_net'."""
    p = {**DEFAULTS[task], **params}
    model = NeuralNet(**p)
    logged = {**p, "hidden": "-".join(map(str, p["hidden"]))}
    res = evaluate(model, task, name="neural_net", params=logged)
    print(f"    stopped at epoch {len(model.history['train_loss'])}, best epoch {model.best_epoch}")
    return res


def sweep(task):
    """Full grid over SWEEP (108 runs per task, roughly 10-20 min per task on CPU)."""
    keys = list(SWEEP)
    results = []
    for values in itertools.product(*(SWEEP[k] for k in keys)):
        params = dict(zip(keys, values))
        res = run(task, **params)
        results.append((res["val_acc"], params))
    results.sort(key=lambda r: -r[0])
    print(f"\n[{task}] top 5 settings by validation accuracy:")
    for acc, params in results[:5]:
        print(f"  {acc:.4f}  {params}")


def curves(task):
    """Train vs validation loss per epoch, without and with regularization (overfitting evidence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_tr, X_val, y_tr, y_val = get_split(task)
    configs = {
        "no regularization (dropout 0, wd 0)": {"dropout": 0.0, "weight_decay": 0.0},
        "with regularization (best settings)": {},
    }
    fig, axes = plt.subplots(1, len(configs), figsize=(11, 4), sharey=True)
    for ax, (label, extra) in zip(axes, configs.items()):
        # Fixed number of epochs with no early stopping, so the overfitting is visible.
        model = NeuralNet(**{**DEFAULTS[task], **extra, "epochs": 60, "patience": 10**9}).fit(X_tr, y_tr)
        h = model.history
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label="train loss")
        ax.plot(range(1, len(h["hold_loss"]) + 1), h["hold_loss"], label="hold-out loss")
        ax.axvline(model.best_epoch, color="gray", ls="--", lw=1, label=f"best epoch ({model.best_epoch})")
        val_acc = (model.predict(X_val) == y_val).mean()
        ax.set_title(f"{label}\nval acc at best epoch: {val_acc:.3f}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("cross-entropy loss")
    fig.suptitle(f"Neural network on {task}: train vs hold-out loss")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"nn_loss_curves_{task}.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["image", "sentiment"], help="default: both")
    ap.add_argument("--sweep", action="store_true", help="hyperparameter grid search")
    ap.add_argument("--curves", action="store_true", help="save train vs hold-out loss plots")
    args = ap.parse_args()

    tasks = [args.task] if args.task else ["image", "sentiment"]
    for t in tasks:
        if args.sweep:
            sweep(t)
        elif args.curves:
            curves(t)
        else:
            run(t)
