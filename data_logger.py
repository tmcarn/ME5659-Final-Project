"""Simple time-series logger for state and controller output data."""

import os
from datetime import datetime

import numpy as np


class DataLogger:
    """
    Records time, state vector, and controller output at each control step.

    Usage:
        logger = DataLogger()
        logger.log(data.time, state, tau)
        ...
        logger.plot()     # writes logs/run_YYYYmmdd_HHMMSS.png
    """

    def __init__(self,
                 state_names=("x", "theta", "x_dot", "theta_dot"),
                 output_names=("tau",),
                 log_dir="logs"):
        self.state_names = list(state_names)
        self.output_names = list(output_names)
        self.log_dir = log_dir
        self.clear()

    def clear(self):
        self._t = []
        self._x = []
        self._u = []

    def __len__(self):
        return len(self._t)

    def log(self, t, state, output):
        """Record one sample. `state` and `output` may be scalars or arrays."""
        state = np.atleast_1d(np.asarray(state, dtype=float)).copy()
        output = np.atleast_1d(np.asarray(output, dtype=float)).copy()

        if state.size != len(self.state_names):
            raise ValueError(f"Expected {len(self.state_names)} states, got {state.size}")
        if output.size != len(self.output_names):
            raise ValueError(f"Expected {len(self.output_names)} outputs, got {output.size}")

        self._t.append(float(t))
        self._x.append(state)
        self._u.append(output)

    def as_arrays(self):
        """Return (t, X, U) with shapes (N,), (N, n_states), (N, n_outputs)."""
        n_x, n_u = len(self.state_names), len(self.output_names)
        t = np.array(self._t)
        X = np.array(self._x) if self._x else np.empty((0, n_x))
        U = np.array(self._u) if self._u else np.empty((0, n_u))
        return t, X, U

    def plot(self, path=None):
        """
        Save a plot of each state and output vs. time to a PNG. Returns the path.

        Uses matplotlib's Figure directly (no pyplot / GUI backend), so it is
        safe to call from mjpython's worker thread.
        """
        from matplotlib.figure import Figure

        if path is None:
            os.makedirs(self.log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.log_dir, f"run_{stamp}.png")

        t, X, U = self.as_arrays()
        names = self.state_names + self.output_names
        cols = np.hstack([X, U])

        fig = Figure(figsize=(8, 1.8 * len(names)))
        axes = np.atleast_1d(fig.subplots(len(names), 1, sharex=True))
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.plot(t, cols[:, i])
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(path, dpi=120)

        print(f"Saved plot to {path}")
        return path
