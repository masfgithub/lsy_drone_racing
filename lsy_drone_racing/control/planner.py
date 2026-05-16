"""Planner interface: abstract base class for all trajectory planners."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["Trajectory", "Planner", "PlanningError", "DEFAULT_MAX_SPEED"]

# Module-level constant shared by every planner.
DEFAULT_MAX_SPEED = 12.0  # m/s


class PlanningError(Exception):
    """Raised when a planner cannot produce a valid trajectory."""


@dataclass
class Trajectory:
    """Output of every planner: sampled positions, velocities and times."""
    positions: np.ndarray   # shape (N, 3)
    velocities: np.ndarray  # shape (N, 3)
    timestamps: np.ndarray  # shape (N,)


class Planner(ABC):
    """Abstract base class for all trajectory planners."""

    @abstractmethod
    def plan(self, start_state, gates, goal_state) -> Trajectory:
        """Compute a trajectory through the gates. Subclasses must implement."""
        ...