"""Planner interface: abstract base class for all trajectory planners."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["Trajectory", "Planner", "PlanningError", "DEFAULT_MAX_SPEED"]

# Module-level constant shared by every planner.
DEFAULT_MAX_SPEED = 12.0  # m/s


class PlanningError(Exception):
    """Raised when a planner cannot produce a valid trajectory."""
    pass


@dataclass
class Trajectory:
    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray


class Planner(ABC):
    @abstractmethod
    def plan(self, start_state, gates, obstacles) -> Trajectory:
        """Compute a trajectory through the gates. Subclasses must implement."""
        ...

