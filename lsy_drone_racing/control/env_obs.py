"""Environment Observation Data Format.

Extract relevant information from the environment in a standardized dataformat.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnvState_t:
    """TBD: State container for environment observations."""
    pBLL: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # position of the body in local coordinates wrt. local coordinates
    vBLL: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # velocity of the body in local coordinates wrt. local coordinates
    wBLL: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # angular velocities of the body in local coordinates wrt. local coordinates
    qBLB: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )  # quaternions of the body from local frame to body coordinates
    pTLL_array: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 3), dtype=np.float64)
    )  # array of positions of the targets in local coordinates wrt. local coordinates
    pTLL_index: np.ndarray = field(default_factory=lambda: np.zeros(3))  # index of the next target
    qTLT_array: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 6), dtype=np.float64)
    )  # array of quaternions of the targets from local frame to target coordinates
    pOLL: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # array of positions of the obstacles in local coordinates wrt. local coordinates
    hT: float = 0.3  # height of the targets (usually it is a constant)
    lT: float = 0.3  # length of the targets (usually it is a constant)
    wT: float = 0.02  # width of the targets (usually it is a constant)


def extract_env_states(obs: dict[str, NDArray[np.floating]]) -> EnvState_t:
    """Extract and return environment states from the observation dictionary."""
    states = EnvState_t()
    states.pBLL = obs["pos"]
    states.vBLL = obs["vel"]
    states.wBLL = obs["ang_vel"]
    states.qBLB = obs["quat"]
    states.pTLL_array = obs["gates_pos"]
    states.pTLL_index = obs["target_gate"]
    states.qTLT_array = obs["gates_quat"]
    states.pOLL = obs["obstacles_pos"]
    states.hT = 0.3  # for now 30 cm, TBD: adjust
    states.lT = 0.3  # for now 30 cm, TBD: adjust
    states.wT = 0.02  # for now 2 cm, TBD: adjust

    return states
