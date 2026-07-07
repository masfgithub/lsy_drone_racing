"""Environment Observation Data Format.

Extract relevant information from the environment in a standardized dataformat.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnvState:
    """Standardized container for the drone/gate/obstacle state extracted from an observation."""

    p_bll: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # position of the body in local coordinates wrt. local coordinates
    v_bll: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # velocity of the body in local coordinates wrt. local coordinates
    w_bll: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # angular velocities of the body in local coordinates wrt. local coordinates
    q_blb: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )  # quaternions of the body from local frame to body coordinates
    p_tll_array: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 3), dtype=np.float64)
    )  # array of positions of the targets in local coordinates wrt. local coordinates
    p_tll_index: np.ndarray = field(default_factory=lambda: np.zeros(3))  # index of the next target
    q_tlt_array: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 6), dtype=np.float64)
    )  # array of quaternions of the targets from local frame to target coordinates
    p_oll_array: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 3), dtype=np.float64)
    )  # array of positions of the obstacles in local coordinates wrt. local coordinates
    h_t: float = 0.3  # height of the targets (usually it is a constant)
    l_t: float = 0.3  # length of the targets (usually it is a constant)
    w_t: float = 0.02  # width of the targets (usually it is a constant)


def extract_env_states(obs: dict[str, NDArray[np.floating]]) -> EnvState:
    """Extract and return environment states from the observation dictionary."""
    states = EnvState()
    states.p_bll = obs["pos"]
    states.v_bll = obs["vel"]
    states.w_bll = obs["ang_vel"]
    states.q_blb = obs["quat"]
    states.p_tll_array = obs["gates_pos"]
    states.p_tll_index = obs["target_gate"]
    states.q_tlt_array = obs["gates_quat"]
    states.p_oll_array = obs["obstacles_pos"]
    # Gate dimensions are not part of the observation dict, so fall back to fixed
    # nominal values (height/length 30 cm, frame width 2 cm) until the env exposes them.
    states.h_t = 0.3
    states.l_t = 0.3
    states.w_t = 0.02

    return states
