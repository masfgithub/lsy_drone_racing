"""<For RUFF: Brief description of what this module does>."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ControllerInterface(ABC):
    """Base class for controller implementations."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], 
                 planner: dict, 
                 info: dict, 
                 config: dict,
                 t_total: int):
        """Initialization of the controller.

        Instructions:
            The controller's constructor has access the initial observation `obs`, the a priori
            information contained in dictionary `info`, and the config of the race track. Use this
            method to initialize constants, counters, pre-plan trajectories, etc.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            planner: TBD
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
        
                information such as disturbance configurations, randomizations, etc.
            t_total: TBD ruff
        """

    @abstractmethod
    def control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Instructions:
            Implement this method to return the target state to be sent to the Crazyflie.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            A drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] in
            absolute coordinates or an attitude command [thrust, roll, pitch, yaw] as a numpy array.
        """

    @abstractmethod
    def set_ref_traj(self, ref_traj: dict):
        """TBD: for Ruff.

        Args:
            ref_traj: TBD.
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return

    def reset(self):
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return

    def set_tick(self, tick: int):
        """TBD: for Ruff.

        Args:
            tick: TBD for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return

    def get_states(self):
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return
    
    def get_setpoint(self) -> np.ndarray:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return
