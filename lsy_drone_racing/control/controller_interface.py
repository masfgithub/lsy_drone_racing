"""Abstract base class defining the controller interface used by the racing pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


class ControllerInterface(ABC):
    """Base class for controller implementations."""

    def __init__(self, obs: EnvState_t, planner: dict, info: dict, config: dict, t_total: int):
        """Initialization of the controller.

        Instructions:
            The controller's constructor has access the initial observation `obs`, the a priori
            information contained in dictionary `info`, and the config of the race track. Use this
            method to initialize constants, counters, pre-plan trajectories, etc.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            planner: The initial planned trajectory, forwarded to `set_ref_traj()` to seed the
                controller's reference.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional

                information such as disturbance configurations, randomizations, etc.
            t_total: Total duration of the episode in seconds, used to size the controller's
                internal horizon/tick bookkeeping.
        """

    @abstractmethod
    def control(self, obs: EnvState_t, info: dict | None = None) -> NDArray[np.floating]:
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
        """Adopt a (re-)planned trajectory as the controller's tracking reference.

        Instructions:
            Implement this method to store the planner's output (e.g. position/velocity
            waypoints) in whatever internal representation the controller needs for
            tracking, and to reset any progress/theta bookkeeping tied to the old
            reference.

        Args:
            ref_traj: The planner's trajectory output (e.g. a Trajectory dataclass with
                `.positions`/`.velocities`, or a planner-specific dict).

        Returns:
            None.
        """
        return

    def reset(self):
        """Reset internal controller state for a new episode.

        Instructions:
            Implement this method to reset counters (e.g. the tick) and any other
            per-episode state so the controller can be reused across resets without
            being reconstructed.

        Returns:
            None.
        """
        return

    def set_tick(self, tick: int):
        """Set the controller's internal tick/step counter.

        Instructions:
            Implement this method to synchronize the controller's notion of elapsed
            steps with the environment's, e.g. after an external tick update.

        Args:
            tick: The current environment step count.

        Returns:
            None.
        """
        return

    def get_states(self):
        """Return the controller's current internal state, for logging/debugging.

        Instructions:
            Implement this method to expose whatever internal state (e.g. solver
            state, predicted commands) is useful for inspection outside the
            controller.

        Returns:
            The controller's internal state, in a controller-specific format.
        """
        return

    def get_predicted_traj(self):
        """Return the controller's predicted position trajectory over its horizon.

        Instructions:
            Implement this method to return the predicted (x, y, z) positions computed
            by the last `control()` call, e.g. for rendering or diagnostics.

        Returns:
            An array of predicted positions over the prediction horizon.
        """
        return

    def get_ref_traj(self):
        """Return the reference trajectory segment the controller is currently tracking.

        Instructions:
            Implement this method to return the slice of the reference trajectory
            (set via `set_ref_traj()`) relevant to the current tick, e.g. for
            rendering or diagnostics.

        Returns:
            An array of reference positions.
        """
        return
