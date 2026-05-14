"""This module implements the pipeline for the drone racing.

TBD specify more in detail.
"""
from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from crazyflow import Sim
    from numpy.typing import NDArray

from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control.basic_planner import BasicPlanner
from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.nmpc.nmpc import NMPC


class DroneRacingPipeline(Controller):
    """This class handles the pipeline for the drone racing. It includes planning and control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the pipeline.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        
        # variable setup
        t_total = 8
        self._tick = 0
        self._finished = False

        # setup for planner
        self._planner = BasicPlanner(config, t_total)
        planner_dict = self._planner.plan()
        
        # setup for controller
        self._controller = NMPC(obs, planner_dict, info, config, t_total)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        self._planner.replan()
        u0 = self._controller.control(obs, info)
        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        self._controller.set_tick(self._tick)
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
        self._controller.set_tick(self._tick)

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._controller.get_setpoint().reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._planner.get_pos_traj()
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
