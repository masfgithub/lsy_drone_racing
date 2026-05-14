"""<For RUFF: Brief description of what this module does>."""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller_interface import ControllerInterface
from lsy_drone_racing.control.nmpc.nmpc_setup import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NMPC(ControllerInterface):
    """MPC using the collective thrust and attitude interface."""

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        planner: dict,
        info: dict,
        config: dict,
        t_total: int,
    ):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            planner: TBD
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            t_total: TBD ruff
        """
        super().__init__(obs, planner, info, config, t_total)
        self._freq = config.env.freq
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self._t_total = t_total  # s
        self.set_ref_traj(planner)

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def control(
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
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        # zero roll, pitch
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def set_ref_traj(self, planner_traj: dict):
        """TBD: for Ruff.

        Args:
            planner_traj: TBD.

        Returns:
            TBD: for Ruff.
        """
        self._des_pos_spline = planner_traj["des_pos_spline"]
        self._des_vel_spline = planner_traj["des_vel_spline"]
        self._waypoints_pos = planner_traj["waypoints_pos"]
        self._waypoints_vel = planner_traj["waypoints_vel"]
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

    def reset(self):
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        self._tick = 0

    def set_tick(self, tick: int):
        """TBD: for Ruff.

        Args:
            tick: TBD for Ruff.

        Returns:
            TBD: for Ruff.
        """
        self._tick = tick

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
        return self._des_pos_spline(self._tick / self._freq)
