"""<For RUFF: Brief description of what this module does>."""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller_interface import ControllerInterface
from lsy_drone_racing.control.nmpc.env_constraints import (
    get_gate_objects,
    get_obstacle_objects,
    set_env_params,
)
from lsy_drone_racing.control.nmpc.nmpc_setup import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


class NMPC(ControllerInterface):
    """MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: EnvState_t, planner: dict, info: dict, config: dict, t_total: int):
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

        self._gates_information = {
            "total_length": 0.72,  # outer frame width  [m] — square gate, outer dim
            "total_height": 0.72,  # outer frame height [m] — square gate, outer dim
            "hole_width": 0.25,  # opening width      [m]
            "hole_height": 0.25,  # opening height     [m]
            "thickness": 0.10,  # frame depth        [m] — not in TOML, physical estimate
            "margin": 0.05,  # constraint margin  [m] — Window class default
        }

        self._obstacles_information = {"d_min": 0.1, "total_height": 2.0}

        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, self._gates, self._obstacles
        )
        set_env_params(self._acados_ocp_solver, self._gates, self._obstacles, self._N)

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self.x_pred = np.zeros((self._N + 1, self._nx))
        self.u_pred = np.ones((self._N, self._nu))

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

        self.set_initial_warm_start(pBLL=obs.pBLL, pos_ref=self._waypoints_pos[0])
        self._u_traj = np.array([self._acados_ocp_solver.get(k, "u") for k in range(self._N)])
        self._infeas_counter = 0

    def set_initial_warm_start(self, pBLL: np.ndarray, pos_ref: np.ndarray):
        """TBD: for Ruff."""
        x0 = np.concatenate((pBLL, np.zeros(3), np.zeros(3), np.zeros(3)))
        x_ref = np.concatenate((pos_ref, np.zeros(3), np.zeros(3), np.zeros(3)))
        x_init = np.linspace(x0, x_ref, self._N + 1)
        for k in range(self._N + 1):
            self.x_pred[k] = x_init[k]
        for k in range(self._N):
            self.u_pred[k] = np.zeros(self._nu)

    def control(self, obs: EnvState_t, info: dict | None = None) -> NDArray[np.floating]:
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
        rpy = R.from_quat(obs.qBLB).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs.qBLB, obs.wBLL)
        x0 = np.concatenate((obs.pBLL, rpy, obs.vBLL, drpy))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)
        self._acados_ocp_solver.set(0, "x", x0)

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
            self._acados_ocp_solver.set(j, "y_ref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Warm start     
        for k in range(self._N):
            self._acados_ocp_solver.set(k, "u", self.u_pred[k])

        for k in range(1, self._N+1):
            self._acados_ocp_solver.set(k, "x", self.x_pred[k])   

        # Update environment parameters
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)
        set_env_params(self._acados_ocp_solver, self._gates, self._obstacles, self._N)

        # Solve the OCP
        u = self.solve(obs, i)
        return u

    def solve(self, obs: EnvState_t, i: int) -> np.ndarray:
        """Solve the OCP and return the control input.

        Handles solver failures gracefully:
        - SUCCESS (0):           extract fresh u_traj/x_pred/u_pred, return u_traj[0]
        - NLP_ITER_MAX (1):      accept solution (often still usable)
        - MINIMUM_STEP_SIZE (3): accept solution (converged to local min)
        - QP_FAILURE (4):        reset warm-start, retry once, then fall back
        - INFEASIBLE (2):        fall back to stored trajectory with offset

        Args:
            obs: Observation object containing obs.pBLL (current position).
            i:   Current waypoint index.

        Returns:
            Control input np.ndarray of shape (n_u,).
        """
        STATUS_MEANINGS = {
            0: "SUCCESS",
            1: "NLP_ITERATION_MAXIMUM",
            2: "INFEASIBLE",
            3: "MINIMUM_STEP_SIZE",
            4: "QP_FAILURE",
            5: "READY",
        }

        def _extract_solution(self):
            """Pull x_pred, u_pred and u_traj from the solver."""
            for k in range(self._N + 1):
                self.x_pred[k] = self._acados_ocp_solver.get(k, "x")
            for k in range(self._N):
                self.u_pred[k] = self._acados_ocp_solver.get(k, "u")
            self._u_traj = self.u_pred.copy()

        status = self._acados_ocp_solver.solve()

        # ── SUCCESS ───────────────────────────────────────────────────────────────
        if status == 0:
            self._infeas_counter = 0
            _extract_solution(self)
            return self._u_traj[0]

        # ── NLP_ITERATION_MAXIMUM or MINIMUM_STEP_SIZE ────────────────────────────
        if status in (1, 3):
            print(
                f"[MPC] Solver status: {STATUS_MEANINGS[status]} — "
                f"accepting solution with caution."
            )
            self._infeas_counter = 0
            _extract_solution(self)
            return self._u_traj[0]

        # ── QP_FAILURE ────────────────────────────────────────────────────────────
        if status == 4:
            print("[MPC] QP failure — resetting warm-start and retrying.")
            pos_reset_idx = min(i + self._N, len(self._waypoints_pos) - 1)
            pos_reset     = self._waypoints_pos[pos_reset_idx]
            self.set_initial_warm_start(obs.pBLL, pos_reset)
            print(f'N+1: {i + self._N}, waypoints: {len(self._waypoints_pos)}, point: {pos_reset}')
            
            retry_status = self._acados_ocp_solver.solve()
            if retry_status == 0:
                print("[MPC] Retry succeeded.")
                self._infeas_counter = 0
                _extract_solution(self)
                return self._u_traj[0]
            else:
                print(
                    f"[MPC] Retry failed: "
                    f"{STATUS_MEANINGS.get(retry_status, 'UNKNOWN')} — "
                    f"falling back to stored trajectory."
                )

        # ── INFEASIBLE or unrecovered failure — open-loop fallback ────────────────
        self._infeas_counter = min(self._infeas_counter + 1, self._N - 1)
        print(
            f"[MPC] {STATUS_MEANINGS.get(status, 'UNKNOWN')} — "
            f"infeasible for {self._infeas_counter} consecutive steps. "
            f"Returning stored u_traj[{self._infeas_counter}]."
        )
        return self._u_traj[self._infeas_counter]


    def set_ref_traj(self, planner_traj: dict):
        """TBD: for Ruff.

        Args:
            planner_traj: TBD.

        Returns:
            TBD: for Ruff.
        """
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

    def get_predicted_traj(self) -> np.ndarray:
        """Return the predicted position trajectory for the whole horizon.

        Returns:
            Array of shape (N+1, 3) — predicted XYZ positions from k=0 to k=N.
        """
        return np.array([self._acados_ocp_solver.get(k, "x")[:3] for k in range(self._N + 1)])
