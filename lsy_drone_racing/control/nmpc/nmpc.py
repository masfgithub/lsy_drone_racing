"""<For RUFF: Brief description of what this module does>."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller_interface import ControllerInterface

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


class NMPC(ControllerInterface):
    """MPC using the collective thrust and attitude interface."""

    def __init__(
        self,
        obs: EnvState_t,
        planner: dict,
        info: dict,
        config: dict,
        t_total: int,
        use_soft: bool = False,
        gate_weight: float = 1e6,
        obstacle_weight: float = 1e6,
    ):
        """Initialize the attitude controller.

        Args:
            obs:             Initial observation.
            planner:         Planner dict with waypoints.
            info:            Additional environment information.
            config:          Environment configuration.
            t_total:         Total time steps.
            use_soft:        If True, use soft constraints (penalty in cost).
                             If False, use hard constraints (con_h_expr).
            gate_weight:     Soft penalty weight for gate violations.
            obstacle_weight: Soft penalty weight for obstacle violations.
        """
        super().__init__(obs, planner, info, config, t_total)
        self._freq = config.env.freq
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._t_total = t_total
        self._use_soft = use_soft

        self.set_ref_traj(planner)
        self.drone_params = load_params("so_rpy", config.sim.drone_model)

        self._gates_information = {
            "total_length": 0.72,
            "total_height": 0.72,
            "hole_width": 0.25,
            "hole_height": 0.25,
            "thickness": 0.3,
            "margin": 0.05,
        }
        self._obstacles_information = {"d_min": 0.15, "total_height": 2.0}

        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)

        if use_soft:
            from lsy_drone_racing.control.nmpc.env_soft_constraints import (
                get_gate_objects,  # returns list[WedgeWindow]
                get_obstacle_objects,
                set_env_params,
            )
            from lsy_drone_racing.control.nmpc.nmpc_soft_setup import create_ocp_solver_soft

            self._get_gate_objects = get_gate_objects
            self._get_obstacle_objects = get_obstacle_objects
            self._set_env_params = set_env_params

            self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
            self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

            self._acados_ocp_solver, self._ocp, self._env = create_ocp_solver_soft(
                self._T_HORIZON,
                self._N,
                self.drone_params,
                self._gates,
                self._obstacles,
                gate_weight=gate_weight,
                obstacle_weight=obstacle_weight,
            )

        else:
            from lsy_drone_racing.control.nmpc.env_constraints import (
                get_gate_objects,
                get_obstacle_objects,
                set_env_params,
            )
            from lsy_drone_racing.control.nmpc.nmpc_setup import create_ocp_solver

            self._get_gate_objects = get_gate_objects
            self._get_obstacle_objects = get_obstacle_objects
            self._set_env_params = set_env_params

            self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
            self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

            self._acados_ocp_solver, self._ocp = create_ocp_solver(
                self._T_HORIZON, self._N, self.drone_params, self._gates, self._obstacles
            )
            self._env = None

        self._set_env_params(self._acados_ocp_solver, self._gates, self._obstacles, self._N)

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
        self._infeas_counter = 0

        self.set_initial_warm_start(pBLL=obs.pBLL, pos_ref=self._waypoints_pos[0])
        self._u_traj = np.array([self._acados_ocp_solver.get(k, "u") for k in range(self._N)])

    def set_initial_warm_start(self, pBLL: np.ndarray, pos_ref: np.ndarray):
        """Initialise solver with a straight-line trajectory."""
        x0 = np.concatenate((pBLL, np.zeros(3), np.zeros(3), np.zeros(3)))
        x_ref = np.concatenate((pos_ref, np.zeros(3), np.zeros(3), np.zeros(3)))
        x_init = np.linspace(x0, x_ref, self._N + 1)
        for k in range(self._N + 1):
            self.x_pred[k] = x_init[k]
        for k in range(self._N):
            self.u_pred[k] = np.zeros(self._nu)

    def control(self, obs: EnvState_t, info: dict | None = None, tick_offset: float = 0.0) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw."""
        i = min(self._tick-tick_offset, self._tick_max-tick_offset)
        if self._tick >= self._tick_max:
            self._finished = True

        # Initial state
        rpy = R.from_quat(obs.qBLB).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs.qBLB, obs.wBLL)
        x0 = np.concatenate((obs.pBLL, rpy, obs.vBLL, drpy))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)
        self._acados_ocp_solver.set(0, "x", x0)

        # Reference trajectory
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # For soft: yref has an extra penalty column (target = 0, already zero)
        if self._use_soft:
            yref_full = np.zeros((self._N, self._ny + 1))
            yref_full[:, : self._ny] = yref
        else:
            yref_full = yref

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "y_ref", yref_full[j])

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]

        if self._use_soft:
            yref_e_full = np.zeros(self._ny_e + 1)
            yref_e_full[: self._ny_e] = yref_e
        else:
            yref_e_full = yref_e

        self._acados_ocp_solver.set(self._N, "y_ref", yref_e_full)

        # Warm start
        for k in range(self._N):
            self._acados_ocp_solver.set(k, "u", self.u_pred[k])
        for k in range(1, self._N + 1):
            self._acados_ocp_solver.set(k, "x", self.x_pred[k])

        # Update environment parameters
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates = self._get_gate_objects(
            obs.pTLL_array, gates_quat_wxyz, self._gates_information
        )
        self._obstacles = self._get_obstacle_objects(obs.pOLL_array, self._obstacles_information)
        self._set_env_params(self._acados_ocp_solver, self._gates, self._obstacles, self._N)

        return self.solve(obs, i)

    def _extract_solution(self):
        for k in range(self._N + 1):
            self.x_pred[k] = self._acados_ocp_solver.get(k, "x")
        for k in range(self._N):
            self.u_pred[k] = self._acados_ocp_solver.get(k, "u")
        self._u_traj = self.u_pred.copy()

    def solve(self, obs: EnvState_t, i: int) -> np.ndarray:
        """Solve the OCP and return the control input."""
        STATUS_MEANINGS = {
            0: "SUCCESS",
            1: "NLP_ITERATION_MAXIMUM",
            2: "INFEASIBLE",
            3: "MINIMUM_STEP_SIZE",
            4: "QP_FAILURE",
            5: "READY",
        }

        status = self._acados_ocp_solver.solve()

        # ── Acceptable solutions ──────────────────────────────────────────────
        if status in (0, 1, 3):
            if status != 0:
                print(f"[MPC] {STATUS_MEANINGS[status]} — accepting with caution.")
            self._infeas_counter = 0
            self._extract_solution()
            return self._u_traj[0]

        # ── Unrecoverable — open-loop fallback ────────────────────────────────
        self._infeas_counter = min(self._infeas_counter + 1, self._N - 1)
        print(
            f"[MPC] {STATUS_MEANINGS.get(status, 'UNKNOWN')} — "
            f"infeasible for {self._infeas_counter} consecutive steps. "
            f"Returning stored u_traj[{self._infeas_counter}]."
        )
        return self._u_traj[self._infeas_counter]

    def set_ref_traj(self, planner_traj: dict):
        """Set reference trajectory from planner."""
        self._waypoints_pos = planner_traj.positions#planner_traj["waypoints_pos"]
        self._waypoints_vel = planner_traj.velocities#planner_traj["waypoints_vel"]
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

    def reset(self):
        """Reset tick counter."""
        self._tick = 0

    def set_tick(self, tick: int):
        """Set current tick."""
        self._tick = tick

    def get_states(self):
        """Return states (TBD)."""
        return

    def get_predicted_traj(self) -> np.ndarray:
        """Return predicted position trajectory for the whole horizon."""
        return np.array([self._acados_ocp_solver.get(k, "x")[:3] for k in range(self._N + 1)])