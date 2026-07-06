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

    from lsy_drone_racing.control.env_obs import EnvState


class NMPC(ControllerInterface):
    """MPC using the collective thrust and attitude interface."""

    def __init__(
        self,
        obs: EnvState,
        planner: dict,
        info: dict,
        config: dict,
        t_total: int,
        use_soft: bool = False,
        gate_weight: float = 1e4,
        obstacle_weight: float = 1e4,
        post_weight: float = 1e4,
        use_input_rate: bool = True,
        df_cmd_rate_max: float | None = 5.0,
        dr_cmd_rate_max: float | None = None,
        dp_cmd_rate_max: float | None = None,
        dy_cmd_rate_max: float | None = None,
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
            post_weight:     Soft penalty weight for gate-post violations.
            use_input_rate:  If True, use the rate-augmented model: the commands
                             [r_cmd, p_cmd, y_cmd, f_cmd] become states (16-state
                             model) and the inputs become their rates, so the input
                             box acts as a per-command slew-rate limit. False
                             keeps the baseline 12-state model bit-identical.
            df_cmd_rate_max: Per-command slew limit for thrust (N/s); only used
                             when use_input_rate=True. Finite value activates;
                             None => inactive (wide default).
            dr_cmd_rate_max: Per-command slew limit for roll (rad/s); see df_cmd_rate_max.
            dp_cmd_rate_max: Per-command slew limit for pitch (rad/s); see df_cmd_rate_max.
            dy_cmd_rate_max: Per-command slew limit for yaw (rad/s); see df_cmd_rate_max.
        """
        super().__init__(obs, planner, info, config, t_total)
        self._freq = config.env.freq
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._t_total = t_total
        self._use_soft = use_soft
        self._use_input_rate = use_input_rate

        self.set_ref_traj(planner)
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._hover = float(self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1])
        # Last applied command [r_cmd, p_cmd, y_cmd, f_cmd]; pins the command states
        # at node 0 each step (only used when use_input_rate=True). Bootstrapped to
        # upright + hover.
        self._last_u_cmd = np.array([0.0, 0.0, 0.0, self._hover])

        self._gates_information = {
            "total_length": 0.8,
            "total_height": 0.8,
            "hole_width": 0.23,
            "hole_height": 0.23,
            "thickness": 0.35,
            "margin": 0.05,
        }
        self._obstacles_information = {"d_min": 0.15, "total_height": 2.0}

        gates_quat_wxyz = np.roll(obs.q_tlt_array, 1, axis=-1)

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

            self._gates = get_gate_objects(
                obs.p_tll_array, gates_quat_wxyz, self._gates_information
            )
            self._obstacles = get_obstacle_objects(obs.p_oll_array, self._obstacles_information)

            self._acados_ocp_solver, self._ocp, self._env = create_ocp_solver_soft(
                self._T_HORIZON,
                self._N,
                self.drone_params,
                self._gates,
                self._obstacles,
                gate_weight=gate_weight,
                obstacle_weight=obstacle_weight,
                post_weight=post_weight,
                use_input_rate=self._use_input_rate,
                df_cmd_rate_max=df_cmd_rate_max,
                dr_cmd_rate_max=dr_cmd_rate_max,
                dp_cmd_rate_max=dp_cmd_rate_max,
                dy_cmd_rate_max=dy_cmd_rate_max,
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

            self._gates = get_gate_objects(
                obs.p_tll_array, gates_quat_wxyz, self._gates_information
            )
            self._obstacles = get_obstacle_objects(obs.p_oll_array, self._obstacles_information)

            self._acados_ocp_solver, self._ocp = create_ocp_solver(
                self._T_HORIZON,
                self._N,
                self.drone_params,
                self._gates,
                self._obstacles,
                use_input_rate=self._use_input_rate,
                df_cmd_rate_max=df_cmd_rate_max,
                dr_cmd_rate_max=dr_cmd_rate_max,
                dp_cmd_rate_max=dp_cmd_rate_max,
                dy_cmd_rate_max=dy_cmd_rate_max,
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

        self.set_initial_warm_start(p_bll=obs.p_bll, pos_ref=self._waypoints_pos[0])
        self._u_traj = np.array([self._acados_ocp_solver.get(k, "u") for k in range(self._N)])

    def set_initial_warm_start(self, p_bll: np.ndarray, pos_ref: np.ndarray):
        """Initialise solver with a straight-line trajectory."""
        x0 = np.concatenate((p_bll, np.zeros(3), np.zeros(3), np.zeros(3)))
        x_ref = np.concatenate((pos_ref, np.zeros(3), np.zeros(3), np.zeros(3)))
        if self._use_input_rate:
            # Append the command-state block [r_cmd, p_cmd, y_cmd, f_cmd] = upright+hover.
            cmd0 = np.array([0.0, 0.0, 0.0, self._hover])
            x0 = np.concatenate((x0, cmd0))
            x_ref = np.concatenate((x_ref, cmd0))
        x_init = np.linspace(x0, x_ref, self._N + 1)
        for k in range(self._N + 1):
            self.x_pred[k] = x_init[k]
        for k in range(self._N):
            self.u_pred[k] = np.zeros(self._nu)

    def control(
        self, obs: EnvState, info: dict | None = None, tick_offset: float = 0.0
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw."""
        i = min(self._tick - tick_offset, self._tick_max - tick_offset)
        if self._tick >= self._tick_max:
            self._finished = True

        # Initial state
        rpy = R.from_quat(obs.q_blb).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs.q_blb, obs.w_bll)
        x0 = np.concatenate((obs.p_bll, rpy, obs.v_bll, drpy))
        if self._use_input_rate:
            # Pin the command states at node 0 to the last applied command, which
            # makes the rate box a slew limit across the MPC-step boundary too.
            x0 = np.concatenate((x0, self._last_u_cmd))
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
        if self._use_input_rate:
            # Terminal f_cmd command-state reference = hover (index 15).
            yref_e[15] = self._hover

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
        gates_quat_wxyz = np.roll(obs.q_tlt_array, 1, axis=-1)
        self._gates = self._get_gate_objects(
            obs.p_tll_array, gates_quat_wxyz, self._gates_information
        )
        self._obstacles = self._get_obstacle_objects(obs.p_oll_array, self._obstacles_information)
        self._set_env_params(self._acados_ocp_solver, self._gates, self._obstacles, self._N)

        return self.solve(obs, i)

    def _extract_solution(self):
        for k in range(self._N + 1):
            self.x_pred[k] = self._acados_ocp_solver.get(k, "x")
        for k in range(self._N):
            self.u_pred[k] = self._acados_ocp_solver.get(k, "u")
        self._u_traj = self.u_pred.copy()

    def _applied_command(self, step: int) -> np.ndarray:
        """Command [r_cmd, p_cmd, y_cmd, f_cmd] to apply at horizon offset ``step``.

        Baseline: the inputs ARE the commands -> u_pred[step].
        Augmented: the commands are states -> read the command block of the
        predicted state one node ahead (x_pred[1+step][12:16]).
        """
        if self._use_input_rate:
            k = min(1 + step, self._N)
            return self.x_pred[k][12:16].copy()
        return self.u_pred[step].copy()

    def solve(self, obs: EnvState, i: int) -> np.ndarray:
        """Solve the OCP and return the control input."""
        status_meanings = {
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
                print(f"[MPC] {status_meanings[status]} — accepting with caution.")
            self._infeas_counter = 0
            self._extract_solution()
            applied = self._applied_command(0)
            if self._use_input_rate:
                self._last_u_cmd = np.asarray(applied, dtype=float).copy()
            return applied

        # ── Unrecoverable — open-loop fallback ────────────────────────────────
        self._infeas_counter = min(self._infeas_counter + 1, self._N - 1)
        print(
            f"[MPC] {status_meanings.get(status, 'UNKNOWN')} — "
            f"infeasible for {self._infeas_counter} consecutive steps. "
            f"Returning stored command[{self._infeas_counter}]."
        )
        applied = self._applied_command(self._infeas_counter)
        if self._use_input_rate:
            self._last_u_cmd = np.asarray(applied, dtype=float).copy()
        return applied

    def set_ref_traj(self, planner_traj: dict):
        """Set reference trajectory from planner."""
        self._waypoints_pos = planner_traj.positions  # planner_traj["waypoints_pos"]
        self._waypoints_vel = planner_traj.velocities  # planner_traj["waypoints_vel"]
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

    def reset(self):
        """Reset tick counter."""
        self._tick = 0

    def set_tick(self, tick: int):
        """Set current tick."""
        self._tick = tick

    def get_states(self):
        """Unused; kept for interface compatibility."""
        return

    def get_predicted_traj(self) -> np.ndarray:
        """Return predicted position trajectory for the whole horizon."""
        return np.array([self._acados_ocp_solver.get(k, "x")[:3] for k in range(self._N + 1)])

    def get_ref_traj(self, i: int | None = None) -> np.ndarray:
        """Return the reference trajectory sample at tick i (defaults to the current tick)."""
        if i is None:
            i = self._tick
        i = int(min(i, self._tick_max))
        return self._waypoints_pos[i : i + self._N + 1]
