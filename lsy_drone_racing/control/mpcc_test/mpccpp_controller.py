"""MPCC++ controller: plain MPCCController + the gate/track tunnel + soft obstacle avoidance.

Tunnel and obstacles are each HARD or SOFT independently
(MPCCppConfig.tunnel_soft / .obstacle_soft); soft uses acados slacks so the OCP
stays feasible.

Obstacles are runtime data, like the path:
  * the OCP is built for a fixed maximum of cfg.n_obstacles sticks;
  * set_obstacles() / set_obstacle() load the nominal positions at startup;
  * update_obstacle() overwrites a stick's position online (e.g. from
    perception) -- call it before solve() each step;
  * unused slots stay disabled (radius 0 -> the keep-out vanishes).

Only _build() and _param_vector() differ from the base controller; the warm
start, solve loop and command extraction are inherited. Requires a reference
with frame()/width(), e.g. TunnelReferencePath.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.mpcc_test.mpcc_controller import MPCCController
from lsy_drone_racing.control.mpcc_test.mpcc_model import NX, NY, NY_E
from lsy_drone_racing.control.mpcc_test.mpcc_reference import ReferencePath
from lsy_drone_racing.control.mpcc_test.mpccpp_model import (
    BNM,
    HIDX,
    N_TUNNEL,
    NRM,
    OBST_DIM,
    OBST_START,
    WIDX,
    MPCCppConfig,
    export_mpccpp_model,
    num_h,
    num_params,
)


class MPCCppController(MPCCController):
    """MPCC++ controller: adds a gate/track tunnel and soft stick-obstacle avoidance."""

    def __init__(
        self,
        cfg: MPCCppConfig,
        reference: ReferencePath,
        json_file: str = "acados_ocp_mpccpp.json",
        build: bool = True,
    ):
        """Set up obstacle state and build (or load) the acados OCP solver."""
        self._npar = num_params(cfg)
        self.n_obst = cfg.n_obstacles if cfg.use_obstacles else 0
        self._obstacles = np.zeros((self.n_obst, OBST_DIM))  # r=0 -> inactive
        super().__init__(cfg, reference, json_file=json_file, build=build)

    # ------------------------------------------------------------ build OCP
    def _build(self, json_file: str) -> None:
        """Build the acados OCP (cost, constraints, slacks) and instantiate the solver."""
        cfg = self.cfg
        ocp = AcadosOcp()
        ocp.model = export_mpccpp_model(cfg)

        # older acados: use ocp.dims.N = self.N
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(NY)
        ocp.cost.W_e = np.eye(NY_E)
        ocp.cost.yref = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)
        ocp.parameter_values = np.zeros(num_params(cfg))

        # box constraints on states / inputs (identical to plain MPCC)
        ocp.constraints.idxbx = np.array([10, 11, 12, 13, 14, 15, 16, 18])
        ocp.constraints.lbx = np.array(
            [-cfg.w_max, -cfg.w_max, -cfg.w_max, cfg.t_min, cfg.t_min, cfg.t_min, cfg.t_min, 0.0]
        )
        ocp.constraints.ubx = np.array(
            [
                cfg.w_max,
                cfg.w_max,
                cfg.w_max,
                cfg.t_max,
                cfg.t_max,
                cfg.t_max,
                cfg.t_max,
                cfg.vtheta_max,
            ]
        )
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        ocp.constraints.lbu = np.array(
            [-cfg.dvtheta_max, -cfg.df_max, -cfg.df_max, -cfg.df_max, -cfg.df_max]
        )
        ocp.constraints.ubu = np.array(
            [cfg.dvtheta_max, cfg.df_max, cfg.df_max, cfg.df_max, cfg.df_max]
        )

        x0 = np.zeros(NX)
        x0[3] = 1.0
        ocp.constraints.x0 = x0

        # --- nonlinear constraints: tunnel (4) then obstacles (n_obstacles) --
        nh = num_h(cfg)
        if nh > 0:
            big = 1e9
            ocp.constraints.lh = np.zeros(nh)  # h >= 0
            ocp.constraints.uh = big * np.ones(nh)
            ocp.constraints.lh_e = np.zeros(nh)
            ocp.constraints.uh_e = big * np.ones(nh)

            # soften each group independently; slack arrays align with idxsh
            soft_idx, zl, zl_quad = [], [], []
            off = 0
            if cfg.use_tunnel:
                if cfg.tunnel_soft:
                    soft_idx += list(range(off, off + N_TUNNEL))
                    zl += [cfg.tunnel_slack_lin] * N_TUNNEL
                    zl_quad += [cfg.tunnel_slack_quad] * N_TUNNEL
                off += N_TUNNEL
            if cfg.use_obstacles:
                if cfg.obstacle_soft:
                    soft_idx += list(range(off, off + cfg.n_obstacles))
                    zl += [cfg.obstacle_slack_lin] * cfg.n_obstacles
                    zl_quad += [cfg.obstacle_slack_quad] * cfg.n_obstacles
                off += cfg.n_obstacles

            if soft_idx:
                idx = np.array(soft_idx)
                zl = np.array(zl)
                zl_quad = np.array(zl_quad)
                ocp.constraints.idxsh = idx
                ocp.constraints.idxsh_e = idx
                ocp.cost.zl, ocp.cost.zu = zl, zl
                ocp.cost.Zl, ocp.cost.Zu = zl_quad, zl_quad
                ocp.cost.zl_e, ocp.cost.zu_e = zl, zl
                ocp.cost.Zl_e, ocp.cost.Zu_e = zl_quad, zl_quad

        # solver options (identical to plain MPCC)
        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4
        so.sim_method_num_steps = 2
        so.nlp_solver_type = "SQP_RTI"
        so.qp_solver_iter_max = 50
        so.levenberg_marquardt = 1e-2

        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)

    # --------------------------------------------------- obstacle management
    def set_obstacles(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """Load all stick centers [(x,y) or (x,y,z)] and keep-out radii.

        Fewer than n_obstacles is fine; the rest stay disabled (r=0).
        """
        centers = np.atleast_2d(np.asarray(centers, dtype=float))
        radii = np.atleast_1d(np.asarray(radii, dtype=float))
        m = len(centers)
        if m > self.n_obst:
            raise ValueError(
                f"{m} obstacles > n_obstacles={self.n_obst}; rebuild with a larger cfg.n_obstacles"
            )
        obs = np.zeros((self.n_obst, OBST_DIM))
        obs[:m, 0] = centers[:, 0]
        obs[:m, 1] = centers[:, 1]
        obs[:m, 2] = radii
        self._obstacles = obs

    def set_obstacle(self, center: np.ndarray, radius: float) -> None:
        """Convenience for the single-stick case."""
        self.set_obstacles([center], [radius])

    def update_obstacle(self, i: int, center: np.ndarray, radius: float | None = None) -> None:
        """Overwrite stick i online (e.g. from perception). Call before solve()."""
        self._obstacles[i, 0] = center[0]
        self._obstacles[i, 1] = center[1]
        if radius is not None:
            self._obstacles[i, 2] = radius

    # ------------------------------------------------- per-node parameters
    def _param_vector(self, theta: float) -> np.ndarray:
        """Build the parameter vector p for a single horizon node at path position theta."""
        pd = self.ref.eval(theta)
        td = self.ref.deriv1(theta)
        pdd = self.ref.deriv2(theta)
        qc = self.ref.qc(theta)
        n, b = self.ref.frame(theta)
        W, H = self.ref.width(theta)
        pvec = np.zeros(self._npar)
        pvec[0:3] = pd
        pvec[3:6] = td
        pvec[6:9] = pdd
        pvec[9] = theta
        pvec[10] = qc
        pvec[11] = self.mu
        pvec[NRM] = n
        pvec[BNM] = b
        pvec[WIDX] = W
        pvec[HIDX] = H
        if self.cfg.use_obstacles and self.n_obst > 0:
            pvec[OBST_START : OBST_START + OBST_DIM * self.n_obst] = self._obstacles.reshape(-1)
        return pvec
