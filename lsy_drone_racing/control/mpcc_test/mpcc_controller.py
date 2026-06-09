"""
mpcc_controller.py
------------------
Builds the acados OCP for the MPCC formulation and runs the receding-horizon
loop (Algorithm 1 of the paper). See mpcc_model.py for the state convention.

In deployment, [p,q,v,w] come from your estimator while [f, theta, vtheta] are
virtual states carried over from the previous prediction -- use
`feedback_state()` to assemble the next x0.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.mpcc_test.mpcc_model import (
    export_mpcc_model, MPCCConfig, NX, NU, NP, NY, NY_E, IDX_THETA, IDX_VTHETA,
)


class MPCCController:
    def __init__(self, cfg: MPCCConfig, reference,
                 json_file="acados_ocp_mpcc.json", build=True):
        self.cfg = cfg
        self.ref = reference
        self.N = cfg.N
        self.dt = cfg.dt
        self.mu = cfg.mu
        self.theta_pred = None
        self.solver = None
        if build:
            self._build(json_file)

    # ------------------------------------------------------------ build OCP
    def _build(self, json_file):
        cfg = self.cfg
        ocp = AcadosOcp()
        ocp.model = export_mpcc_model(cfg)

        # older acados: use `ocp.dims.N = self.N`
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(NY)            # weights are folded into the residual
        ocp.cost.W_e = np.eye(NY_E)
        ocp.cost.yref = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)
        ocp.parameter_values = np.zeros(NP)

        ocp.constraints.idxbx = np.array([10, 11, 12, 13, 14, 15, 16, 18])
        ocp.constraints.lbx = np.array([-cfg.w_max, -cfg.w_max, -cfg.w_max,
                                        cfg.T_min, cfg.T_min, cfg.T_min, cfg.T_min, 0.0])
        ocp.constraints.ubx = np.array([cfg.w_max, cfg.w_max, cfg.w_max,
                                        cfg.T_max, cfg.T_max, cfg.T_max, cfg.T_max, cfg.vtheta_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        ocp.constraints.lbu = np.array([-cfg.dvtheta_max,
                                        -cfg.df_max, -cfg.df_max, -cfg.df_max, -cfg.df_max])
        ocp.constraints.ubu = np.array([cfg.dvtheta_max,
                                        cfg.df_max, cfg.df_max, cfg.df_max, cfg.df_max])

        x0 = np.zeros(NX); x0[3] = 1.0
        ocp.constraints.x0 = x0

        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"   # PSD Hessian -> robust on this non-convex OCP
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4
        so.sim_method_num_steps = 2
        so.nlp_solver_type = "SQP_RTI"
        so.qp_solver_iter_max = 50
        so.levenberg_marquardt = 1e-2

        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)

    # ----------------------------------------------------- initial full state
    def initial_state(self, p, q, v, w, vtheta=1.0):
        x = np.zeros(NX)
        x[0:3] = p
        x[3:7] = self._norm_quat(q)
        x[7:10] = v
        x[10:13] = w
        x[13:17] = self.cfg.mass * self.cfg.g / 4.0
        x[IDX_THETA] = self.ref.project(np.asarray(p))
        x[IDX_VTHETA] = vtheta
        return x

    def feedback_state(self, p, q, v, w):
        x = np.zeros(NX)
        x[0:3] = p; x[3:7] = self._norm_quat(q); x[7:10] = v; x[10:13] = w
        x[13:17] = self._f_pred
        x[IDX_THETA] = self._theta_pred1
        x[IDX_VTHETA] = self._vtheta_pred1
        return x

    # --------------------------------------------------------------- warm start
    def _init_warmstart(self, x0):
        theta0 = float(x0[IDX_THETA])
        vth0 = float(x0[IDX_VTHETA]) if x0[IDX_VTHETA] > 1e-3 else 1.0
        self.theta_pred = np.array([theta0 + k * self.dt * vth0 for k in range(self.N + 1)])
        for k in range(self.N + 1):
            self.solver.set(k, "x", self._nominal_state(self.theta_pred[k], vth0))
        for k in range(self.N):
            self.solver.set(k, "u", np.zeros(NU))

    def _nominal_state(self, theta, vth):
        x = np.zeros(NX); x[3] = 1.0
        x[0:3] = self.ref.eval(theta)
        x[7:10] = vth * self.ref.tangent(theta)
        x[13:17] = self.cfg.mass * self.cfg.g / 4.0
        x[IDX_THETA] = theta
        x[IDX_VTHETA] = vth
        return x

    def _param_vector(self, theta):
        """Per-node runtime parameters (NP=12). Overridden by MPCC++."""
        pd = self.ref.eval(theta)
        td = self.ref.deriv1(theta)
        pdd = self.ref.deriv2(theta)
        qc = self.ref.qc(theta)
        return np.concatenate([pd, td, pdd, [theta], [qc], [self.mu]])

    # --------------------------------------------------------------- main solve
    def solve(self, x):
        x = np.array(x, dtype=float)
        x[3:7] = self._norm_quat(x[3:7])
        if self.theta_pred is None:
            self._init_warmstart(x)

        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        for k in range(self.N + 1):
            self.solver.set(k, "p", self._param_vector(float(self.theta_pred[k])))

        status = self.solver.solve()

        u0 = self.solver.get(0, "u")
        x1 = self.solver.get(1, "x")

        self._f_pred = x1[13:17].copy()
        self._theta_pred1 = float(x1[IDX_THETA])
        self._vtheta_pred1 = float(x1[IDX_VTHETA])

        sol_theta = np.array([self.solver.get(k, "x")[IDX_THETA] for k in range(self.N + 1)])
        self.theta_pred = np.concatenate([sol_theta[1:], [sol_theta[-1] + self.dt * self._vtheta_pred1]])

        return dict(
            status=status, u0=u0, x1=x1,
            omega_cmd=x1[10:13].copy(),
            collective_thrust=float(np.sum(x1[13:17])),
            theta=self._theta_pred1, vtheta=self._vtheta_pred1,
        )

    # --------------------------------------------------------------- utilities
    @staticmethod
    def _norm_quat(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
