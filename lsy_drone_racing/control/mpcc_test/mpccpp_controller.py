"""
mpccpp_controller.py
--------------------
MPCC++ controller: subclasses the plain MPCCController and adds the gate/track
tunnel as a HARD or SOFT constraint (MPCCppConfig.tunnel_soft). Soft uses the
acados slack-variable mechanism -- the systematic soft-constraint approach the
paper references for real-time numerical stability, and the reason the OCP
stays feasible even when the drone grazes the tunnel walls. Set
use_tunnel=False to recover plain MPCC.

Only two things change versus the base controller:
  * _build(): export the MPCC++ model and wire up the tunnel constraint;
  * _param_vector(): append the tunnel frame (n, b) and half-extents (W, H).
Everything else (warm start, solve loop, command extraction) is inherited.

Requires a reference with frame()/width(), e.g. TunnelReferencePath.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.mpcc_test.mpcc_controller import MPCCController
from lsy_drone_racing.control.mpcc_test.mpcc_model import NX, NU, NY, NY_E, IDX_THETA, IDX_VTHETA
from lsy_drone_racing.control.mpcc_test.mpccpp_model import (
    export_mpccpp_model, MPCCppConfig, NP_PP, N_TUNNEL, NRM, BNM, WIDX, HIDX,
)


class MPCCppController(MPCCController):

    # ------------------------------------------------------------ build OCP
    def _build(self, json_file):
        cfg = self.cfg
        ocp = AcadosOcp()
        ocp.model = export_mpccpp_model(cfg)

        # older acados: use `ocp.dims.N = self.N`
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(NY)
        ocp.cost.W_e = np.eye(NY_E)
        ocp.cost.yref = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)
        ocp.parameter_values = np.zeros(NP_PP)

        # box constraints on states / inputs (identical to plain MPCC)
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

        # --- gate/track tunnel: hard box on h, or softened via slacks --------
        if cfg.use_tunnel:
            nh = N_TUNNEL
            big = 1e9
            ocp.constraints.lh = np.zeros(nh)          # h >= 0 (inside prism)
            ocp.constraints.uh = big * np.ones(nh)
            ocp.constraints.lh_e = np.zeros(nh)
            ocp.constraints.uh_e = big * np.ones(nh)
            if cfg.tunnel_soft:
                idx = np.arange(nh)
                ocp.constraints.idxsh = idx
                ocp.constraints.idxsh_e = idx
                zl = cfg.tunnel_slack_lin * np.ones(nh)
                Zl = cfg.tunnel_slack_quad * np.ones(nh)
                ocp.cost.zl, ocp.cost.zu = zl, zl
                ocp.cost.Zl, ocp.cost.Zu = Zl, Zl
                ocp.cost.zl_e, ocp.cost.zu_e = zl, zl
                ocp.cost.Zl_e, ocp.cost.Zu_e = Zl, Zl

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

    # ------------------------------------------------- per-node parameters
    def _param_vector(self, theta):
        pd = self.ref.eval(theta)
        td = self.ref.deriv1(theta)
        pdd = self.ref.deriv2(theta)
        qc = self.ref.qc(theta)
        n, b = self.ref.frame(theta)
        W, H = self.ref.width(theta)
        pvec = np.zeros(NP_PP)
        pvec[0:3] = pd; pvec[3:6] = td; pvec[6:9] = pdd
        pvec[9] = theta; pvec[10] = qc; pvec[11] = self.mu
        pvec[NRM] = n; pvec[BNM] = b; pvec[WIDX] = W; pvec[HIDX] = H
        return pvec
