"""
mpccpp_model.py
---------------
MPCC++ model: the plain MPCC model plus the prismatic gate/track tunnel
constraint (Krinner et al., RSS 2024, eq.(7)).

Dynamics and the least-squares cost residual are reused unchanged from
mpcc_model.py. The only additions are:
  * 8 extra runtime parameters per node (frame n, b and half-extents W, H),
  * the four tunnel halfspace constraints exposed as con_h_expr.

Tunnel constraint, with centerline point pd, frame (n, b), half-extents (W, H),
and platform position p:  d = p - pd
        W + d.n >= 0,   W - d.n >= 0,    H + d.b >= 0,   H - d.b >= 0
i.e. the platform must stay inside the prism. The frame / extents are evaluated
at the predicted progress theta_bar (re-linearized each RTI step), so the four
halfspaces are LINEAR in the position -- ideal for the QP. Whether they are
enforced as HARD or SOFT (slack-variable) constraints is decided in the
controller via MPCCppConfig.tunnel_soft.

Parameters (NP_PP = 20) per node:
    p = [ pd(3), td(3), pdd(3), theta_bar(1), qc(1), mu(1),   # first 12 = MPCC
          n(3), b(3), W(1), H(1) ]                            # tunnel
"""

from dataclasses import dataclass
import casadi as ca
from acados_template import AcadosModel

from lsy_drone_racing.control.mpcc_test.mpcc_model import (
    MPCCConfig, quadrotor_dynamics, mpcc_residual, NX, NU, IDX_P,
)

# extended parameter vector
NP_PP = 20
N_TUNNEL = 4
NRM = slice(12, 15)
BNM = slice(15, 18)
WIDX = 18
HIDX = 19


@dataclass
class MPCCppConfig(MPCCConfig):
    use_tunnel: bool = True             # add the gate/track tunnel at all
    tunnel_soft: bool = True            # True: soft (acados slacks); False: hard
    tunnel_slack_lin: float = 1e3       # L1 slack penalty  (zl, zu)
    tunnel_slack_quad: float = 1e3      # L2 slack penalty  (Zl, Zu)


def tunnel_constraint(x, p):
    """Four halfspaces of the MPCC++ prism, eq.(7); all >= 0 when inside."""
    pos = x[IDX_P]
    pd = p[0:3]; n = p[NRM]; b = p[BNM]; W = p[WIDX]; H = p[HIDX]
    d = pos - pd
    dn = ca.dot(d, n)
    db = ca.dot(d, b)
    return ca.vertcat(W + dn, W - dn, H + db, H - db)


def export_mpccpp_model(cfg: MPCCppConfig) -> AcadosModel:
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP_PP)
    xdot = ca.SX.sym("xdot", NX)

    model = AcadosModel()
    model.name = "quadrotor_mpccpp"
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.f_expl_expr = quadrotor_dynamics(x, u, cfg)
    model.f_impl_expr = xdot - model.f_expl_expr
    # residual reads only p[0:12] -> reused unchanged (Gauss-Newton)
    model.cost_y_expr = mpcc_residual(x, u, p, cfg, terminal=False)
    model.cost_y_expr_e = mpcc_residual(x, u, p, cfg, terminal=True)

    if cfg.use_tunnel:
        h = tunnel_constraint(x, p)
        model.con_h_expr = h            # stages 1 .. N-1
        model.con_h_expr_e = h          # terminal stage N (stage 0 left free)
    return model
