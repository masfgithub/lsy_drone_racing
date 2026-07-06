"""acados model for Model Predictive Contouring Control (MPCC) of a quadrotor.

Romero, Sun, Foehn, Scaramuzza, IEEE T-RO 2022.

State (NX = 19), eq.(15) augmented with progress dynamics:
    x = [ p(3), q(4), v(3), w(3), f(4), theta(1), vtheta(1) ]
Input (NU = 5), eq.(15)/(16):
    u = [ dvtheta(1), df(4) ]
Dynamics: eq.(12)+(14)+(16).

Cost: the MPCC cost (eq.17) is implemented as a NONLINEAR least-squares
residual so the solver can use a GAUSS-NEWTON Hessian. GN is always positive
semidefinite, which is essential here: the OCP is non-convex (Sec. VIII-A), so
an EXACT Hessian produces indefinite / NaN QP data and HPIPM fails.

Per-node weights qc (contour) and mu (progress) vary along the path, so they are
FOLDED INTO the residual (multiplied as sqrt) rather than living in the static
weight matrix W; they are passed as runtime parameters.

NOTE on the progress term: the paper maximizes progress with a linear reward
-mu*vtheta, which is not a least-squares term. The GN-compatible equivalent
used here drives vtheta toward its cap via the residual sqrt(mu)*(vtheta -
vtheta_max). The effect is the same trade-off the paper describes -- large mu
=> go fast (hug the speed cap, accept contour error), small mu => track tightly
-- with mu and vtheta_max as the two knobs.

Residual layout (NY = 13 stage, NY_E = 8 terminal):
    stage:    [ sqrt(qc)*e_c(3), sqrt(ql)*e_l(1), sqrt(Qw)*w(3),
                sqrt(mu)*(vtheta-vtheta_max)(1),
                sqrt(r_dv)*dvtheta(1), sqrt(Rdf)*df(4) ]
    terminal: the first 8 entries (no input terms).

Parameters (NP = 12) per node:
    p = [ pd(3), td(3), pdd(3), theta_bar(1), qc(1), mu(1) ]
"""

from dataclasses import dataclass

import casadi as ca
from acados_template import AcadosModel

# ---- layout ----------------------------------------------------------------
NX, NU, NP = 19, 5, 12
NY, NY_E = 13, 8
IDX_P = slice(0, 3)
IDX_Q = slice(3, 7)
IDX_V = slice(7, 10)
IDX_W = slice(10, 13)
IDX_F = slice(13, 17)
IDX_THETA = 17
IDX_VTHETA = 18


@dataclass
class MPCCConfig:
    """Physical parameters, actuator limits, horizon, and cost weights for the MPCC model."""

    # --- physical parameters (Table I, "RPG Quad") --------------------------
    mass: float = 0.85
    arm: float = 0.15
    inertia: tuple = (2.5e-3, 2.1e-3, 4.3e-3)
    c_tau: float = 0.022
    drag: tuple = (0.0, 0.0, 0.0)  # identify per platform
    g: float = 9.81

    # --- limits (Table I) ---------------------------------------------------
    t_min: float = 0.0
    t_max: float = 7.0
    w_max: float = 10.0
    vtheta_max: float = 30.0  # progress speed cap = progress target
    dvtheta_max: float = 60.0
    df_max: float = 200.0

    # --- horizon ------------------------------------------------------------
    N: int = 20
    dt: float = 0.05

    # --- cost weights (qc and mu are runtime params, see controller) --------
    q_lag: float = 1000.0
    q_w: tuple = (1.0, 1.0, 1.0)
    r_dvtheta: float = 1e-2
    r_df: tuple = (1e-3, 1e-3, 1e-3, 1e-3)
    mu: float = 1.0  # progress weight (quadratic; tune)


# ---------------------------------------------------------------------------
def _quat_to_rot(q: ca.SX) -> ca.SX:
    """Rotation matrix from a (normalized) [w,x,y,z] quaternion."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    n = ca.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + 1e-12)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return ca.vertcat(
        ca.horzcat(1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)),
        ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)),
    )


def quadrotor_dynamics(x: ca.SX, u: ca.SX, cfg: MPCCConfig) -> ca.SX:
    """Continuous-time quadrotor + progress dynamics, eq.(12)+(14)+(16)."""
    q = x[IDX_Q]
    v = x[IDX_V]
    w = x[IDX_W]
    f = x[IDX_F]
    vtheta = x[IDX_VTHETA]
    dvtheta = u[0]
    df = u[1:5]

    m, arm_len, ctau = cfg.mass, cfg.arm, cfg.c_tau
    Jd = ca.DM(list(cfg.inertia))
    R = _quat_to_rot(q)

    f_total = f[0] + f[1] + f[2] + f[3]
    fT_body = ca.vertcat(0, 0, f_total)
    gvec = ca.vertcat(0, 0, -cfg.g)
    D = ca.diag(ca.DM(list(cfg.drag)))
    v_dot = gvec + (R @ fT_body) / m - R @ (D @ (R.T @ v))

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = w[0], w[1], w[2]
    q_dot = 0.5 * ca.vertcat(
        -qx * wx - qy * wy - qz * wz,
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
    )

    tau = ca.vertcat(
        arm_len / ca.sqrt(2.0) * (f[0] + f[1] - f[2] - f[3]),
        arm_len / ca.sqrt(2.0) * (-f[0] + f[1] + f[2] - f[3]),
        ctau * (f[0] - f[1] + f[2] - f[3]),
    )
    Jw = ca.vertcat(Jd[0] * wx, Jd[1] * wy, Jd[2] * wz)
    w_cross_Jw = ca.vertcat(
        wy * Jw[2] - wz * Jw[1], wz * Jw[0] - wx * Jw[2], wx * Jw[1] - wy * Jw[0]
    )
    w_dot = ca.vertcat(
        (tau[0] - w_cross_Jw[0]) / Jd[0],
        (tau[1] - w_cross_Jw[1]) / Jd[1],
        (tau[2] - w_cross_Jw[2]) / Jd[2],
    )

    return ca.vertcat(v, q_dot, v_dot, w_dot, df, vtheta, dvtheta)


def mpcc_residual(x: ca.SX, u: ca.SX, p: ca.SX, cfg: MPCCConfig, terminal: bool = False) -> ca.SX:
    """Least-squares residual for the MPCC cost (eq.17).

    Reads only p[0:12], so it is reused unchanged by the MPCC++ model (which
    appends tunnel params).
    """
    pos = x[IDX_P]
    w = x[IDX_W]
    theta = x[IDX_THETA]
    vtheta = x[IDX_VTHETA]

    pd = p[0:3]
    td = p[3:6]
    pdd = p[6:9]
    theta_bar = p[9]
    qc = p[10]
    mu = p[11]

    s = theta - theta_bar
    pd_theta = pd + td * s + 0.5 * pdd * s * s
    t_raw = td + pdd * s
    t_hat = t_raw / ca.sqrt(ca.sumsqr(t_raw) + 1e-9)

    e = pos - pd_theta
    e_l = ca.dot(t_hat, e)
    e_c = e - e_l * t_hat

    Qw = ca.DM(list(cfg.q_w))
    y = ca.vertcat(
        ca.sqrt(qc + 1e-9) * e_c,  # contour (3)
        ca.sqrt(cfg.q_lag) * e_l,  # lag (1)
        ca.sqrt(Qw[0]) * w[0],  # body-rate reg (3)
        ca.sqrt(Qw[1]) * w[1],
        ca.sqrt(Qw[2]) * w[2],
        ca.sqrt(mu + 1e-9) * (vtheta - cfg.vtheta_max),  # progress (1)
    )
    if not terminal:
        dvtheta = u[0]
        df = u[1:5]
        Rdf = ca.DM(list(cfg.r_df))
        y = ca.vertcat(
            y,
            ca.sqrt(cfg.r_dvtheta) * dvtheta,  # input reg (1)
            ca.sqrt(Rdf[0]) * df[0],  # thrust-rate reg (4)
            ca.sqrt(Rdf[1]) * df[1],
            ca.sqrt(Rdf[2]) * df[2],
            ca.sqrt(Rdf[3]) * df[3],
        )
    return y


def export_mpcc_model(cfg: MPCCConfig) -> AcadosModel:
    """Build the acados model for the MPCC quadrotor: dynamics and cost."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    xdot = ca.SX.sym("xdot", NX)

    model = AcadosModel()
    model.name = "quadrotor_mpcc"
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.f_expl_expr = quadrotor_dynamics(x, u, cfg)
    model.f_impl_expr = xdot - model.f_expl_expr
    model.cost_y_expr = mpcc_residual(x, u, p, cfg, terminal=False)
    model.cost_y_expr_e = mpcc_residual(x, u, p, cfg, terminal=True)
    return model


def make_dynamics_fn(cfg: MPCCConfig) -> ca.Function:
    """Build a CasADi function evaluating the continuous-time dynamics f(x, u)."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    return ca.Function("f_dyn", [x, u], [quadrotor_dynamics(x, u, cfg)])
