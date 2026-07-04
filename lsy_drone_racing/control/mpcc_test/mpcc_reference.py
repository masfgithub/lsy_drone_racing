"""mpcc_reference.py
-----------------
Arc-length-parameterized 3D reference path for MPCC (Romero et al. 2022).

The MPCC controller only requires a *continuously differentiable* 3D path
p_d(theta), parameterized by arc length theta. It does NOT need to be
dynamically feasible and carries no time information. This class builds such a
path from a sequence of sampled 3D points (e.g. the output of a point-mass /
min-snap planner) by:

  1. Computing a chord-length parameter as an approximation of arc length.
  2. Fitting C2 cubic splines x(theta), y(theta), z(theta).
  3. Exposing p_d, p_d', p_d'' for the controller.
  4. Providing qc(theta): a Gaussian "bump" of the contour weight at gate
     locations (Sec. IV-C).

Only scipy + numpy are required here (no acados / casadi).
"""

import numpy as np
from scipy.interpolate import CubicSpline


class ReferencePath:
    def __init__(self, points, closed=False, gate_indices=None,
                 qc_nom=1.0, qc_gate=100.0, gate_sigma=0.6):
        points = np.asarray(points, dtype=float)
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (P,3)"

        if closed and not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.s = s
        self.length = float(s[-1])
        self.closed = bool(closed)

        bc = "periodic" if closed else "natural"
        self._sx = CubicSpline(s, points[:, 0], bc_type=bc)
        self._sy = CubicSpline(s, points[:, 1], bc_type=bc)
        self._sz = CubicSpline(s, points[:, 2], bc_type=bc)
        self._dx, self._dy, self._dz = (sp.derivative(1) for sp in
                                        (self._sx, self._sy, self._sz))
        self._d2x, self._d2y, self._d2z = (sp.derivative(2) for sp in
                                           (self._sx, self._sy, self._sz))

        if gate_indices is not None:
            self.gate_s = np.array([s[i] for i in gate_indices], dtype=float)
        else:
            self.gate_s = np.array([], dtype=float)

        self.qc_nom = float(qc_nom)
        self.qc_gate = float(qc_gate)
        self.gate_sigma = float(gate_sigma)

    # ------------------------------------------------------------------ utils
    def _wrap(self, theta):
        if self.closed:
            return float(np.mod(theta, self.length))
        return float(np.clip(theta, 0.0, self.length))

    def eval(self, theta):
        th = self._wrap(theta)
        return np.array([float(self._sx(th)), float(self._sy(th)), float(self._sz(th))])

    def deriv1(self, theta):
        th = self._wrap(theta)
        return np.array([float(self._dx(th)), float(self._dy(th)), float(self._dz(th))])

    def deriv2(self, theta):
        th = self._wrap(theta)
        return np.array([float(self._d2x(th)), float(self._d2y(th)), float(self._d2z(th))])

    def tangent(self, theta):
        t = self.deriv1(theta)
        n = np.linalg.norm(t)
        return t / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def qc(self, theta):
        """Dynamic contour weight: nominal away from gates, raised near gates."""
        if self.gate_s.size == 0:
            return self.qc_nom
        th = self._wrap(theta)
        d = th - self.gate_s
        if self.closed:
            d = (d + self.length / 2.0) % self.length - self.length / 2.0
        bump = float(np.exp(-0.5 * (d / self.gate_sigma) ** 2).sum())
        bump = min(bump, 1.0)
        return self.qc_nom + (self.qc_gate - self.qc_nom) * bump

    # ------------------------------------------------------------- projection
    def project(self, pos, theta_guess=None, n_coarse=400, n_newton=5):
        pos = np.asarray(pos, dtype=float)
        if theta_guess is None:
            grid = np.linspace(0.0, self.length, n_coarse, endpoint=not self.closed)
        else:
            w = max(self.length / 6.0, 3.0)
            grid = np.linspace(theta_guess - w, theta_guess + w, n_coarse)
        dists = [np.sum((pos - self.eval(t)) ** 2) for t in grid]
        theta = float(grid[int(np.argmin(dists))])

        for _ in range(n_newton):
            e = self.eval(theta) - pos
            d1 = self.deriv1(theta)
            d2 = self.deriv2(theta)
            g = float(e @ d1)
            h = float(d1 @ d1 + e @ d2)
            if abs(h) < 1e-9:
                break
            theta -= g / h
        return self._wrap(theta) if self.closed else float(np.clip(theta, 0, self.length))
