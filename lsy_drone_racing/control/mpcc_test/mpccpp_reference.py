"""MPCC++ reference with a gate-aligned tunnel (Krinner et al., RSS 2024, Sec. IV-A).

The tunnel is built so that at every gate it *coincides with the
gate opening*: the cross section there is the gate rectangle, aligned with the
gate plane and sized to the gate. Staying inside the tunnel therefore
guarantees the drone flies through the gate.

How it works
------------
* Each gate has a center, a through-normal, and an opening (half-width
  gate_w_half, half-height gate_h_half). The normal is given or derived from the
  local track direction.
* Centerline: the gate centers are turned into a spline whose tangent AT each
  gate equals the gate normal -- achieved by inserting two collinear helper
  points c +/- delta*n around every gate ("the centerline matches the first
  derivative with the gate normal").
* Tunnel frame (n, b): at a gate it is the gate's (lateral, vertical) axes;
  between gates it is a smooth blend of the two neighbouring gate frames,
  re-orthonormalized against the path tangent.
* Tunnel size: pinches to the gate opening at each gate and widens to
  (W_nom, H_nom) in between.

The base ReferencePath (spline, qc, projection) is reused unchanged.
"""

import numpy as np

from lsy_drone_racing.control.mpcc_test.mpcc_reference import ReferencePath


def _perp(t: np.ndarray, up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit vector perpendicular to t, close to up."""
    up = np.asarray(up, float)
    if abs(np.dot(t, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(t, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
    n = up - np.dot(up, t) * t
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])


def _gate_axes(n: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lateral (horizontal) and vertical axes of a gate with through-normal n."""
    w = np.cross(up, n)
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(np.array([1.0, 0.0, 0.0]), n)
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(np.array([0.0, 1.0, 0.0]), n)
    w = w / np.linalg.norm(w)
    h = np.cross(n, w)
    h = h / np.linalg.norm(h)
    return w, h


class TunnelReferencePath(ReferencePath):
    """Gate-aligned tunnel reference path that pinches to each gate's opening."""

    def __init__(
        self,
        gate_centers: np.ndarray,
        gate_normals: np.ndarray | None = None,
        gate_w_half: float = 1.0,
        gate_h_half: float = 1.0,
        closed: bool = False,
        qc_nom: float = 1.0,
        qc_gate: float = 120.0,
        gate_sigma: float = 0.8,
        W_nom: float = 3.0,
        H_nom: float | None = None,
        tunnel_sigma: float = 1.0,
        frame_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        gate_tangent_len: float = 0.5,
    ):
        """Build the tunnel centerline and per-gate frame/size data.

        Args:
            gate_centers: Gate center positions, shape (M, 3).
            gate_normals: Gate through-normals, shape (M, 3); derived from the
                track direction if None.
            gate_w_half: Half-width of the gate opening(s).
            gate_h_half: Half-height of the gate opening(s).
            closed: Whether the track is a closed loop.
            qc_nom: Nominal contouring cost weight.
            qc_gate: Contouring cost weight boost near gates.
            gate_sigma: Width (in path length) of the qc_gate boost around each gate.
            W_nom: Nominal tunnel half-width away from gates.
            H_nom: Nominal tunnel half-height away from gates; defaults to W_nom.
            tunnel_sigma: Width (in path length) of the pinch to the gate opening.
            frame_up: Approximate "up" direction used to disambiguate gate axes.
            gate_tangent_len: Offset used to fit the centerline tangent to the gate normal.
        """
        centers = np.asarray(gate_centers, dtype=float)
        assert centers.ndim == 2 and centers.shape[1] == 3, "gate_centers must be (M,3)"
        M = len(centers)
        up = np.asarray(frame_up, float)

        # --- gate through-normals -------------------------------------------
        if gate_normals is None:
            normals = np.zeros((M, 3))
            for i in range(M):
                if closed:
                    normals[i] = centers[(i + 1) % M] - centers[(i - 1) % M]
                else:
                    normals[i] = centers[min(i + 1, M - 1)] - centers[max(i - 1, 0)]
        else:
            normals = np.asarray(gate_normals, float)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        gw = np.zeros((M, 3))
        gh = np.zeros((M, 3))
        for i in range(M):
            gw[i], gh[i] = _gate_axes(normals[i], up)

        gate_hw = np.broadcast_to(gate_w_half, (M,)).astype(float).copy()
        gate_hh = np.broadcast_to(gate_h_half, (M,)).astype(float).copy()

        # --- centerline knots: tangent == gate normal at each gate ----------
        delta = float(gate_tangent_len)
        aug, gidx = [], []
        for i in range(M):
            c, n = centers[i], normals[i]
            aug.append(c - delta * n)
            aug.append(c)
            gidx.append(len(aug) - 1)
            aug.append(c + delta * n)
        aug = np.array(aug)

        super().__init__(aug, closed=closed, gate_indices=gidx,
                         qc_nom=qc_nom, qc_gate=qc_gate, gate_sigma=gate_sigma)

        # store gate geometry (gate_s was set by the base from gidx)
        self.gate_centers = centers
        self.gate_n, self.gate_w, self.gate_h = normals, gw, gh
        self.gate_hw, self.gate_hh = gate_hw, gate_hh
        self.W_nom = float(W_nom)
        self.H_nom = float(W_nom if H_nom is None else H_nom)
        self.tunnel_sigma = float(tunnel_sigma)
        self._up = up

    # ----------------------------------------------------------- tunnel frame
    def _segment(self, th: float) -> tuple[int, int, float]:
        """Index of the gate segment containing th and the interpolation fraction within it."""
        gs = self.gate_s
        M = len(gs)
        L = self.length
        if M == 1:
            return 0, 0, 0.0
        if self.closed:
            if th < gs[0]:
                lo, hi = gs[-1] - L, gs[0]
                return M - 1, 0, (th - lo) / (hi - lo + 1e-12)
            if th >= gs[-1]:
                lo, hi = gs[-1], gs[0] + L
                return M - 1, 0, (th - lo) / (hi - lo + 1e-12)
        else:
            th = min(max(th, gs[0]), gs[-1])
        i = int(np.searchsorted(gs, th, side="right")) - 1
        i = max(0, min(M - 2, i))
        return i, i + 1, (th - gs[i]) / (gs[i + 1] - gs[i] + 1e-12)

    def frame(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Tunnel lateral/vertical axes (n, b) at path position theta."""
        t = self.tangent(theta)
        i, j, a = self._segment(self._wrap(theta))
        ew = (1 - a) * self.gate_w[i] + a * self.gate_w[j]
        eh = (1 - a) * self.gate_h[i] + a * self.gate_h[j]
        n = ew - np.dot(ew, t) * t                  # orthonormalize vs tangent
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else _perp(t)
        b = np.cross(t, n)
        if np.dot(b, eh) < 0:                       # keep b aligned with vertical
            n, b = -n, -b
        return n, b

    # ------------------------------------------------------------- tunnel size
    def width(self, theta: float) -> tuple[float, float]:
        """Tunnel half-width and half-height at path position theta."""
        th = self._wrap(theta)
        d = th - self.gate_s
        if self.closed:
            d = (d + self.length / 2.0) % self.length - self.length / 2.0
        g = np.exp(-0.5 * (d / self.tunnel_sigma) ** 2)          # per-gate bump
        W = self.W_nom - float(np.sum((self.W_nom - self.gate_hw) * g))
        H = self.H_nom - float(np.sum((self.H_nom - self.gate_hh) * g))
        return max(W, 0.05), max(H, 0.05)

    # -------------------------------------------------------------- plotting
    def gate_rect(self, i: int) -> np.ndarray:
        """Closed polygon (5,3) of gate i's opening rectangle, for plotting."""
        c, w, h = self.gate_centers[i], self.gate_w[i], self.gate_h[i]
        hw, hh = self.gate_hw[i], self.gate_hh[i]
        return np.array([c + hw * w + hh * h, c - hw * w + hh * h,
                         c - hw * w - hh * h, c + hw * w - hh * h,
                         c + hw * w + hh * h])
