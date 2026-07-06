"""Window obstacle — geometry, constraints, and visualisation."""

import numpy as np
from casadi import MX, fabs, fmax, fmin, sqrt, vertcat


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


class Window:
    """A rectangular wall with a rectangular hole, placed arbitrarily in 3D."""

    N_PARAMS = 17  # [cx, cy, cz, R(9), hx, hl, hh, hw, hho]

    def __init__(
        self,
        position: list | np.ndarray,
        quaternion: list | np.ndarray,
        total_length: float,
        total_height: float,
        hole_width: float,
        hole_height: float,
        thickness: float,
        margin: float = 0.05,
    ):
        """Initialize a window with given pose, outer dimensions, hole size, and margin."""
        self.position = np.asarray(position, dtype=float)
        self.quaternion = np.asarray(quaternion, dtype=float)
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.total_length = total_length
        self.total_height = total_height
        self.hole_width = hole_width
        self.hole_height = hole_height
        self.thickness = thickness
        self.margin = margin
        self.R = _quat_to_rot(self.quaternion).T  # world-to-gate
        self._panels = self._build_panels()

    def param_vector(self) -> np.ndarray:
        """Return the flat parameter vector [cx, cy, cz, R(9), hx, hl, hh, hw, hho]."""
        cx, cy, cz = self.position
        return np.array(
            [
                cx,
                cy,
                cz,
                *self.R.flatten(),
                self.thickness / 2.0,
                self.total_length / 2.0,
                self.total_height / 2.0,
                self.hole_width / 2.0,
                self.hole_height / 2.0,
            ]
        )

    def update(
        self,
        position: list | np.ndarray | None = None,
        quaternion: list | np.ndarray | None = None,
        total_length: float | None = None,
        total_height: float | None = None,
        hole_width: float | None = None,
        hole_height: float | None = None,
        thickness: float | None = None,
    ):
        """Update any subset of window parameters in-place and rebuild panels."""
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        if quaternion is not None:
            self.quaternion = np.asarray(quaternion, dtype=float)
            self.quaternion /= np.linalg.norm(self.quaternion)
            self.R = _quat_to_rot(self.quaternion).T  # world-to-gate

        if total_length is not None:
            self.total_length = total_length
        if total_height is not None:
            self.total_height = total_height
        if hole_width is not None:
            self.hole_width = hole_width
        if hole_height is not None:
            self.hole_height = hole_height
        if thickness is not None:
            self.thickness = thickness
        self._panels = self._build_panels()

    def _build_panels(self) -> list[dict]:
        R = self.R
        pos = self.position
        hx = self.thickness / 2.0
        hl = self.total_length / 2.0
        hh = self.total_height / 2.0
        hw = self.hole_width / 2.0
        hho = self.hole_height / 2.0
        side_hy = (hl - hw) / 2.0
        side_cy = hw + side_hy
        cap_hz = (hh - hho) / 2.0
        cap_cz = hho + cap_hz

        def panel(c_local: list, hy: float, hz: float) -> dict:
            c_world = pos + R.T @ np.asarray(c_local)
            return {"center": c_world.tolist(), "half_extents": [hx, hy, hz], "R": R}

        return [
            panel([0.0, -side_cy, 0.0], side_hy, hh),
            panel([0.0, side_cy, 0.0], side_hy, hh),
            panel([0.0, 0.0, -cap_cz], hw, cap_hz),
            panel([0.0, 0.0, cap_cz], hw, cap_hz),
        ]

    @property
    def panels(self) -> list[dict]:
        """Return the four solid panels that form the window frame."""
        return self._panels

    @property
    def hole_centre_world(self) -> np.ndarray:
        """Return the world-space centre of the hole (coincides with window position)."""
        return self.position.copy()

    @property
    def z_top_world(self) -> float:
        """Return the world-space z coordinate of the top edge of the window."""
        return float(self.position[2] + self.total_height / 2.0)

    def _point_in_solid(self, pos_world: np.ndarray) -> bool:
        dp = pos_world - self.position
        p_local = self.R @ dp
        hx = self.thickness / 2.0 + self.margin
        hl = self.total_length / 2.0
        hh = self.total_height / 2.0
        hw = self.hole_width / 2.0
        hho = self.hole_height / 2.0
        in_slab = abs(p_local[0]) <= hx
        in_total = abs(p_local[1]) <= hl and abs(p_local[2]) <= hh
        in_hole = abs(p_local[1]) <= hw and abs(p_local[2]) <= hho
        return bool(in_slab and in_total and not in_hole)

    def verify(self, x_traj: np.ndarray) -> bool:
        """Return True if no trajectory node penetrates the window solid."""
        clean = True
        for k in range(x_traj.shape[0]):
            pos = x_traj[k, :3]
            if self._point_in_solid(pos):
                dp = pos - self.position
                p_local = self.R @ dp
                print(
                    f"  VIOLATION node={k:3d} "
                    f"local=[{p_local[0]:.3f}, {p_local[1]:.3f}, {p_local[2]:.3f}]"
                )
                clean = False
        return clean

    @staticmethod
    def casadi_constraints_sym(pos_sym: MX, p_win: MX) -> MX:
        """Return a smooth signed-distance expression that is positive when pos_sym is free."""
        eps = 1e-4
        cx, cy, cz = p_win[0], p_win[1], p_win[2]
        R_sx = MX(3, 3)
        for i in range(3):
            for j in range(3):
                R_sx[i, j] = p_win[3 + i * 3 + j]
        hx = p_win[12]
        hl = p_win[13]
        hh = p_win[14]
        hw = p_win[15]
        hho = p_win[16]
        dp = vertcat(pos_sym[0] - cx, pos_sym[1] - cy, pos_sym[2] - cz)
        p = R_sx @ dp
        d_slab = fabs(p[0]) - hx
        d_outside_extent = fmax(fabs(p[1]) - hl, fabs(p[2]) - hh)
        in_hole = fmin(hw - fabs(p[1]), hho - fabs(p[2]))
        free = fmax(fmax(d_slab, d_outside_extent), in_hole)
        return sqrt(free * free + eps) + free - sqrt(eps)

    def draw(self, ax: object, color: str = "saddlebrown", alpha: float = 0.45):
        """Draw all four window panels on a Matplotlib Axes3D."""
        for panel in self._panels:
            self._draw_panel(ax, panel, color, alpha)

    @staticmethod
    def _draw_panel(ax: object, panel: dict, color: str, alpha: float):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        cx, cy, cz = panel["center"]
        bx, by, bz = panel["half_extents"]
        R = panel["R"]
        centre = np.array([cx, cy, cz])
        cl = np.array(
            [
                [-bx, -by, -bz],
                [bx, -by, -bz],
                [bx, by, -bz],
                [-bx, by, -bz],
                [-bx, -by, bz],
                [bx, -by, bz],
                [bx, by, bz],
                [-bx, by, bz],
            ]
        )
        cw = (R.T @ cl.T).T + centre
        faces = [
            [cw[0], cw[1], cw[2], cw[3]],
            [cw[4], cw[5], cw[6], cw[7]],
            [cw[0], cw[1], cw[5], cw[4]],
            [cw[2], cw[3], cw[7], cw[6]],
            [cw[0], cw[3], cw[7], cw[4]],
            [cw[1], cw[2], cw[6], cw[5]],
        ]
        ax.add_collection3d(
            Poly3DCollection(faces, facecolor=color, alpha=alpha, edgecolor="sienna", linewidth=0.6)
        )
