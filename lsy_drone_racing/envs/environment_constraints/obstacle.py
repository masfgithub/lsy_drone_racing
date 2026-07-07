"""Cylinder obstacle — geometry, constraints, and visualisation."""

import numpy as np
from casadi import MX, sqrt, sumsqr, vertcat

# Number of parameters per obstacle in the parameter vector
# [cx, cy]  — only XY position needed (infinite-height cylinder)
N_PARAMS_OBSTACLE = 2


class CylinderObstacle:
    """An infinite-height cylinder obstacle defined by its XY centre.

    Args:
        position: World-space XY centre [x, y] or [x, y, z] (z ignored).
        d_min:    Minimum allowed distance (constraint lower bound).
    """

    N_PARAMS = N_PARAMS_OBSTACLE

    def __init__(self, position: list | np.ndarray, obstacles_information: dict):
        """Initialize cylinder obstacle with XY centre and minimum clearance distance."""
        self.position = np.asarray(position[:2], dtype=float)
        self._obstacles_information = obstacles_information
        self.d_min = obstacles_information["d_min"]
        self.total_height = obstacles_information["total_height"]

    def param_vector(self) -> np.ndarray:
        """Return [cx, cy]."""
        return self.position.copy()

    def update(self, position: list | np.ndarray | None = None):
        """Update position in-place."""
        if position is not None:
            self.position = np.asarray(position[:2], dtype=float)

    @staticmethod
    def casadi_constraint_sym(pos_sym: MX, p_obs: MX) -> MX:
        """Return SDF expression using a parameter slice [cx, cy].

        Args:
            pos_sym: CasADi MX shape (3,) — robot position.
            p_obs:   CasADi MX shape (2,) — obstacle XY centre.

        Returns:
            Scalar MX: distance in XY from robot to obstacle centre.
        """
        diff = vertcat(pos_sym[0] - p_obs[0], pos_sym[1] - p_obs[1])
        return sqrt(sumsqr(diff))

    def verify(self, x_traj: np.ndarray) -> bool:
        """Check all trajectory nodes against this obstacle."""
        clean = True
        for k in range(x_traj.shape[0]):
            pos = x_traj[k, :2]
            dist = np.linalg.norm(pos - self.position)
            if dist < self.d_min - 1e-3:
                print(f"  VIOLATION node={k:3d} dist={dist:.4f} < d_min={self.d_min:.4f}")
                clean = False
        return clean

    def draw(
        self, ax: object, color: str = "tab:red", alpha: float = 0.18, d_min: float | None = None
    ):
        """Draw cylinder on a Matplotlib Axes3D."""
        import numpy as np

        r = d_min if d_min is not None else self.d_min
        theta = np.linspace(0, 2 * np.pi, 60)
        Theta, Z = np.meshgrid(theta, np.linspace(0, self.total_height, 2))
        ax.plot_surface(
            self.position[0] + r * np.cos(Theta),
            self.position[1] + r * np.sin(Theta),
            Z,
            alpha=alpha,
            color=color,
            zorder=2,
        )
        for z_ring in [0, self.total_height]:
            ax.plot(
                self.position[0] + r * np.cos(theta),
                self.position[1] + r * np.sin(theta),
                z_ring,
                "r--",
                linewidth=0.9,
                alpha=0.6,
            )
        ax.text(
            self.position[0],
            self.position[1],
            self.total_height + 0.15,
            "obs",
            fontsize=7,
            color=color,
        )
