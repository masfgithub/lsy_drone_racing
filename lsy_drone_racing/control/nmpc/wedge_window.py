"""Wedge prism window gate — geometry, soft constraints, and visualisation.

Each of the four wall bars is a wedge-shaped prism fully determined by the
standard gate dimensions.  No extra parameters are required.

Gate-local coordinate frame
----------------------------
    x  — gate normal  (drone flies through along ±x)
    y  — horizontal span  (width direction)
    z  — vertical span    (height direction)

Three orthographic views of the LEFT bar
-----------------------------------------
    Front view (along x): trapezoid
        wide  side at y = -hl : z ∈ [-hh,  +hh ]  (full gate height)
        narrow side at y = -hw : z ∈ [-hho, +hho]  (hole height)

    Side view (along y): rectangle
        x ∈ [-a_x, +a_x],  z ∈ [-hh, +hh]

    Top view (along -z): triangle
        base at y = -hl : x ∈ [-a_x, +a_x]
        tip  at y = -hw : x = 0  (sharp edge)

The four tips meet exactly at the four corners of the hole.

Inside-test (left bar)
-----------------------
Given gate-local point (px, py, pz), the point is inside the left wedge when:

    t = (py - (-hl)) / ((-hw) - (-hl))   ∈ [0, 1]   (depth fraction, base→tip)
    |px| ≤ a_x  * (1 - t)                             (x tapers to 0 at tip)
    |pz| ≤ hh   * (1 - t) + hho * t                   (z tapers from hh to hho)

Right / Top / Bottom bars are handled by mirroring py or pz.

Parameter vector (N_PARAMS = 17 — identical to original Window)
---------------------------------------------------------------
    index  0- 2 : cx, cy, cz          gate centre in world frame
    index  3-11 : R_00..R_22          world-to-gate rotation, row-major
    index    12 : a_x                 thickness / 2
    index    13 : hl                  total_length / 2
    index    14 : hh                  total_height / 2
    index    15 : hw                  hole_width   / 2
    index    16 : hho                 hole_height  / 2
"""

from __future__ import annotations

import numpy as np
from casadi import MX, fabs, fmax, fmin, tanh, vertcat


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3×3 rotation matrix."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


class WedgeWindow:
    """Gate frame built from four wedge-shaped prisms.

    The wedge geometry is fully determined by the standard gate dimensions:
    total_length, total_height, hole_width, hole_height, thickness.
    Each tip meets exactly the corresponding corner of the hole.

    Args:
        position:     World-space gate centre [x, y, z].
        quaternion:   Gate orientation [qw, qx, qy, qz].
        total_length: Outer gate width  (y-span in gate frame), metres.
        total_height: Outer gate height (z-span in gate frame), metres.
        hole_width:   Hole width  (y-span), metres.
        hole_height:  Hole height (z-span), metres.
        thickness:    Wedge base depth along gate normal (x-axis), metres.
                      a_x = thickness / 2.
        margin:       Clearance for verify() checks, metres.
    """

    N_PARAMS: int = 17

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
        self.position = np.asarray(position, dtype=float)
        self.quaternion = np.asarray(quaternion, dtype=float)
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.total_length = float(total_length)
        self.total_height = float(total_height)
        self.hole_width = float(hole_width)
        self.hole_height = float(hole_height)
        self.thickness = float(thickness)
        self.margin = float(margin)
        self.R = _quat_to_rot(self.quaternion).T  # world-to-gate
        self._update_derived()

    # ------------------------------------------------------------------
    # Internal geometry
    # ------------------------------------------------------------------

    def _update_derived(self):
        self.a_x = self.thickness / 2.0
        self.hl = self.total_length / 2.0
        self.hh = self.total_height / 2.0
        self.hw = self.hole_width / 2.0
        self.hho = self.hole_height / 2.0

    # ------------------------------------------------------------------
    # Parameter vector
    # ------------------------------------------------------------------

    def param_vector(self) -> np.ndarray:
        """Flat parameter vector of length N_PARAMS=17."""
        return np.array(
            [
                *self.position,  # 0-2
                *self.R.flatten(),  # 3-11
                self.a_x,  # 12
                self.hl,  # 13
                self.hh,  # 14
                self.hw,  # 15
                self.hho,  # 16
            ]
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

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
        """Update any subset of gate parameters in-place."""
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        if quaternion is not None:
            self.quaternion = np.asarray(quaternion, dtype=float)
            self.quaternion /= np.linalg.norm(self.quaternion)
            self.R = _quat_to_rot(self.quaternion).T
        if total_length is not None:
            self.total_length = float(total_length)
        if total_height is not None:
            self.total_height = float(total_height)
        if hole_width is not None:
            self.hole_width = float(hole_width)
        if hole_height is not None:
            self.hole_height = float(hole_height)
        if thickness is not None:
            self.thickness = float(thickness)
        self._update_derived()

    # ------------------------------------------------------------------
    # CasADi symbolic penalty
    # ------------------------------------------------------------------

    @staticmethod
    def casadi_penalty_sym(pos_sym: MX, p_win: MX) -> MX:
        """Soft penalty for all four wedge bars.

        Inside-test for the LEFT bar
        ------------------------------
        Depth fraction (base → tip, clamped to [0, 1]):

            t = (py - (-hl)) / (hl - hw)   =   (py + hl) / (hl - hw)

        Tapered extents at depth t:

            hx(t) = a_x * (1 - t)              x tapers from a_x to 0
            hz(t) = hh  * (1 - t) + hho * t    z tapers from hh  to hho

        Penetration depths (positive when inside):

            pen_x = hx(t) - |px|
            pen_z = hz(t) - |pz|

        Soft penalty contribution:

            viol = max(0, pen_x) * max(0, pen_z)   product of both violations
            inside_depth = sigmoid(t) * sigmoid(1 - t)   smooth [0,1] gate
            contribution = viol^2 * inside_depth

        Using the product of the two linear penetrations rather than a
        single SDF keeps the expression smooth and provides gradient
        information in both x and z simultaneously.

        The other three bars are handled by mirroring py (right) or pz
        (top/bottom) and swapping the roles of y↔z.

        Args:
            pos_sym: CasADi MX (3,) — robot world position.
            p_win:   CasADi MX (17,) — gate parameter slice.

        Returns:
            Scalar MX — unweighted sum of all four wedge penalties.
        """
        eps = 1e-6
        sharpness = 30.0  # sigmoid steepness for depth-range gate

        # ── Unpack ────────────────────────────────────────────────────────
        cx, cy, cz = p_win[0], p_win[1], p_win[2]

        R_sx = MX(3, 3)
        for i in range(3):
            for j in range(3):
                R_sx[i, j] = p_win[3 + i * 3 + j]

        a_x = p_win[12]
        hl = p_win[13]
        hh = p_win[14]
        hw = p_win[15]
        hho = p_win[16]

        # ── Gate-local position ───────────────────────────────────────────
        dp = vertcat(pos_sym[0] - cx, pos_sym[1] - cy, pos_sym[2] - cz)
        p_loc = R_sx @ dp
        px = p_loc[0]
        py = p_loc[1]
        pz = p_loc[2]

        # ── Per-bar wedge penalty ─────────────────────────────────────────
        #
        # depth_coord : signed coordinate along the bar's depth axis,
        #               oriented so that the base is at -hl_d and tip at -hw_d
        # hl_d        : half-extent at base  (hl for L/R bars, hh for T/B)
        # hw_d        : half-extent at tip   (hw for L/R bars, hho for T/B)
        # ax_coord    : coordinate in the x direction (always px, gate normal)
        # h_ax        : half-extent in x direction    (always a_x)
        # perp_coord  : coordinate in the perpendicular direction
        #               (pz for L/R bars, py for T/B bars)
        # hh_base     : half-extent in perp at base   (hh for L/R, hl for T/B)
        # hh_tip      : half-extent in perp at tip     (hho for L/R, hl... wait)

        # For L/R bars:  depth=py, ax=px, perp=pz
        #   base: py=-hl, x∈[-a_x,+a_x], z∈[-hh,+hh]
        #   tip:  py=-hw, x=0,            z∈[-hho,+hho]
        # For T/B bars:  depth=pz, ax=px, perp=py
        #   base: pz=+hh, x∈[-a_x,+a_x], y∈[-hl,+hl]
        #   tip:  pz=+hho, x=0,           y∈[-hl,+hl]  ← perp does NOT taper for T/B
        #
        # Wait — for the TOP bar the front view trapezoid is:
        #   wide at z=+hh: y∈[-hl,+hl]  (full gate width)
        #   narrow at z=+hho: y∈[-hw,+hw]  (hole width)
        # So the perp (y) DOES taper for T/B bars too, from hl to hw.

        def wedge_pen(
            depth_coord, base_d, tip_d, ax_coord, h_ax, perp_coord, h_perp_base, h_perp_tip
        ):
            """Penalty for one wedge bar.

            depth_coord  : coordinate along depth axis (signed, base at base_d,
                           tip at tip_d, with base_d < tip_d so we flip sign)
            base_d       : depth value at outer base  (negative, e.g. -hl)
            tip_d        : depth value at tip         (negative, e.g. -hw)
            ax_coord     : gate-normal coord (px), tapers from h_ax to 0
            perp_coord   : perpendicular coord (pz or py), tapers from
                           h_perp_base to h_perp_tip
            """
            span = tip_d - base_d  # positive (tip_d > base_d since tip is less negative)

            # Depth fraction t ∈ [0, 1]: 0 at base, 1 at tip
            t_raw = (depth_coord - base_d) / (span + eps)

            # Smooth gate: contribution is 0 outside [0, 1]
            inside_base = 0.5 * (1.0 + tanh(sharpness * t_raw))  # 1 past base
            inside_tip = 0.5 * (1.0 - tanh(sharpness * (t_raw - 1.0)))  # 1 before tip
            inside_depth = inside_base * inside_tip

            # Tapered half-extents at depth t
            t_clamp = fmax(MX(0), fmin(MX(1), t_raw))
            hx_t = h_ax * (1.0 - t_clamp)  # x: a_x → 0
            hperp_t = h_perp_base * (1.0 - t_clamp) + h_perp_tip * t_clamp

            # Penetration in each dimension (positive when inside)
            pen_x = fmax(MX(0), hx_t - fabs(ax_coord))
            pen_perp = fmax(MX(0), hperp_t - fabs(perp_coord))

            # Penalty: product of both penetrations squared, gated by depth
            return pen_x * pen_x * pen_perp * pen_perp * inside_depth

        # ── Left bar:   base at py = -hl, tip at py = -hw ────────────────
        #   depth = py,  ax = px (a_x→0),  perp = pz (hh→hho)
        pen_L = wedge_pen(
            depth_coord=py,
            base_d=-hl,
            tip_d=-hw,
            ax_coord=px,
            h_ax=a_x,
            perp_coord=pz,
            h_perp_base=hh,
            h_perp_tip=hho,
        )

        # ── Right bar:  base at py = +hl, tip at py = +hw ────────────────
        #   mirror py → use -py so base is at -hl again
        pen_R = wedge_pen(
            depth_coord=-py,
            base_d=-hl,
            tip_d=-hw,
            ax_coord=px,
            h_ax=a_x,
            perp_coord=pz,
            h_perp_base=hh,
            h_perp_tip=hho,
        )

        # ── Top bar:    base at pz = +hh, tip at pz = +hho ───────────────
        #   depth = -pz (so base is at -hh and tip at -hho in local terms)
        #   ax = px (a_x→0),  perp = py (hl→hw)
        pen_T = wedge_pen(
            depth_coord=-pz,
            base_d=-hh,
            tip_d=-hho,
            ax_coord=px,
            h_ax=a_x,
            perp_coord=py,
            h_perp_base=hl,
            h_perp_tip=hw,
        )

        # ── Bottom bar: base at pz = -hh, tip at pz = -hho ───────────────
        pen_B = wedge_pen(
            depth_coord=pz,
            base_d=-hh,
            tip_d=-hho,
            ax_coord=px,
            h_ax=a_x,
            perp_coord=py,
            h_perp_base=hl,
            h_perp_tip=hw,
        )

        return pen_L + pen_R + pen_T + pen_B

    # ------------------------------------------------------------------
    # Numerical verification
    # ------------------------------------------------------------------

    def _inside_wedge(
        self,
        depth: float,
        base_d: float,
        tip_d: float,
        ax: float,
        perp: float,
        h_perp_base: float,
        h_perp_tip: float,
    ) -> bool:
        """Return True if the point is inside one wedge bar (NumPy)."""
        span = tip_d - base_d
        if abs(span) < 1e-9:
            return False
        t = (depth - base_d) / span
        if t < 0.0 or t > 1.0:
            return False
        hx_t = self.a_x * (1.0 - t)
        hperp_t = h_perp_base * (1.0 - t) + h_perp_tip * t
        return abs(ax) <= hx_t + self.margin and abs(perp) <= hperp_t + self.margin

    def _point_in_any_wedge(self, pos_world: np.ndarray) -> bool:
        dp = pos_world - self.position
        p_loc = self.R @ dp
        px, py, pz = p_loc
        return bool(
            self._inside_wedge(py, -self.hl, -self.hw, px, pz, self.hh, self.hho)  # L
            or self._inside_wedge(-py, -self.hl, -self.hw, px, pz, self.hh, self.hho)  # R
            or self._inside_wedge(-pz, -self.hh, -self.hho, px, py, self.hl, self.hw)  # T
            or self._inside_wedge(pz, -self.hh, -self.hho, px, py, self.hl, self.hw)  # B
        )

    def verify(self, x_traj: np.ndarray) -> bool:
        """Return True if no trajectory node is inside any wedge."""
        clean = True
        for k in range(x_traj.shape[0]):
            pos = x_traj[k, :3]
            if self._point_in_any_wedge(pos):
                dp = pos - self.position
                p_loc = self.R @ dp
                print(
                    f"  VIOLATION node={k:3d} "
                    f"local=[{p_loc[0]:.3f}, {p_loc[1]:.3f}, {p_loc[2]:.3f}]"
                )
                clean = False
        return clean

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def hole_centre_world(self) -> np.ndarray:
        return self.position.copy()

    @property
    def z_top_world(self) -> float:
        return float(self.position[2] + self.hh)

    # ------------------------------------------------------------------
    # Matplotlib visualisation
    # ------------------------------------------------------------------

    def draw(self, ax: object, color: str = "saddlebrown", alpha: float = 0.45):
        """Draw the four wedge prisms on a Matplotlib Axes3D."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        def to_world(pts: np.ndarray) -> np.ndarray:
            return self.position + (self.R.T @ pts.T).T

        def draw_prism(base_d, tip_d, depth_idx, ax_idx, perp_idx, h_perp_base, h_perp_tip):
            """Draw one wedge prism (6 vertices, 5 faces)."""

            def v(d, xv, perpv):
                pt = np.zeros(3)
                pt[depth_idx] = d
                pt[ax_idx] = xv
                pt[perp_idx] = perpv
                return pt

            # 4 base corners, 2 tip corners
            B = [
                v(base_d, -self.a_x, h_perp_base),  # B0: -x, +perp
                v(base_d, self.a_x, h_perp_base),  # B1: +x, +perp
                v(base_d, self.a_x, -h_perp_base),  # B2: +x, -perp
                v(base_d, -self.a_x, -h_perp_base),
            ]  # B3: -x, -perp
            T = [
                v(tip_d, 0.0, h_perp_tip),  # T0: tip, +perp
                v(tip_d, 0.0, -h_perp_tip),
            ]  # T1: tip, -perp

            verts = to_world(np.array(B + T))
            B0, B1, B2, B3, T0, T1 = verts

            faces = [
                [B0, B1, B2, B3],  # base rectangle
                [B0, B1, T0],  # +perp slant top  (triangle)
                [B3, B2, T1],  # -perp slant bot  (triangle)
                [B0, B3, T1, T0],  # -x slant face    (quad)
                [B1, B2, T1, T0],  # +x slant face    (quad)
            ]
            if abs(h_perp_tip) > 1e-6:
                # Replace triangles with quads when tip has finite width
                faces[1] = [B0, B1, T0]
                faces[2] = [B2, B3, T1]

            ax.add_collection3d(
                Poly3DCollection(
                    faces, facecolor=color, alpha=alpha, edgecolor="sienna", linewidth=0.5
                )
            )

        # Left:   depth=y(1), ax=x(0), perp=z(2), base at -hl, tip at -hw
        draw_prism(-self.hl, -self.hw, 1, 0, 2, self.hh, self.hho)
        # Right:  mirror — base at +hl, tip at +hw
        draw_prism(self.hl, self.hw, 1, 0, 2, self.hh, self.hho)
        # Top:    depth=z(2), ax=x(0), perp=y(1), base at +hh, tip at +hho
        draw_prism(self.hh, self.hho, 2, 0, 1, self.hl, self.hw)
        # Bottom: base at -hh, tip at -hho
        draw_prism(-self.hh, -self.hho, 2, 0, 1, self.hl, self.hw)
