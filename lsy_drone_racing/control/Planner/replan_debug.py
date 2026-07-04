from __future__ import annotations

import os

import matplotlib

# CHANGE 1: Use an interactive backend (TkAgg, QtAgg, etc.) instead of 'Agg'
matplotlib.use("TkAgg")  

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

DEFAULT_FRAME_WIDTH = 0.72   
DEFAULT_FRAME_OPENING = 0.4  
DEFAULT_R_OBSTACLE = 0.2     
DEFAULT_CLEARANCE = 0.1      
DEFAULT_OBSTACLE_PILLAR_R = 0.15  
DEFAULT_OBSTACLE_HEIGHT = 2.0    


class ReplanDebugger:
    """Spawns an interactive 3D Matplotlib GUI window every time `.dump()` is called,
    while saving a copy of the PNG to the output directory.
    """

    def __init__(
        self,
        output_dir: str = "replan_debug",
        frame_width: float = DEFAULT_FRAME_WIDTH,
        frame_opening: float = DEFAULT_FRAME_OPENING,
        r_obstacle: float = DEFAULT_R_OBSTACLE,
        clearance: float = DEFAULT_CLEARANCE,
        obstacle_pillar_r: float = DEFAULT_OBSTACLE_PILLAR_R,
        obstacle_height: float = DEFAULT_OBSTACLE_HEIGHT,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.counter = 0

        self.frame_width = frame_width
        self.frame_opening = frame_opening
        self.r_obstacle = r_obstacle
        self.clearance = clearance
        self.obstacle_pillar_r = obstacle_pillar_r
        self.obstacle_height = obstacle_height

    def dump(
        self,
        env_states,
        trajectory,
        planner=None,
        tag: str | None = None,
        t_elapsed: float | None = None,
    ) -> str:
        # --- pull data out of the inputs --------------------------------------
        start = np.asarray(env_states.pBLL, dtype=float)
        gates = np.asarray(env_states.pTLL_array, dtype=float)
        quats = np.asarray(env_states.qTLT_array, dtype=float)
        yaws = R.from_quat(quats).as_euler("ZYX")[:, 0] if len(quats) else np.zeros(0)
        obstacles = np.asarray(env_states.pOLL_array, dtype=float).reshape(-1, 3)
        gate_idx = int(getattr(env_states, "pTLL_index", 0))

        if hasattr(trajectory, "positions"):
            traj = np.asarray(trajectory.positions, dtype=float)
        else:
            traj = np.asarray(trajectory, dtype=float)

        # CHANGE 2: Dynamically pull waypoints from the planner to match your second function
        wps = None
        if planner is not None:
            wps_attr = getattr(planner, "_waypoints", None)
            if wps_attr is not None and len(wps_attr):
                wps = np.asarray(wps_attr, dtype=float)

        # --- build filename ---------------------------------------------------
        name = tag if tag else "replan"
        filename = f"{self.counter:03d}_{name}.png"
        save_path = os.path.join(self.output_dir, filename)
        self.counter += 1

        # --- draw, save, and show ---------------------------------------------
        self._make_plot(
            save_path=save_path,
            start=start,
            gates=gates,
            yaws=yaws,
            gate_idx=gate_idx,
            obstacles=obstacles,
            traj=traj,
            wps=wps,
            tag=name,
            t_elapsed=t_elapsed,
        )
        return save_path

    def _make_plot(
        self,
        save_path: str,
        start: np.ndarray,
        gates: np.ndarray,
        yaws: np.ndarray,
        gate_idx: int,
        obstacles: np.ndarray,
        traj: np.ndarray,
        wps: np.ndarray | None,
        tag: str,
        t_elapsed: float | None,
    ) -> None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Trajectory
        if traj is not None and len(traj):
            ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                "-", color="tab:green", lw=2.5, label=f"trajectory ({len(traj)} pts)",
            )

        # Waypoints visualization
        if wps is not None and len(wps):
            ax.scatter(
                wps[:, 0], wps[:, 1], wps[:, 2],
                c="orange", marker="D", s=55, edgecolor="k",
                depthshade=False, label=f"waypoints ({len(wps)})",
            )
            for i, p in enumerate(wps):
                ax.text(p[0], p[1], p[2] + 0.05, str(i), fontsize=8)

        # Drone start
        ax.scatter(start[0], start[1], start[2], c="k", s=100, marker="o", label="drone start")

        # Gates
        for k, (c, yaw) in enumerate(zip(gates, yaws)):
            is_active = (k == gate_idx)
            outer_color = "tab:red" if is_active else "tab:blue"
            opening_color = "tab:pink" if is_active else "tab:cyan"
            self._draw_gate(ax, c, yaw, self.frame_width / 2, outer_color, lw=2.5 if is_active else 1.8)
            self._draw_gate(ax, c, yaw, self.frame_opening / 2, opening_color, lw=1.5)
            label = f"G{k}*" if is_active else f"G{k}"
            ax.text(c[0], c[1], c[2] + 0.12, label, fontsize=10, color=outer_color, weight="bold")

        # Obstacles
        for j, o in enumerate(obstacles):
            self._draw_cylinder(
                ax, o, self.obstacle_pillar_r, 0.0, self.obstacle_height,
                "tab:red", 0.55,
            )
            self._draw_cylinder(
                ax, o, self.r_obstacle + self.clearance, 0.0, self.obstacle_height,
                "tab:orange", 0.12,
            )
            ax.text(o[0], o[1], self.obstacle_height + 0.05, f"O{j}",
                    fontsize=10, color="tab:red")

        # Labels and title
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        title = f"Replan #{self.counter - 1}: {tag}"
        if t_elapsed is not None:
            title += f"    (t = {t_elapsed:.2f}s)"
        title += f"\nactive gate: G{gate_idx}  |  obstacles: {len(obstacles)}"
        if wps is not None:
            title += f"  |  waypoints: {len(wps)}"
        ax.set_title(title, fontsize=11)

        ax.legend(loc="upper left", fontsize=9)
        self._equal_3d(ax)

        # Save standard PNG artifact
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"[replan_debug] wrote {save_path}")
        plt.show(block=True)

        # CHANGE 3: Display popup window without locking the thread execution loop
        plt.show(block=False)
        plt.pause(0.1) # Small pause allows the GUI framework to catch up and draw the window

    @staticmethod
    def _draw_gate(ax, c, yaw, half, color, lw=2.0):
        w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        zz = np.array([0.0, 0.0, 1.0])
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        pts = np.array([np.asarray(c) + a * half * w + b * half * zz for a, b in corners])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw)

    @staticmethod
    def _draw_cylinder(ax, c, radius, z0, z1, color, alpha):
        th = np.linspace(0, 2 * np.pi, 28)
        T, Z = np.meshgrid(th, np.array([z0, z1]))
        X = c[0] + radius * np.cos(T)
        Y = c[1] + radius * np.sin(T)
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)

    @staticmethod
    def _equal_3d(ax):
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]))