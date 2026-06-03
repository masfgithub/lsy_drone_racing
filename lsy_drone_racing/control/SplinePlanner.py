"""Time-parameterized cubic-spline planner for state-control drone racing."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.planner import Planner, Trajectory
from lsy_drone_racing.control.env_obs import EnvState_t


class SplinePlanner(Planner):
    """Plans a smooth path through the remaining gates and assigns a
    constant-speed time profile along it (no MPCC needed)."""

    def __init__(self, obs: EnvState_t, info: dict, config: dict,
                 t_total: float, max_speed: float = 2.0):
        self.info = info
        self.config = config
        self._t_total = float(t_total)
        self.max_speed = float(max_speed)
        self.freq = config.env.freq 
        self._speed = None
        
        self.clearance = 0.3
        self.trajectory = self.plan(obs, info, config)

    def plan(self, obs: EnvState_t, info: dict, config: dict,
             t0: float = 0.0) -> Trajectory:
        start = np.asarray(obs.pBLL, dtype=float)
        gates = self._remaining_gates(obs)

        waypoints = self._build_waypoints(obs)
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])

        if self._speed is None:
            self._speed = total / self._t_total
        speed = self._speed

        spline = CubicSpline(s, waypoints, axis=0)
        duration = total / speed
        n = max(2, int(round(duration * self.freq)))
        s_samp = np.linspace(0.0, total, n)

        positions = spline(s_samp)
        positions[:, 2] = np.maximum(positions[:, 2], 0.1)
        #print(obs.pBLL)
        #print(positions)
        tangents = spline(s_samp, 1)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        velocities = tangents * speed
        timestamps = t0 + s_samp / speed              # absolute time
        print(f"n={n}  duration={duration:.2f}s  total_len={total:.2f}m")
        return Trajectory(positions, velocities, timestamps)

    def replan(self, obs: EnvState_t, elapsed: float = 0.0) -> Trajectory:
        """Full replan from the live state through remaining gates."""
        self.trajectory = self.plan(obs, self.info, self.config, elapsed)
        return self.trajectory

    def _remaining_gates(self, obs: EnvState_t) -> np.ndarray:
        gates = np.asarray(obs.pTLL_array, dtype=float)
        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        if idx < 0:
            return np.empty((0, 3))
        return gates[idx:]
    
    def _gate_normals(self, obs: EnvState_t) -> np.ndarray:
        quats = np.asarray(obs.qTLT_array, dtype=float)
        return Rotation.from_quat(quats).as_matrix()[:, :, 0]

    def _build_waypoints(self, obs: EnvState_t) -> np.ndarray:
        start = np.asarray(obs.pBLL, dtype=float)
        gates = self._remaining_gates(obs)
        if gates.shape[0] == 0:
            return start[None, :]

        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        normals_all, y_all, z_all = self._gate_frames(obs)
        normals = normals_all[idx:]
        y_axes = y_all[idx:]
        z_axes = z_all[idx:]

        d = 0.6
        detour_dist = 0.65
        angle_thresh = 120.0

        prev = start
        oriented_normals = []
        gate_blocks = []                      # the 5 approach points per gate
        for gi, (c, n) in enumerate(zip(gates, normals)):
            n = n / (np.linalg.norm(n) + 1e-9)
            if np.dot(c - prev, n) < 0:
                n = -n
            oriented_normals.append(n)
            block = []
            for offset in (-d, -d/2, 0.0, d/2, d):
                wp = c + offset * n
                if gi == 0 and np.dot(wp - start, n) < 0.0:
                    continue
                block.append(wp)
            gate_blocks.append(block)
            prev = c + d * n

        # assemble, inserting detour waypoints between backtracking pairs
        wps = [start]
        for gi in range(len(gate_blocks)):
            wps.extend(gate_blocks[gi])

            # check reversal between this gate's exit and next gate's entry
            if gi < len(gate_blocks) - 1:
                p1 = gate_blocks[gi][-1]            # last (exit) of gate gi
                p2 = gate_blocks[gi + 1][0]         # first (entry) of gate gi+1
                v = p2 - p1
                vn = np.linalg.norm(v)
                if vn > 1e-6:
                    n_i = oriented_normals[gi]
                    cosang = np.clip(np.dot(v, n_i) / vn, -1.0, 1.0)
                    ang = np.degrees(np.arccos(cosang))
                    if ang > angle_thresh:          # backtracking detected
                        c = gates[gi]
                        y_ax, z_ax = y_axes[gi], z_axes[gi]
                        vproj = v - np.dot(v, n_i) * n_i
                        if np.linalg.norm(vproj) < 1e-6:
                            ddir = y_ax
                        else:
                            beta = np.degrees(np.arctan2(np.dot(vproj, z_ax),
                                                         np.dot(vproj, y_ax)))
                            if -90 <= beta < 45:
                                ddir = y_ax
                            elif 45 <= beta < 135:
                                ddir = z_ax
                            else:
                                ddir = -y_ax
                        wps.append(c + detour_dist * ddir)

        wps = np.array(wps)
        #wps = self._avoid_obstacles(wps, obs, safe=0.3)
        #wps = self._avoid_gateframe(wps, obs)
        #wps = self._smooth_waypoints(wps)
        wps = self._avoid_all(wps, obs)
        return wps
        return wps
    
    def _avoid_obstacles(self, wps: np.ndarray, obs: EnvState_t,
                     safe: float = 0.3) -> np.ndarray:
        obstacles = np.asarray(obs.pOLL_array, dtype=float)
        if obstacles.size == 0:
            return wps

        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            return wps
        prelim = CubicSpline(s, wps, axis=0)
        n = max(50, int(s[-1] / 0.02))
        pts = prelim(np.linspace(0.0, s[-1], n))

        for o in obstacles:
            kept = []
            inside = False
            entry_i = None
            for i, p in enumerate(pts):
                d_xy = np.linalg.norm(p[:2] - o[:2])
                if d_xy < safe:
                    if not inside:
                        inside, entry_i = True, i
                else:
                    if inside:
                        inside = False
                        p_in, p_out = pts[entry_i], pts[i]
                        e_in = p_in[:2] - o[:2]
                        e_out = p_out[:2] - o[:2]
                        bis = e_in + e_out
                        nb = np.linalg.norm(bis)
                        if nb < 1e-3:
                            tv = p_out[:2] - p_in[:2]
                            bis = np.array([-tv[1], tv[0]])
                            nb = np.linalg.norm(bis) + 1e-6
                        new_xy = o[:2] + safe * bis / nb
                        new_z = (p_in[2] + p_out[2]) / 2
                        kept.append([new_xy[0], new_xy[1], new_z])
                    kept.append(p)
            if inside:
                kept.append(pts[-1])
            pts = np.array(kept)
        return pts
    
    def setpoint_at(self, t: float, lookahead_t: float = 0.15) -> np.ndarray:
        ts = self.trajectory.timestamps
        tq = min(t + lookahead_t, ts[-1])
        return np.array([np.interp(tq, ts, self.trajectory.positions[:, k])
                         for k in range(3)])

    def get_pos_traj(self) -> np.ndarray:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return self.trajectory.positions

    @property
    def duration(self) -> float:
        return float(self.trajectory.timestamps[-1])
    
    def _avoid_gateframe(self, wps: np.ndarray, obs: EnvState_t) -> np.ndarray:
        gates = np.asarray(obs.pTLL_array, dtype=float)
        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        if idx <= 0:
            return wps
        start = np.asarray(obs.pBLL, dtype=float)
        passed = gates[:idx]
        quats = np.asarray(obs.qTLT_array, dtype=float)[:idx]
        yaws = Rotation.from_quat(quats).as_euler('xyz')[:, 2]

        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            return wps
        prelim = CubicSpline(s, wps, axis=0)
        n = max(50, int(s[-1] / 0.02))
        pts = prelim(np.linspace(0.0, s[-1], n))

        push_r = 0.72 / 2 + self.clearance

        for c, yaw in zip(passed, yaws):
            # SKIP this gate if the drone is currently inside/near it —
            # otherwise we'd shove the trajectory's start point out of a
            # frame the drone is physically occupying (the "weird lurch").
            if np.linalg.norm(start[:2] - c[:2]) < 0.6:
                continue

            kept, inside, entry_i = [], False, None
            for i, p in enumerate(pts):
                if self._check_gate_clearance(p, c, yaw):
                    if not inside:
                        inside, entry_i = True, i
                else:
                    if inside:
                        inside = False
                        p_in, p_out = pts[entry_i], pts[i]
                        bis = (p_in[:2] - c[:2]) + (p_out[:2] - c[:2])
                        nb = np.linalg.norm(bis)
                        if nb < 1e-3:
                            tv = p_out[:2] - p_in[:2]
                            bis = np.array([-tv[1], tv[0]]); nb = np.linalg.norm(bis) + 1e-6
                        new_xy = c[:2] + push_r * bis / nb
                        kept.append([new_xy[0], new_xy[1], (p_in[2] + p_out[2]) / 2])
                    kept.append(p)
            if inside:
                kept.append(pts[-1])
            pts = np.array(kept)
        return pts

    def _check_gate_clearance(self, point, gate_center, yaw) -> bool:
        """True if point is inside the gate FRAME (hazard); False if in the
        opening or clear."""
        half_outer = 0.72 / 2 + self.clearance
        half_open  = 0.40 / 2
        half_thick = 0.1 + self.clearance

        diff = point - gate_center
        lx =  diff[0] * np.cos(yaw) + diff[1] * np.sin(yaw)
        ly = -diff[0] * np.sin(yaw) + diff[1] * np.cos(yaw)
        lz =  diff[2]

        in_depth = abs(lx) < half_thick
        in_outer = (abs(ly) < half_outer) and (abs(lz) < half_outer)
        in_open  = (abs(ly) < half_open)  and (abs(lz) < half_open)
        return in_depth and in_outer and not in_open
    

    def _gate_frames(self, obs: EnvState_t):
        quats = np.asarray(obs.qTLT_array, dtype=float)
        R = Rotation.from_quat(quats).as_matrix()
        return R[:, :, 0], R[:, :, 1], R[:, :, 2]   # normal, y(width), z(height)

    def _smooth_waypoints(self, wps: np.ndarray, window: int = 3) -> np.ndarray:
        """Light moving-average smoothing to remove avoider back-and-forth,
        keeping endpoints fixed."""
        if len(wps) < window + 2:
            return wps
        out = wps.copy()
        half = window // 2
        for i in range(half, len(wps) - half):
            out[i] = wps[i - half:i + half + 1].mean(axis=0)
        out[0] = wps[0]
        out[-1] = wps[-1]
        return out

    def _avoid_all(self, wps: np.ndarray, obs: EnvState_t,
                   obst_safe: float = 0.3) -> np.ndarray:
        obstacles = np.asarray(obs.pOLL_array, dtype=float)
        gates = np.asarray(obs.pTLL_array, dtype=float)
        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        start = np.asarray(obs.pBLL, dtype=float)

        # passed-gate frames as hazards (skip the one we're currently in/near)
        gate_hazards = []
        if idx > 0:
            quats = np.asarray(obs.qTLT_array, dtype=float)[:idx]
            yaws = Rotation.from_quat(quats).as_euler('xyz')[:, 2]
            for c, yaw in zip(gates[:idx], yaws):
                if np.linalg.norm(start[:2] - c[:2]) > 0.6:
                    gate_hazards.append((c, yaw))

        gate_push = 0.72 / 2 + self.clearance

        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            return wps
        prelim = CubicSpline(s, wps, axis=0)
        n = max(50, int(s[-1] / 0.02))
        pts = prelim(np.linspace(0.0, s[-1], n))

        def hazard_at(p):
            """Return (center_xy, push_radius) of the hazard p violates, else None."""
            for o in obstacles:
                if np.linalg.norm(p[:2] - o[:2]) < obst_safe:
                    return o[:2], obst_safe
            for c, yaw in gate_hazards:
                if self._check_gate_clearance(p, c, yaw):
                    return c[:2], gate_push
            return None

        kept = []
        inside = False
        entry_i = None
        entry_haz = None
        for i, p in enumerate(pts):
            haz = hazard_at(p)
            if haz is not None:
                if not inside:
                    inside, entry_i, entry_haz = True, i, haz
                # inside points dropped
            else:
                if inside:
                    inside = False
                    p_in, p_out = pts[entry_i], pts[i]
                    cxy, push = entry_haz                 # hazard we entered
                    e_in = p_in[:2] - cxy
                    e_out = p_out[:2] - cxy
                    bis = e_in + e_out                    # bisector
                    nb = np.linalg.norm(bis)
                    if nb < 1e-3:
                        tv = p_out[:2] - p_in[:2]
                        bis = np.array([-tv[1], tv[0]]); nb = np.linalg.norm(bis) + 1e-6
                    new_xy = cxy + push * bis / nb
                    new_z = (p_in[2] + p_out[2]) / 2
                    kept.append([new_xy[0], new_xy[1], new_z])
                kept.append(p)
        if inside:
            kept.append(pts[-1])
        return np.array(kept)