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
        self.t_total = float(t_total)
        self.max_speed = float(max_speed)
        self.freq = config.env.freq          # 50 Hz on this env
        self._speed = None
        self.trajectory = self.plan(obs, info, config, t_total)

    # --- ABC interface --------------------------------------------------
    def plan(self, obs: EnvState_t, info: dict, config: dict,
             t_total: float, t0: float = 0.0) -> Trajectory:
        start = np.asarray(obs.pBLL, dtype=float)
        gates = self._remaining_gates(obs)

        if gates.shape[0] == 0:                       # nothing left → hold
            return self._hold(start)

        waypoints = self._build_waypoints(obs)
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total < 1e-6:
            return self._hold(start)

        if self._speed is None:                       # cruise speed, set ONCE
            self._speed = total / self.t_total
        speed = self._speed

        spline = CubicSpline(s, waypoints, axis=0)
        duration = total / speed
        n = max(2, int(round(duration * self.freq)))  # points = seconds * Hz
        s_samp = np.linspace(0.0, total, n)

        positions = spline(s_samp)
        positions[:, 2] = np.maximum(positions[:, 2], 0.1)
        tangents = spline(s_samp, 1)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        velocities = tangents * speed
        timestamps = t0 + s_samp / speed              # absolute time
        print(f"n={n}  duration={duration:.2f}s  total_len={total:.2f}m")
        return Trajectory(positions, velocities, timestamps)

    # --- per-tick convenience ------------------------------------------
    def replan(self, obs: EnvState_t, elapsed: float = 0.0) -> Trajectory:
        """Full replan from the live state through remaining gates."""
        self.trajectory = self.plan(obs, self.info, self.config, self.t_total, t0=elapsed)
        return self.trajectory

    # --- helpers --------------------------------------------------------
    def _remaining_gates(self, obs: EnvState_t) -> np.ndarray:
        gates = np.asarray(obs.pTLL_array, dtype=float)      # (n, 3)
        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        if idx < 0:                                          # all passed
            return np.empty((0, 3))
        return gates[idx:]

    def _hold(self, pos: np.ndarray) -> Trajectory:
        P = np.repeat(pos[None, :], 2, axis=0)
        return Trajectory(P, np.zeros_like(P),
                          np.array([0.0, 1.0 / self.freq]))
    
    def _gate_normals(self, obs: EnvState_t) -> np.ndarray:
        quats = np.asarray(obs.qTLT_array, dtype=float)
        return Rotation.from_quat(quats).as_matrix()[:, :, 0]   # col0 = normal

    def _build_waypoints(self, obs: EnvState_t) -> np.ndarray:
        start = np.asarray(obs.pBLL, dtype=float)
        gates = self._remaining_gates(obs)
        if gates.shape[0] == 0:
            return start[None, :]

        idx = int(np.atleast_1d(obs.pTLL_index).ravel()[0])
        normals_all = self._gate_normals(obs)
        normals = normals_all[idx:]                  # match remaining gates

        d = 0.6
        prev = start
        wps = [start]
        for c, n in zip(gates, normals):
            n = n / (np.linalg.norm(n) + 1e-9)
            if np.dot(c - prev, n) < 0:
                n = -n
            for offset in (-d, -d/2, 0.0, d/2, d):     # eq. 4, K=5
                wps.append(c + offset * n)
            prev = c + d * n

        wps = np.array(wps)
        wps = self._avoid_obstacles(wps, obs, safe=0.3)
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
        pts = prelim(np.linspace(0.0, s[-1], n))      # dense samples

        for o in obstacles:                            # sequential, like the source
            kept = []
            inside = False
            entry_i = None
            for i, p in enumerate(pts):
                d_xy = np.linalg.norm(p[:2] - o[:2])   # eq. 13
                if d_xy < safe:
                    if not inside:
                        inside, entry_i = True, i
                    # inside points are dropped
                else:
                    if inside:
                        inside = False
                        p_in, p_out = pts[entry_i], pts[i]
                        e_in = p_in[:2] - o[:2]         # eq. 14
                        e_out = p_out[:2] - o[:2]
                        bis = e_in + e_out             # eq. 15
                        nb = np.linalg.norm(bis)
                        if nb < 1e-3:                  # degenerate straight-through
                            tv = p_out[:2] - p_in[:2]  # use travel-perpendicular
                            bis = np.array([-tv[1], tv[0]])
                            nb = np.linalg.norm(bis) + 1e-6
                        new_xy = o[:2] + safe * bis / nb          # eq. 16
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

    @property
    def duration(self) -> float:
        return float(self.trajectory.timestamps[-1])