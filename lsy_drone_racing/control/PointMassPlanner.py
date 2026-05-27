"""Point-mass trajectory planner"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control.env_obs import EnvState_t

import numpy as np
from lsy_drone_racing.control.planner import (
    DEFAULT_MAX_SPEED, Planner, PlanningError, Trajectory)

__all__ = ["PointMassPlanner"]

# Module-level constants.
GRAVITY = 9.81                  # gravitational acceleration
DEFAULT_MASS = 0.8              # weigth of the drone
DEFAULT_THRUST_TO_WEIGHT = 2.5  # used to calculate the maximum thrust


@dataclass
class _Node:
    gate_index: int # start position: -1, first gate: 0, ...
    position: np.ndarray # xyz coordinates of the note position
    velocity: np.ndarray # velocity vector in xyz


class _AxisSolution(NamedTuple):
    t1: float # acceleration duration
    t2: float # break duration
    p1: float       # position at the switch
    v1: float       # velocity at the switch
    u_acc: float    # maximum acceleration thrust
    u_brake: float  # maximum break thrust

@dataclass
class _Gate:
    position: np.ndarray   # gate centre
    yaw: float             # facing direction in the xy-plane, radians


class PointMassPlanner(Planner):
    # Constructor
    def __init__(self, obs: EnvState_t, info: dict, config: dict, t_total: int,  mass: float = DEFAULT_MASS,
                 thrust_to_weight: float = DEFAULT_THRUST_TO_WEIGHT,
                 max_speed: float = 2.0,
                 samples_per_gate: int = 27) -> None:
        super().__init__()
        self.mass = mass
        self.thrust_to_weight = thrust_to_weight
        self.max_speed = max_speed
        self.samples_per_gate = samples_per_gate
        self._last_trajectory: Trajectory | None = None
        self.t_total = t_total
        self.freq = config.env.freq

    @property
    def max_acceleration(self) -> float:
        """Largest acceleration the point mass can sustain, gravity removed."""
        return (self.thrust_to_weight - 1.0) * GRAVITY

    # Planner to be called in control
    def plan(self, obs: EnvState_t, info: dict | None = None) -> Trajectory:

        gate_objs = []
        for i in range(len(obs.qTLT_array)):
            gate_objs.append(_Gate(obs.pTLL_array[i], R.from_quat(obs.qTLT_array[i]).as_euler("xyz")[2]))

        start_node = _Node(obs.pTLL_index, obs.pBLL, obs.vBLL)

        columns = self._build_graph(start_node, gate_objs)
        path = self._shortest_path(columns)

        return self._build_trajectory(path, self.freq, self.t_total)

    def reset(self) -> None:
        """Clear cached state between runs."""
        self._last_trajectory = None

    # Compute close formed solution to Equation 6 try with +uacc and -uacc return times
    def _solve_axis_minimum_time(self, p0: float, pf: float,
                                v0: float, vf: float,
                                u_acc: float, u_brake: float) -> _AxisSolution:
        candidates = [
            self._solve_axis(p0, pf, v0, vf, u_acc, u_brake),
            self._solve_axis(p0, pf, v0, vf, u_brake, u_acc),
        ]
        valid = [s for s in candidates if s is not None]
        if not valid:
            raise PlanningError(
                f"no bang-bang solution for axis (p0={p0}, pf={pf}, "
                f"v0={v0}, vf={vf})")
        return min(valid, key=lambda s: s.t1 + s.t2)

    # returns closed form solution to Equation 6
    @staticmethod
    def _solve_axis(p0: float, pf: float, v0: float, vf: float,
                    u_acc: float, u_brake: float) -> _AxisSolution | None:
        
        if u_brake == 0.0 or u_acc == u_brake:
            return None  # degenerate: division by zero below
        a = 0.5 * u_acc * (1.0 - u_acc / u_brake)
        if a == 0.0:
            return None
        b = v0 - v0 * u_acc / u_brake
        c = (p0 - pf
             + v0 * (vf - v0) / u_brake
             + 0.5 * (vf - v0) ** 2 / u_brake)
        p = b / a
        q = c / a
        discriminant = (p / 2.0) ** 2 - q
        if discriminant < 0.0:
            return None
        t1 = -p / 2.0 + np.sqrt(discriminant)
        t2 = (vf - v0 - u_acc * t1) / u_brake
        p1 = p0 + v0 * t1 + 0.5 * u_acc * t1 ** 2
        v1 = v0 + u_acc * t1
        return _AxisSolution(t1, t2, p1, v1, u_acc, u_brake)

    # returns the maximum axis time from the computed axis
    def _synchronize_segment(self, p0, pf, v0, vf, u_acc, u_brake) -> float:
        
        axis_times = []
        for i in range(3):
            sol = self._solve_axis_minimum_time(
                p0[i], pf[i], v0[i], vf[i], u_acc[i], u_brake[i])
            axis_times.append(sol.t1 + sol.t2)
        return max(axis_times)

    # adjusts alpha to match the faster axis the slowest axis
    def _resolve_segment(self, p0: np.ndarray, pf: np.ndarray,
                        v0: np.ndarray, vf: np.ndarray,
                        u_acc: np.ndarray, u_brake: np.ndarray
                        ) -> list[_AxisSolution]:
        
        # compute the axis times and find max axis
        per_axis = [
            self._solve_axis_minimum_time(
                p0[i], pf[i], v0[i], vf[i], u_acc[i], u_brake[i])
            for i in range(3)
        ]
        axis_times = [s.t1 + s.t2 for s in per_axis]
        segment_time = max(axis_times)
        
        # adjust alpha to slow faster axis down
        resolved: list[_AxisSolution] = []
        for i in range(3):
            if axis_times[i] >= segment_time - 1e-9: # finds the slowest axis
                resolved.append(per_axis[i])
            else:
                stretched = self._adjust_alpha(
                    segment_time, p0[i], pf[i], v0[i], vf[i],
                    u_acc[i], u_brake[i])
                resolved.append(stretched)
            
        return resolved

    def _adjust_alpha(self, slowest_time, p0, pf, v0, vf, u_acc, u_brake):
        tol = 1e-6
        alpha_low, alpha_high = 0.01, 1.0
        sol = None
        for _ in range(60):
            alpha = 0.5 * (alpha_low + alpha_high)
            sol = self._solve_axis_minimum_time(p0, pf, v0, vf,
                                u_acc * alpha, u_brake * alpha)
            T_total = sol.t1 + sol.t2
            if abs(T_total - slowest_time) < tol:
                break
            if T_total > slowest_time:
                alpha_low = alpha
            else:
                alpha_high = alpha
        return sol
    
    # returns the maximum axis times computed in _synchronize_segment()
    def _segment_time(self, node_a: _Node, node_b: _Node) -> float:
        
        u = self.max_acceleration
        u_acc = np.full(3, +u)    # +max accel on every axis
        u_brake = np.full(3, -u)  # -max accel on every axis
        return self._synchronize_segment(
            node_a.position, node_b.position,
            node_a.velocity, node_b.velocity,
            u_acc, u_brake,
        )

    # sample different velocities at the gate
    def _sample_velocities(self, gate: _Gate, gate_index: int) -> list[_Node]:
        facing = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0.0])

        half_angle = np.deg2rad(30.0)   # cone half opening width
        n_speeds = 3                    # speed levels per direction
        n_yaw = 3                       # yaw offsets across the cone
        n_pitch = 3                     # pitch offsets across the cone

        speeds = np.linspace(0.3 * self.max_speed, self.max_speed, n_speeds)
        yaw_offsets = np.linspace(-half_angle, half_angle, n_yaw)
        pitch_offsets = np.linspace(-half_angle, half_angle, n_pitch)

        nodes: list[_Node] = []
        for speed in speeds:
            for d_yaw in yaw_offsets:
                for d_pitch in pitch_offsets:
                    yaw = gate.yaw + d_yaw
                    direction = np.array([
                        np.cos(yaw) * np.cos(d_pitch),
                        np.sin(yaw) * np.cos(d_pitch),
                        np.sin(d_pitch),
                    ])
                    velocity = speed * direction
                    nodes.append(_Node(gate_index=gate_index,
                                        position=gate.position.copy(),
                                        velocity=velocity))
        return nodes

    # create the speed nodes for all gates
    def _build_graph(self, start_node: _Node,
                    gates: list[_Gate]) -> list[list[_Node]]:
        columns: list[list[_Node]] = [[start_node]]
        for i, gate in enumerate(gates):
            columns.append(self._sample_velocities(gate, gate_index=i))
        return columns
    
    # find cheapest notes
    def _shortest_path(self, columns: list[list[_Node]]) -> list[_Node]:

        # init cost array and previous node array
        cost = [[float("inf")] * len(col) for col in columns]
        prev = [[None] * len(col) for col in columns]
        for r in range(len(columns[0])):
            cost[0][r] = 0.0
        
        # compute the cost from start to node for every waypoint and update if a cheaper one was found
        for c in range(len(columns) - 1):
            for r_a, node_a in enumerate(columns[c]):
                if cost[c][r_a] == float("inf"):
                    continue
                for r_b, node_b in enumerate(columns[c + 1]):
                    edge = self._segment_time(node_a, node_b)
                    new_cost = cost[c][r_a] + edge
                    if new_cost < cost[c + 1][r_b]:
                        cost[c + 1][r_b] = new_cost
                        prev[c + 1][r_b] = r_a

        # pick the cheapest path from the last nodes
        last = len(columns) - 1
        best_r = min(range(len(columns[last])), key=lambda r: cost[last][r])
        
        # create path list of nodes with minimal cost
        path: list[_Node] = []
        c, r = last, best_r
        while r is not None:
            path.append(columns[c][r])
            r = prev[c][r]
            c -= 1
        path.reverse()
        return path
    
    # adjust alpha to match the axis times and construct trajectory with the lowest cost sample
    def _build_trajectory(self, path: list[_Node],
                      freq: float,
                      samples_per_segment: int = 100,
                      v_min: float = 0.3,
                      v_max: float = 0.8,
                      angle_sharpness: float = 2.0) -> dict:
        """Build a curvature-shaped trajectory and resample onto a uniform
        time grid at `freq` Hz, so NMPC indexing waypoints[i:i+N] gets the
        reference at exactly tick i.

        freq             : controller frequency in Hz (e.g. config.env.freq)
        v_min            : minimum speed (m/s) in sharpest corners
        v_max            : maximum speed (m/s) in straightest sections
        angle_sharpness  : how aggressively speed drops with angle
        """
        from scipy.interpolate import CubicSpline

        u = self.max_acceleration
        u_acc = np.full(3, +u)
        u_brake = np.full(3, -u)

        # --- 1. Sample geometry from the bang-bang solver (velocities thrown away) ---
        all_pos: list[np.ndarray] = []
        for seg in range(len(path) - 1):
            node_a, node_b = path[seg], path[seg + 1]
            axes = self._resolve_segment(
                node_a.position, node_b.position,
                node_a.velocity, node_b.velocity,
                u_acc, u_brake,
            )
            seg_time = max(ax.t1 + ax.t2 for ax in axes)
            last = (seg == len(path) - 2)
            ts = np.linspace(0.0, seg_time, samples_per_segment, endpoint=last)
            for t in ts:
                pos = np.empty(3)
                for i in range(3):
                    pos[i], _ = self._evaluate_axis(
                        axes[i], node_a.position[i], node_a.velocity[i], t)
                all_pos.append(pos)
        raw_positions = np.array(all_pos)
        n = len(raw_positions)

        # --- 2. Distance to neighbours (per-sample) ---
        # use the average of (dist to prev) and (dist to next) so it's symmetric
        diffs = np.diff(raw_positions, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)        # length n-1
        dist_per_sample = np.zeros(n)
        dist_per_sample[1:-1] = 0.5 * (seg_lengths[:-1] + seg_lengths[1:])
        dist_per_sample[0]    = seg_lengths[0]
        dist_per_sample[-1]   = seg_lengths[-1]

        # normalize distance to [0, 1]
        d_max = dist_per_sample.max()
        if d_max < 1e-9:
            dist_factor = np.zeros(n)
        else:
            dist_factor = dist_per_sample / d_max

        # --- 3. Turning angle (per-sample) ---
        angles = np.zeros(n)
        for i in range(1, n - 1):
            v_in = raw_positions[i] - raw_positions[i - 1]
            v_out = raw_positions[i + 1] - raw_positions[i]
            n_in = np.linalg.norm(v_in)
            n_out = np.linalg.norm(v_out)
            if n_in < 1e-9 or n_out < 1e-9:
                continue
            cos_a = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
            angles[i] = np.arccos(cos_a)
        if n > 1:
            angles[0] = angles[1]
            angles[-1] = angles[-2]

        # smooth single-sample kinks
        window = 11
        kernel = np.ones(window) / window
        angles = np.convolve(angles, kernel, mode='same')

        # angle penalty: 1 in straights, decays to 0 in sharp turns
        angle_factor = np.exp(-angle_sharpness * angles)

        # --- 4. Combine: speed = v_min + (v_max - v_min) * dist_factor * angle_factor ---
        # dist_factor pushes speed up where samples are spread out (long straights),
        # angle_factor pulls it back down where the path bends sharply.
        combined = dist_factor * angle_factor
        speeds = v_min + (v_max - v_min) * combined
        # safety clamp
        speeds = np.clip(speeds, v_min, v_max)

        # --- 5. Raw timestamps from per-segment travel time ---
        # dt per segment = segment_length / average_speed
        avg_speed = np.maximum(0.5 * (speeds[:-1] + speeds[1:]), 1e-6)
        dt = seg_lengths / avg_speed
        raw_timestamps = np.concatenate(([0.0], np.cumsum(dt)))

        eps = 1e-9
        for i in range(1, len(raw_timestamps)):
            if raw_timestamps[i] <= raw_timestamps[i - 1]:
                raw_timestamps[i] = raw_timestamps[i - 1] + eps

        total_time = float(raw_timestamps[-1])

        # --- 6. Velocity vectors (speed * unit tangent) ---
        raw_velocities = np.zeros_like(raw_positions)
        for i in range(n):
            if i == 0:
                tangent = raw_positions[1] - raw_positions[0]
            elif i == n - 1:
                tangent = raw_positions[-1] - raw_positions[-2]
            else:
                tangent = raw_positions[i + 1] - raw_positions[i - 1]
            norm = np.linalg.norm(tangent)
            if norm < 1e-9:
                continue
            raw_velocities[i] = speeds[i] * (tangent / norm)

        # --- 7. Splines on the raw irregular grid ---
        des_pos_spline = CubicSpline(raw_timestamps, raw_positions, axis=0)
        des_vel_spline = CubicSpline(raw_timestamps, raw_velocities, axis=0)

        # --- 8. Resample onto uniform 1/freq grid for the NMPC ---
        n_ticks = int(np.ceil(total_time * freq)) + 1
        uniform_t = np.arange(n_ticks) / freq
        uniform_t[-1] = min(uniform_t[-1], total_time)

        waypoints_pos = des_pos_spline(uniform_t)
        waypoints_vel = des_vel_spline(uniform_t)

        # cache for get_pos_traj() etc.
        self.des_pos_spline = des_pos_spline
        self.des_vel_spline = des_vel_spline
        self.waypoints_pos = waypoints_pos
        self.waypoints_vel = waypoints_vel
        self.t_total = total_time



        return {
            "des_pos_spline": des_pos_spline,
            "des_vel_spline": des_vel_spline,
            "waypoints_pos": waypoints_pos,
            "waypoints_vel": waypoints_vel,
            "timestamps": uniform_t,
            "total_time": total_time,
        }

    @staticmethod
    def _evaluate_axis(axis: _AxisSolution, p0: float, v0: float,
                       t: float) -> tuple[float, float]:
        """Evaluate one axis of equation (6) at time t -> (position, velocity)."""
        if t <= axis.t1:
            pos = p0 + v0 * t + 0.5 * axis.u_acc * t * t
            vel = v0 + axis.u_acc * t
        else:
            dt = t - axis.t1
            pos = axis.p1 + axis.v1 * dt + 0.5 * axis.u_brake * dt * dt
            vel = axis.v1 + axis.u_brake * dt
        return pos, vel
    
    def get_pos_traj(self) -> np.ndarray:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return self.des_pos_spline(np.linspace(0, self.t_total, 100))