"""Point-mass trajectory planner"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control.env_obs import EnvState_t

import numpy as np
from lsy_drone_racing.control.Planner.planner import (
    DEFAULT_MAX_SPEED, Planner, PlanningError, Trajectory)

__all__ = ["PointMassPlanner"]

# Module-level constants.
DEFAULT_THRUST = 10
HORIZON = 2


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
    def __init__(self, obs: EnvState_t, info: dict, config: dict, t_total: int,
                 thrust: float = DEFAULT_THRUST,
                 max_speed: float = 2.0,
                 samples_per_gate: int = 27,
                 horizon: int = HORIZON) -> None:
            super().__init__()
            self.thrust = thrust
            self.max_speed = max_speed
            self.samples_per_gate = samples_per_gate
            self._last_trajectory: Trajectory | None = None
            self.t_total = t_total
            self.freq = config.env.freq
            self.horizon = horizon

    @property
    def max_acceleration(self) -> float:
        """Largest acceleration the point mass can sustain, gravity removed."""
        return self.thrust

    # Planner to be called in control
    def plan(self, obs: EnvState_t, time: float, info: dict | None = None) -> Trajectory:
        return self._replan(obs, time, info=info)

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
            return None
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

        root = np.sqrt(discriminant)
        best = None
        for t1 in (-p / 2.0 + root, -p / 2.0 - root):
            if t1 < -1e-9:
                continue                                  # negative accel phase -> invalid
            t2 = (vf - v0 - u_acc * t1) / u_brake
            if t2 < -1e-9:
                continue                                  # negative brake phase -> invalid
            cand = _AxisSolution(
                t1, t2,
                p0 + v0 * t1 + 0.5 * u_acc * t1 ** 2,      # p1
                v0 + u_acc * t1,                           # v1
                u_acc, u_brake)
            if best is None or (t1 + t2) < (best.t1 + best.t2):
                best = cand
        return best

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
                    seg = self._segment_points(node_a, node_b)
                    if not self._clearance(seg):
                        continue   # segment hits a pole -> skip this edge
                    if not all(self._point_clear_of_gates(p) for p in seg):
                        continue   # segment hits a gate post -> skip this edge
                    edge = self._segment_time(node_a, node_b)
                    new_cost = cost[c][r_a] + edge
                    if new_cost < cost[c + 1][r_b]:
                        cost[c + 1][r_b] = new_cost
                        prev[c + 1][r_b] = r_a

        n_pole = n_post = n_solve = 0
        for c in range(len(columns) - 1):
            for r_a, node_a in enumerate(columns[c]):
                if cost[c][r_a] == float("inf"):
                    continue
                for r_b, node_b in enumerate(columns[c + 1]):
                    try:
                        seg = self._segment_points(node_a, node_b)
                    except PlanningError:
                        n_solve += 1
                        continue
                    if not self._clearance(seg):
                        n_pole += 1
                        continue
                    if not all(self._point_clear_of_gates(p) for p in seg):
                        n_post += 1
                        continue
                    edge = self._segment_time(node_a, node_b)
                    new_cost = cost[c][r_a] + edge
                    if new_cost < cost[c + 1][r_b]:
                        cost[c + 1][r_b] = new_cost
                        prev[c + 1][r_b] = r_a

        # after the loops, before reconstruction:
        print(f"REJECTS pole={n_pole} post={n_post} solve={n_solve}", flush=True)

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
    def _build_trajectory(self, path: list[_Node], time, 
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

        if len(path) < 2:
            if self._last_trajectory is not None:
                return self._last_trajectory
            raise PlanningError("no feasible path found and no previous trajectory")

        u = self.max_acceleration
        u_acc = np.full(3, +u)
        u_brake = np.full(3, -u)

        seg_axes = []
        seg_times = []
        for seg in range(len(path) - 1):
            node_a, node_b = path[seg], path[seg + 1]
            axes = self._resolve_segment(
                node_a.position, node_b.position,
                node_a.velocity, node_b.velocity,
                u_acc, u_brake,
            )
            seg_axes.append(axes)
            seg_times.append(max(ax.t1 + ax.t2 for ax in axes))

        # sanity: reject a plan if any segment fails to reach its endpoint
        for i in range(len(seg_axes)):
            end_pos = np.array([self._evaluate_axis(seg_axes[i][j], path[i].position[j],
                                                    path[i].velocity[j], seg_times[i])[0]
                                for j in range(3)])
            if np.linalg.norm(end_pos - path[i + 1].position) > 0.1:
                if self._last_trajectory is not None:
                    return self._last_trajectory
                raise PlanningError("bad segment solve and no previous trajectory")

        seg_times = np.array(seg_times)
        seg_ratio = seg_times / seg_times.sum()

        horizon_time = seg_times.sum()
        n_samples = int(round(self.freq * horizon_time))
        n_samples = max(n_samples, 2)
        print("n_samples:", n_samples, flush=True)
        n_seg = np.round(seg_ratio * n_samples).astype(int)
        n_seg = np.maximum(n_seg, 1)
        all_pos = []
        for i in range(len(seg_times)):

            ts = np.linspace(0.0, seg_times[i], n_seg[i])
            #print('ts during the segment',ts)
            #print('size of ts during the segment',len(ts))
            #print('i',i)
            node_a = path[i]
            axes = seg_axes[i]
            for t in ts:
                pos = np.empty(3)
                for j in range(3):
                    pos[j], _ = self._evaluate_axis(
                        axes[j], node_a.position[j], node_a.velocity[j], t)
                all_pos.append(pos)

        all_pos = np.array(all_pos)

        #all_speed = np.diff(all_pos, axis=0)*self.freq
        all_speed = np.diff(all_pos, axis=0)/(seg_times.sum()/n_samples)

        all_speed = np.append(all_speed, all_speed[-1:], axis=0)

        self._waypoints_vel = all_speed
        self._waypoints_pos = all_pos
        # DEBUG
        print("BUILD: len(path)=", len(path), "n_segments=", len(seg_times), flush=True)

        planner_dict = {
            "waypoints_pos": self._waypoints_pos,
            "waypoints_vel": self._waypoints_vel,
        }
        self._last_trajectory = planner_dict
        return planner_dict
                

        return planner_dict


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
        return self._waypoints_pos

    def _clearance(self, segment: np.ndarray, clearance: float = 0.2) -> bool:
        """Return True if the segment stays clear of every obstacle.

        segment   : array of points along the trajectory, shape (N, 3) or (N, 2)
        clearance : minimum allowed distance to any obstacle (metres)
        """
        for i in range(len(segment)):
            for j in range(len(self.obsticles)):
                distance = np.linalg.norm(segment[i] - self.obsticles[j])
                if distance < clearance:
                    return False
        return True
    
    def _segment_points(self, node_a: _Node, node_b: _Node,
                    n_check: int = 20) -> np.ndarray:
        """Sample xy points along the curved segment between two nodes.

        Returns an array of shape (n_check, 2) -- the trajectory's x and y at
        n_check evenly spaced times from segment start to end.
        """
        u = self.max_acceleration
        u_acc = np.full(3, +u)
        u_brake = np.full(3, -u)

        # The real (alpha-synchronized) segment -- same thing _build_trajectory uses.
        axes = self._resolve_segment(
            node_a.position, node_b.position,
            node_a.velocity, node_b.velocity,
            u_acc, u_brake,
        )
        seg_time = max(ax.t1 + ax.t2 for ax in axes)

        ts = np.linspace(0.0, seg_time, n_check)
        points = np.empty((n_check, 2))
        for k, t in enumerate(ts):
            for j in range(2):                      # x and y only (j=0, j=1)
                points[k, j], _ = self._evaluate_axis(
                    axes[j], node_a.position[j], node_a.velocity[j], t)
        return points
    
    def _point_clear_of_gates(self, point_xy: np.ndarray,
                          half_width: float = 0.15,
                          clearance_x: float = 0.04,
                          clearance_y: float = 0.03) -> bool:
        """True if a single xy point clears every gate's two posts.

        For each gate, rotate the point into the gate's local frame (gate faces +x):
        local[0] = along the fly-through direction
        local[1] = sideways, where the posts are
        A point is too close to a post only if it is within clearance_x along the
        facing axis AND within clearance_y along the sideways axis -- i.e. inside a
        little box around the post, checked per axis instead of as one circle.
        """
        pillars_local = np.array([[0.0, half_width], [0.0, -half_width]])

        for gate in self.gate_objs:
            center = gate.position[:2]
            yaw = gate.yaw
            d = point_xy - center                        # shift gate centre to origin
            c, s = np.cos(yaw), np.sin(yaw)
            local = np.array([ c * d[0] + s * d[1],       # R(-yaw) @ d -> local frame
                            -s * d[0] + c * d[1]])
            for pillar in pillars_local:
                dx = abs(local[0] - pillar[0])            # distance along facing axis
                dy = abs(local[1] - pillar[1])            # distance along sideways axis
                if dx < clearance_x and dy < clearance_y:
                    return False
        return True
    
    def _detour_point(self, gate_from, gate_to, obstacle_xy,
                  clearance: float = 0.3, margin: float = 0.1,
                  behind: float = 0.3):
        
        
        return np.array([1.3, 0.2])

    def _replan(self, obs: EnvState_t, time: float,
                info: dict | None = None) -> Trajectory:
        gate_idx = obs.pTLL_index
        if gate_idx < 0:
            gate_idx = 0

        gate_objs = []
        for i in range(gate_idx, len(obs.qTLT_array)):
            gate_objs.append(_Gate(obs.pTLL_array[i],
                                    R.from_quat(obs.qTLT_array[i]).as_euler("xyz")[2]))

        start_node = _Node(gate_idx, obs.pBLL, obs.vBLL)
        self.obsticles = obs.pOLL_array[:, :2]

        # 1. drop the first gate if we're basically on it
        if len(gate_objs) >= 1:
            dist_to_first = np.linalg.norm(obs.pBLL - gate_objs[0].position)
            if dist_to_first < 0.2:
                gate_objs = gate_objs[1:]

        # 2. slice to horizon
        gate_objs = gate_objs[:self.horizon]

        # 3. insert detour only if the obstacle is actually in the current leg
        if len(gate_objs) >= 2 and len(self.obsticles) > 0:
            a = gate_objs[0].position[:2]
            b = gate_objs[1].position[:2]
            obstacle_xy = self.obsticles[1]

            ab = b - a
            t = np.clip(np.dot(obstacle_xy - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
            closest = a + t * ab
            dist_to_leg = np.linalg.norm(obstacle_xy - closest)

            if dist_to_leg < 0.4:
                detour_xy = self._detour_point(gate_objs[0], gate_objs[1], obstacle_xy)
                toward_next = gate_objs[1].position[:2] - detour_xy
                yaw_detour = np.arctan2(toward_next[1], toward_next[0])
                z_mid = 0.5 * (gate_objs[0].position[2] + gate_objs[1].position[2])
                detour_pos = np.array([detour_xy[0], detour_xy[1], z_mid])
                gate_objs.insert(1, _Gate(detour_pos, yaw_detour))

        self.gate_objs = gate_objs

        columns = self._build_graph(start_node, gate_objs)
        path = self._shortest_path(columns)
        return self._build_trajectory(path, time)