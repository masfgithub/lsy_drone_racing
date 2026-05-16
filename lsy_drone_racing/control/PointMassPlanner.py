"""Point-mass trajectory planner for the drone racing project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from lsy_drone_racing.control.planner import (
    DEFAULT_MAX_SPEED, Planner, PlanningError, Trajectory)

__all__ = ["PointMassPlanner"]

# Module-level constants.
GRAVITY = 9.81                  # m/s^2
DEFAULT_MASS = 0.8              # kg
DEFAULT_THRUST_TO_WEIGHT = 2.5  # dimensionless


@dataclass
class _Node:
    """One node in the velocity search graph: a gate reached at some velocity.

    ``position`` is the gate's fixed location; ``velocity`` is the sampled
    pass-through velocity this node represents.
    """
    gate_index: int
    position: np.ndarray   # shape (3,)
    velocity: np.ndarray   # shape (3,)


class _AxisSolution(NamedTuple):
    """Closed-form bang-bang result for one axis — fully self-contained:
    everything needed to evaluate equation (6) at any time t."""
    t1: float       # first-phase (switch) duration
    t2: float       # second-phase duration
    p1: float       # position at the switch
    v1: float       # velocity at the switch
    u_acc: float    # signed acceleration, phase 1  (scaled if slowed)
    u_brake: float  # signed acceleration, phase 2  (scaled if slowed)

@dataclass
class _Gate:
    """A gate normalized into the planner's own vocabulary.

    Created once in plan() from whatever the sim hands over, so the
    planner's interior never depends on the sim's data format.
    """
    position: np.ndarray   # shape (3,) — gate centre
    yaw: float             # facing direction in the xy-plane, radians


class PointMassPlanner(Planner):
    """Time-optimal planner that models the drone as a point mass.

    Rotational dynamics are ignored: the quadrotor is a point with bounded
    acceleration. The planner samples candidate velocities at each gate,
    connects them with closed-form bang-bang segments, and runs a
    shortest-path search to find the minimum-time sequence.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, mass: float = DEFAULT_MASS,
                 thrust_to_weight: float = DEFAULT_THRUST_TO_WEIGHT,
                 max_speed: float = DEFAULT_MAX_SPEED,
                 samples_per_gate: int = 27) -> None:
        super().__init__()
        self.mass = mass
        self.thrust_to_weight = thrust_to_weight
        self.max_speed = max_speed
        self.samples_per_gate = samples_per_gate
        self._last_trajectory: Trajectory | None = None

    @property
    def max_acceleration(self) -> float:
        """Largest acceleration the point mass can sustain, gravity removed."""
        return (self.thrust_to_weight - 1.0) * GRAVITY

    # ------------------------------------------------------------------
    # public interface (the Planner contract)
    # ------------------------------------------------------------------
    def plan(self, start_state, gates, goal_state) -> Trajectory:
        if len(gates) == 0:
            raise PlanningError("cannot plan a trajectory with no gates")

        # 1. normalize sim data into the planner's own vocabulary
        start_node = _Node(gate_index=-1,
                        position=np.asarray(start_state.position),
                        velocity=np.asarray(start_state.velocity))
        gate_objs = [_Gate(position=np.asarray(g.position), yaw=g.yaw)
                    for g in gates]

        # 2. build the velocity graph and find the minimum-time path
        columns = self._build_graph(start_node, gate_objs)
        path = self._shortest_path(columns)

        # 3. turn the winning node sequence into a sampled Trajectory
        return self._build_trajectory(path)

    def reset(self) -> None:
        """Clear cached state between runs."""
        self._last_trajectory = None

    # ------------------------------------------------------------------
    # layer 1: per-axis bang-bang primitive
    # ------------------------------------------------------------------

    def _solve_axis_minimum_time(self, p0: float, pf: float,
                                v0: float, vf: float,
                                u_acc: float, u_brake: float) -> _AxisSolution:
        """Minimum-time solution for one axis, trying both bang-bang orderings.

        ``u_acc`` and ``u_brake`` are the two acceleration bounds (opposite
        signs). This tries accelerate-then-brake AND brake-then-accelerate,
        and returns whichever valid solution has the smaller total time.
        """
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


    @staticmethod
    def _solve_axis(p0: float, pf: float, v0: float, vf: float,
                    u_acc: float, u_brake: float) -> _AxisSolution | None:
        """Closed-form minimum-time solution for one axis.

        Applies acceleration ``u_acc`` then ``u_brake``. Returns ``None`` when
        no real solution exists for this ordering of the bounds.
        """
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

    # ------------------------------------------------------------------
    # layer 2: synchronize the three axes onto a common duration
    # ------------------------------------------------------------------
    def _synchronize_segment(self, p0, pf, v0, vf, u_acc, u_brake) -> float:
        """Minimum synchronized duration T* for one segment.

        T* is just the slowest axis — slowing the faster axes to match does
        not change the maximum. No alpha, no bisection.
        """
        axis_times = []
        for i in range(3):
            sol = self._solve_axis_minimum_time(
                p0[i], pf[i], v0[i], vf[i], u_acc[i], u_brake[i])
            axis_times.append(sol.t1 + sol.t2)
        return max(axis_times)

    def _resolve_segment(self, p0: np.ndarray, pf: np.ndarray,
                        v0: np.ndarray, vf: np.ndarray,
                        u_acc: np.ndarray, u_brake: np.ndarray
                        ) -> list[_AxisSolution]:
        """Full per-axis motion for ONE chosen segment.

        Solves all three axes, then stretches the faster ones to the common
        duration T* via alpha scaling. Returns the three synchronized
        _AxisSolutions — everything needed to sample the segment's points.

        Runs only on the ~5 winning segments, never during the graph search.
        """
        # Step 1: per-axis minimum-time solves (same as _synchronize_segment).
        per_axis = [
            self._solve_axis_minimum_time(
                p0[i], pf[i], v0[i], vf[i], u_acc[i], u_brake[i])
            for i in range(3)
        ]
        axis_times = [s.t1 + s.t2 for s in per_axis]
        segment_time = max(axis_times)

        # Step 2: stretch every axis that finishes early to segment_time.
        resolved: list[_AxisSolution] = []
        for i in range(3):
            if axis_times[i] >= segment_time - 1e-9:
                resolved.append(per_axis[i])          # slowest axis — unchanged
            else:
                stretched = self._adjust_alpha(
                    segment_time, p0[i], pf[i], v0[i], vf[i],
                    u_acc[i], u_brake[i])
                resolved.append(stretched)            # <-- alpha is called HERE
            
        return resolved

    def _adjust_alpha(self, slowest_time, p0, pf, v0, vf, u_acc, u_brake):
        tol = 1e-6
        alpha_low, alpha_high = 0.01, 1.0
        sol = None
        for _ in range(60):                       # fixed iteration count
            alpha = 0.5 * (alpha_low + alpha_high)
            sol = self._solve_axis_minimum_time(p0, pf, v0, vf,
                                u_acc * alpha, u_brake * alpha)
            T_total = sol.t1 + sol.t2
            if abs(T_total - slowest_time) < tol:
                break
            if T_total > slowest_time:
                alpha_low = alpha                 # too slow -> more acceleration
            else:
                alpha_high = alpha
        return sol
    
    def _segment_time(self, node_a: _Node, node_b: _Node) -> float:
        """Minimum time to fly between two graph nodes — one edge weight.

        The thin seam between the graph (nodes) and the math (arrays):
        unpacks the two nodes and delegates to _synchronize_segment.
        """
        u = self.max_acceleration
        u_acc = np.full(3, +u)    # +max accel on every axis
        u_brake = np.full(3, -u)  # -max accel on every axis
        return self._synchronize_segment(
            node_a.position, node_b.position,
            node_a.velocity, node_b.velocity,
            u_acc, u_brake,
        )

    # ------------------------------------------------------------------
    # layer 4: velocity sampling at a gate
    # ------------------------------------------------------------------
    def _sample_velocities(self, gate: _Gate, gate_index: int) -> list[_Node]:
        """Candidate pass-through velocities for one gate — the fan of arrows.

        Samples velocities in a cone around the gate's facing direction
        (cos yaw, sin yaw, 0). Returns one _Node per sample, all sharing the
        gate's fixed position.
        """
        # The direction the drone should travel through the gate.
        facing = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0.0])

        half_angle = np.deg2rad(30.0)   # cone half-width
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
                    # Rotate the facing direction by the yaw/pitch offsets.
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

    # ------------------------------------------------------------------
    # layer 5: graph construction and shortest-path search
    # ------------------------------------------------------------------
    def _build_graph(self, start_node: _Node,
                    gates: list[_Gate]) -> list[list[_Node]]:
        """Build the velocity-graph columns: the start node, then one column
        of sampled velocity nodes per gate."""
        columns: list[list[_Node]] = [[start_node]]
        for i, gate in enumerate(gates):
            columns.append(self._sample_velocities(gate, gate_index=i))
        return columns

    def _shortest_path(self, columns: list[list[_Node]]) -> list[_Node]:
        """Dijkstra over the velocity graph. Edges run only between adjacent
        columns, weighted by _segment_time. Returns one node per column —
        the minimum-time path from the start node through every gate."""
        # cost[c][r] = best known time to reach node r in column c.
        cost = [[float("inf")] * len(col) for col in columns]
        prev = [[None] * len(col) for col in columns]
        for r in range(len(columns[0])):
            cost[0][r] = 0.0

        # Columns are strictly ordered, so a single forward sweep suffices —
        # no priority queue needed.
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

        # Pick the cheapest node in the final column, walk prev back.
        last = len(columns) - 1
        best_r = min(range(len(columns[last])), key=lambda r: cost[last][r])
        path: list[_Node] = []
        c, r = last, best_r
        while r is not None:
            path.append(columns[c][r])
            r = prev[c][r]
            c -= 1
        path.reverse()
        return path

    def _gate_from_sim(position: np.ndarray, quat_xyzw: np.ndarray) -> _Gate:
        """Convert one sim gate (position + quaternion) into a _Gate.

        quat_xyzw must be in (x, y, z, w) order — scipy's convention.
        """
        from scipy.spatial.transform import Rotation
        forward = Rotation.from_quat(quat_xyzw).apply([1.0, 0.0, 0.0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        return _Gate(position=np.asarray(position, dtype=float), yaw=yaw)
    
    # ------------------------------------------------------------------
    # trajectory reconstruction
    # ------------------------------------------------------------------
    def _build_trajectory(self, path: list[_Node],
                          samples_per_segment: int = 20) -> Trajectory:
        """Sample the winning node path into a Trajectory.

        For each segment between consecutive nodes, _resolve_segment gives
        the three synchronized per-axis solutions; equation (6) is then
        evaluated at evenly spaced times to produce position/velocity
        samples.
        """
        u = self.max_acceleration
        u_acc = np.full(3, +u)
        u_brake = np.full(3, -u)

        all_pos: list[np.ndarray] = []
        all_vel: list[np.ndarray] = []
        all_time: list[float] = []
        clock = 0.0

        for seg in range(len(path) - 1):
            node_a, node_b = path[seg], path[seg + 1]
            axes = self._resolve_segment(
                node_a.position, node_b.position,
                node_a.velocity, node_b.velocity,
                u_acc, u_brake,
            )
            seg_time = max(ax.t1 + ax.t2 for ax in axes)

            last = (seg == len(path) - 2)
            ts = np.linspace(0.0, seg_time, samples_per_segment,
                             endpoint=last)
            for t in ts:
                pos = np.empty(3)
                vel = np.empty(3)
                for i in range(3):
                    pos[i], vel[i] = self._evaluate_axis(
                        axes[i], node_a.position[i],
                        node_a.velocity[i], t)
                all_pos.append(pos)
                all_vel.append(vel)
                all_time.append(clock + t)
            clock += seg_time

        return Trajectory(
            positions=np.array(all_pos),
            velocities=np.array(all_vel),
            timestamps=np.array(all_time),
        )

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