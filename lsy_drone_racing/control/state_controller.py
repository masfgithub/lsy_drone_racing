"""Controller that follows a point-mass-planner trajectory through the gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.PointMassPlanner import PointMassPlanner, _Gate

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class _StartState:
    """Minimal start-state object: what PointMassPlanner.plan() expects."""
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position
        self.velocity = velocity


class StateController(Controller):
    """State controller following a planned point-mass trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict,
                 config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False

        # max_speed kept low on purpose: the point-mass optimum is far too
        # aggressive for the real drone to track. Raise this gradually.
        self._planner = PointMassPlanner(max_speed=2.0)

        gates = [
            self._gate_from_obs(obs["gates_pos"][i], obs["gates_quat"][i])
            for i in range(len(obs["gates_pos"]))
        ]
        start = _StartState(
            position=np.asarray(obs["pos"], dtype=float).reshape(3),
            velocity=np.asarray(obs["vel"], dtype=float).reshape(3),
        )

        self._trajectory = self._planner.plan(start, gates, None)
        self._t_total = float(self._trajectory.timestamps[-1])
        print(f"planned trajectory: {len(self._trajectory.positions)} "
              f"samples, total time {self._t_total:.3f} s")

    @staticmethod
    def _gate_from_obs(position, quat_xyzw) -> _Gate:
        """Convert one sim gate (position + quaternion) into a planner _Gate."""
        forward = Rotation.from_quat(quat_xyzw).apply([1.0, 0.0, 0.0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        return _Gate(position=np.asarray(position, dtype=float).reshape(3),
                     yaw=yaw)

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        """Sample the planned trajectory at the current time."""
        t = self._tick / self._freq
        if t >= self._t_total:
            self._finished = True

        traj = self._trajectory
        idx = int(np.searchsorted(traj.timestamps, t))
        idx = min(idx, len(traj.positions) - 1)
        des_pos = traj.positions[idx]

        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def step_callback(self, action, obs, reward, terminated, truncated,
                      info) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the planned trajectory and the current setpoint."""
        from crazyflow.sim.visualize import draw_line, draw_points

        traj = self._trajectory
        draw_line(sim, traj.positions, rgba=(0.0, 1.0, 0.0, 1.0))

        idx = int(np.searchsorted(traj.timestamps,
                                  self._tick / self._freq))
        idx = min(idx, len(traj.positions) - 1)
        setpoint = traj.positions[idx].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)