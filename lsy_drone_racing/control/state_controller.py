"""Controller driving the SplinePlanner (replans only when triggered)."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.env_obs import extract_env_states
from lsy_drone_racing.control.SplinePlanner import SplinePlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._t_total = 12
        

        env_states = extract_env_states(obs)
        self.old_env = env_states
        self._planner = SplinePlanner(env_states, info, config,
                                      self._t_total, max_speed=2.0)
        self._trajectory = self._planner.trajectory
        self._setpoint = env_states.pBLL.copy()


    def compute_control(self, obs, info=None):
        env_states = extract_env_states(obs)
        t = self._tick / self._freq
        if self._should_replan(env_states):
            self._trajectory = self._planner.replan(env_states, t)
        des_pos = self._planner.setpoint_at(t).copy()
        des_pos[2] = max(des_pos[2], 0.1)
        if int(np.atleast_1d(env_states.pTLL_index).ravel()[0]) < 0:
            self._finished = True
        if t >= self._planner.duration:
            self._finished = True
        self._setpoint = des_pos
        return np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0

    def render_callback(self, sim: Sim):
        from crazyflow.sim.visualize import draw_line, draw_points
        positions = self._trajectory.positions
        step = max(1, len(positions) // 100)
        draw_line(sim, positions[::step], rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._setpoint.reshape(1, -1),
                    rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        
    def _should_replan(self, obs) -> bool:
        gate_margin = 0.01
        
        old_gates = np.asarray(self.old_env.pTLL_array)
        current_gates = np.asarray(obs.pTLL_array)

        gate_distance = np.linalg.norm(old_gates - current_gates)

        if gate_distance > gate_margin:
            return True
        
        obsticles = np.asarray(self.old_env.pOLL_array)
        
        pos = self._trajectory.positions
        for o in obsticles:
            if np.any(np.linalg.norm(pos[:, :2] - o[:2], axis=1) < 0.2):
                return True
        return False