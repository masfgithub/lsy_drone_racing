"""Top-level controller wiring a planner and an MPC controller together.

`DroneRacingPipeline` selects a planner/controller pair based on `CONTROLLER_TYPE`
("mpccpp", "mpcc", or "nmpc"), builds them from the initial observation, and on
every `compute_control()` call forwards the current observation to the controller
(replanning first if a gate/obstacle moved, for the MPCC++ path). The rest of the
module holds simulator-visualization helpers (drawing gates, obstacles, the
MPCC++ tunnel, etc.) used by the pipeline's `render_callback`.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.drone_racing_pipeline_config import PLANNER_TYPE
from lsy_drone_racing.control.env_obs import extract_env_states
from lsy_drone_racing.control.mpcc.mpccpp import MPCCpp
from lsy_drone_racing.utils.mpccpp_visualizer import render_mpccpp

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState

if PLANNER_TYPE == "Smart":
    from lsy_drone_racing.control.planner.smart_planner import SplinePlanner
elif PLANNER_TYPE == "Simple":
    from lsy_drone_racing.control.planner.spline_planner_2 import SplinePlanner


class DroneRacingPipeline(Controller):
    """Pipeline for drone racing: planner + controller, supporting MPCC++/MPCC/NMPC."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the pipeline."""
        super().__init__(obs, info, config)

        t_total = 12
        env_states = extract_env_states(obs)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        self._force_replan = False

        # Online adaptive planner: re-roots its trajectory at the current
        # drone position and is replanned when a gate/obstacle moves. MPCC++
        # consumes only the path geometry (its own v_theta sets the speed).
        # The planner is time-parameterized and crashes if t_elapsed >= its
        # horizon (t_remaining <= 0), so keep that horizon comfortably above
        # the real flight time. Since MPCC++ ignores the timing, a large
        # value only makes the sampled path denser -- the geometry is identical.
        self._planner = SplinePlanner(env_states, info, config, t_total)
        trajectory = self._planner.plan(env_states, 0.0)
        self._controller = MPCCpp(env_states, trajectory, info, config, t_total)
        # Replan trigger bookkeeping.
        self.nominal_gates_position = env_states.p_tll_array
        self.nominal_obstacles_position = env_states.p_oll_array
        self._t_replan = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
        env_states = extract_env_states(obs)

        # Replan when a gate/obstacle has moved (or after an episode reset).
        # The planner re-roots at the current drone position; the controller
        # then rebuilds its tube and resets theta to 0 (drone = new start).
        if self._force_replan or self._get_replan_reason(env_states):
            self.nominal_gates_position = env_states.p_tll_array
            self.nominal_obstacles_position = env_states.p_oll_array
            trajectory = self._planner.plan(env_states, self._tick / self._freq)
            self._t_replan = self._tick / self._freq
            self._controller.replan_reference(trajectory, env_states)
            self._force_replan = False
            print("[MPCC++] Replanned -> tube + theta reset")
        return self._controller.control(env_states, info)

        # MPCC / NMPC: original flow.
        self._planner.replan()
        return self._controller.control(env_states, info)

    def _get_replan_reason(self, env_states: EnvState) -> bool:
        """True if the active gate or any obstacle moved more than 1 cm."""
        idx = int(getattr(env_states, "p_tll_index", 0))
        if np.linalg.norm(self.nominal_gates_position[idx] - env_states.p_tll_array[idx]) > 0.01:
            return True
        n = min(len(self.nominal_obstacles_position), len(env_states.p_oll_array))
        for i in range(n):
            if (
                np.linalg.norm(self.nominal_obstacles_position[i] - env_states.p_oll_array[i])
                > 0.01
            ):
                return True
        return False

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        self._controller.set_tick(self._tick)
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
        self._controller.reset()
        self._controller.set_tick(self._tick)
        # Force a fresh replan (re-root + tube rebuild) on the next control step.
        self._force_replan = True
        self._t_replan = 0.0

    def render_callback(self, sim: Sim):
        """Visualize the planned path, MPC predictions, gates, and obstacles."""
        render_mpccpp(sim, self._planner, self._controller)
