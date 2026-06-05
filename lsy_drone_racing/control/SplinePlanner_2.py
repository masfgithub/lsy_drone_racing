"""Subclass definition of Splineplanner."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState_t
from lsy_drone_racing.control.planner import Planner, Trajectory


class SplinePlanner(Planner):
    """Class to generate smooth Drone Trajectory for MPC."""
    
    def __init__(
        self,
        obs: EnvState_t,
        info: dict,
        config: dict,
        t_total: float,
        max_speed: float = 2.0
    ):
        """Initialize SplinePlanner.
        
        Args:
            obs:        Initial observation.
            info:       Additional environment information.
            config:     Environment configuration.
            t_total:    Assumed total time for the trajectory.
            max_speed:  Maximum assumed speed the drone can reach.
        """
        super().__init__(obs, info, config)
        self._t_total = t_total
        self.max_speed = max_speed

    def plan(
            self,
            obs: EnvState_t,
            info: dict,
            config: dict,
            t_elapsed: float
    ) -> Trajectory:
        """Function called at the initilazion of the drone racing pipline.
        
        Args:
            obs:                Current Observed environment.
            info:               Additional Environment Information.
            config:             Environment configuration.
            t_elapsed:          Time passed so far.
            
        Returns:
            trajectory:         pos, vel, time in a trajectory class.
        """
        # Create Waypoints with designated function
        p_WLL_array = self._build_waypoints(obs)
        
        # Cubic Spline
        spline_ref_array, t_sample = self._create_spline(p_WLL_array, t_elapsed)

        # Call Trajectory Class
        p_ref_array = spline_ref_array(t_sample)
        v_ref_array = spline_ref_array(t_sample, nu=1)
        self.trajectory = Trajectory(p_ref_array, v_ref_array, t_sample)

        return self.trajectory

    def _build_waypoints(
            self,
            obs: EnvState_t
    ) -> np.ndarray:
        """Creates waypoints to avoid hindrances and complete gates.
        
        Args:
            obs:                Observed environment states.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Current dron position
        pDLL = obs.pBLL

        # Read out gates
        pGLL_array, y_GBL_array = self._gate(obs)

        # Parameter defined to set helping points in front and behind the gates
        Distance = 0.6

        # Create waypoint matrix
        p_WLL_array = pDLL

        pPrevLL = np.zeros(3)
        pNextLL = np.zeros(3)

        for i in range(len(pGLL_array)):
            pPrevLL[0] = pGLL_array[i,0] - Distance*np.cos(y_GBL_array[i])
            pPrevLL[1] = pGLL_array[i,1] - Distance*np.sin(y_GBL_array[i])
            pPrevLL[2] = pGLL_array[i,2]

            p_WLL_array = np.vstack([p_WLL_array, pPrevLL])

            p_WLL_array = np.vstack([p_WLL_array, pGLL_array[i]])
            
            pNextLL[0] = pGLL_array[i,0] + Distance*np.cos(y_GBL_array[i])
            pNextLL[1] = pGLL_array[i,1] + Distance*np.sin(y_GBL_array[i])
            pNextLL[2] = pGLL_array[i,2]

            p_WLL_array = np.vstack([p_WLL_array, pNextLL])

        #p_WLL_array = self._avoid_hindrance(obs, p_WLL_array)

        return p_WLL_array
    
    def _create_spline(
            self,
            p_WLL_array: np.ndarray,
            t_elapsed: float
    ) -> tuple[CubicSpline, np.ndarray]:
        """Creates a Cubic spline.
        
        Arg:
            p_WLL_array:            Waypoints the Spline has to bend around.
            t_elapsed:              Time elapsed in the race.
        """
        # Compute initial times at gates and time samples needed for the remaining time
        t_remaining = self._t_total - t_elapsed
        t_gates = np.linspace(0, t_remaining, len(p_WLL_array))
        t_sample = np.linspace(0, t_remaining, int(np.round(t_remaining*self.freq)))
        
        # Cubic Spline
        spline_ref_array = CubicSpline(t_gates, p_WLL_array, axis=0)

        return spline_ref_array, t_sample
    
    def _avoid_hindrance(
            self,
            obs: EnvState_t,
            p_WLL_array: np.ndarray,
            t_elapsed: float
            ) -> np.ndarray:
        """Check if current waypoints hit obsticles or gate frames and replan.
        
        Args:
            obs:                Environment state observation.
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
            t_elapsed:          Time passed during the race.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Build spline
        spline_test_array, t_sample = self._create_spline(p_WLL_array, t_elapsed)

        p_test_array = spline_test_array(t_sample)
        
        return None