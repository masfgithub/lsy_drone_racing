"""Shared configuration constants for the drone racing pipeline."""

# Active planner: "Smart" | "Lightweight"
PLANNER_TYPE = "Smart"

# Basic MPCC++ parameters (for more details, modify the parameters in the constructor
# -> see mpccpp.py)
N_HORIZON = 40  # Prediction horizon (number of nodes)
MU = 8.0  # Weight/Reward for advancement along the path
T_HORIZON = 0.7  # Prediction horizon (seconds)

# Visualization Setup:
DRAW_MPCCPP_TUNNEL = False  # If pred. horizon N is too large, visualization might fail
# (exceed maximum of drawable polygons, hint: reduce N_HORIZON).
DRAW_PLANNER_LINE = True  # Draw the planned trajectory line (from the planner) in the visualizer.
DRAW_MPCCPP_PREDICTION = True  # Draw the predicted trajectory line (from MPCC++) in the visualizer.
DRAW_ENVIRONMENT_SOFT_OBJECTS = False  # Draw the environment obstacles/gates in the visualizer.
