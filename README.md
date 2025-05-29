# ros_waypoint_logger
A simple ROS2 package that implements a waypoint logger for saving human-driven trajectories.

## Overview
This package consists of two main components:
1. A **ROS2 trajectory logger node** that records vehicle state data during operation
2. A **graphical editor** for modifying and optimizing recorded trajectories

## Trajectory Logger
The trajectory logger node subscribes to vehicle state topics and saves position, orientation, and velocity data to CSV files.

## Graphical Editor
The graphical editor (`graphic_editor/application.py`) provides an interactive interface for editing trajectory data:

### Features
- **Trajectory Visualization**: Displays the racing line in an X-Y position plot and a corresponding velocity profile
- **Interactive Control Points**: Drag points to modify the racing line shape
- **Velocity Editing**: Adjust velocity at any point along the path
- **Velocity Visualization**: Color-coded representation of velocity (green=slow, yellow=medium, red=fast) for both plots
- **Dynamic Resampling**: Change the number of control points using a slider. Add and remove points dynamically with double-click actions.
- **Save Modified Trajectories**: Export the modified trajectory with computed path parameters

### Usage
```bash
python graphic_editor/application.py [path/to/trajectory.csv]
```

### Interaction
- **Position Plot**: 
  - Drag control points to change the racing line
  - Double-click on the line to add new control points
  - Double-click on a control point to remove it
- **Velocity Plot**:
  - Drag velocity points up/down to adjust speed at that position
  - Changes are immediately reflected in the color visualization
- **Controls**:
  - Save Waypoints button: Saves the modified trajectory to a new CSV file
  - Reset Velocities button: Restores the original velocity profile
  - Slider: Adjusts the number of control points

The exported trajectory includes computed parameters such as path distance, position, heading angle, curvature, velocity, and acceleration.

## Authors
- Ahmad Amine
- Nandan Tumu
