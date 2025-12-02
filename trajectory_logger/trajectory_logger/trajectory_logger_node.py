import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    DurabilityPolicy,
    LivelinessPolicy,
)
from nav_msgs.msg import Odometry
import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse


class TrajectoryLogger(Node):
    """
    ROS2 node that logs vehicle trajectory data from odometry messages.

    Subscribes to odometry topics and records position, velocity, and acceleration
    data. Automatically saves the trajectory to a CSV file when the node shuts down
    or when the vehicle loops back to the starting position.
    """

    def __init__(self, file_name="trajectory_log.csv"):
        """
        Initialize the trajectory logger node.

        Args:
            file_name (str): Name of the output CSV file to save trajectory data.
                            Defaults to "trajectory_log.csv".
        """
        super().__init__("trajectory_logger")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1,
        )

        # Declare and get the odometry topic parameter
        self.declare_parameter("odom_topic", "/fixposition/odometry")
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.get_logger().info(f"Subscribing to odometry topic: {odom_topic}")

        self.pose_sub = self.create_subscription(
            Odometry, odom_topic, self.state_callback, qos_profile
        )
        self.declare_parameter("min_ds", 0.1)
        min_ds = self.get_parameter("min_ds").get_parameter_value().double_value
        self.get_logger().info(f"Using minimum between points: {min_ds} m")

        # Initialize time tracking for acceleration calculation
        self.prev_time = self.get_clock().now().seconds_nanoseconds()
        self.prev_time = self.prev_time[0] + self.prev_time[1] * 1e-9
        self.prev_vx = None

        # Data storage lists for trajectory points
        self.timestamps = []
        self.xs = []  # X positions (meters)
        self.ys = []  # Y positions (meters)
        self.zs = []  # Z positions (meters)
        self.vxs = []  # X velocities (m/s)
        self.vys = []  # Y velocities (m/s)
        self.vzs = []  # Z velocities (m/s)
        self.axs = []  # X accelerations (m/s²)

        # Loop detection parameters
        self.min_num_points = 100  # Minimum points before checking for loop closure
        self.loop_back_threshold = (
            0.5  # Distance threshold (meters) to detect return to start
        )

        self.output_dir = Path(os.getcwd()) / "trajectory_logs"
        self.output_dir.mkdir(exist_ok=True)
        self.save_file_name = file_name

        self.get_logger().info("Trajectory logger node started.")

    def state_callback(self, msg):
        """
        Callback function for odometry messages.

        Processes incoming odometry data, filters points based on minimum distance,
        detects loop closure, and stores trajectory data.

        Args:
            msg (nav_msgs.msg.Odometry): Odometry message containing pose and twist data.
        """
        self.get_logger().debug("Received odometry message.")
        if msg is None:
            self.get_logger().warn("Received empty odometry message.")
            return

        # Extract position from Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Filter points: skip if too close to previous point (min_ds threshold)
        if self.xs and self.ys:
            prev_x = self.xs[-1]
            prev_y = self.ys[-1]
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if (
                distance
                < self.get_parameter("min_ds").get_parameter_value().double_value
            ):
                return

        # Loop closure detection: if we've collected enough points and returned near start, shutdown
        if (
            len(self.xs) >= self.min_num_points
            and np.linalg.norm([x - self.xs[0], y - self.ys[0]])
            < self.loop_back_threshold
        ):
            self.get_logger().info("Looped back to start. Shutting down...")
            # Raise KeyboardInterrupt to trigger graceful shutdown in main()
            raise KeyboardInterrupt

        # Extract linear velocity from Odometry message
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        # Get current timestamp and convert to seconds
        now = self.get_clock().now().seconds_nanoseconds()
        now = now[0] + now[1] * 1e-9

        # Store trajectory data
        self.timestamps.append(now)
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.vxs.append(vx)
        self.vys.append(vy)
        self.vzs.append(vz)

        # Calculate acceleration from velocity change over time
        self.axs.append(
            (vx - self.prev_vx) / (now - self.prev_time)
            if self.prev_vx is not None
            else 0
        )

        # Update previous values for next iteration
        self.prev_time = now
        self.prev_vx = vx

    def save_data(self):
        """
        Save collected trajectory data to a CSV file.

        Computes derived quantities (path distance, heading, curvature, pitch, roll)
        from the stored position and velocity data, then writes everything to a CSV file
        in the trajectory_logs directory.

        The output CSV includes:
        - s_m: Cumulative path distance (meters)
        - x_m, y_m, z_m: Position coordinates (meters)
        - psi_rad: Heading angle (radians)
        - kappa_radpm: Path curvature (radians per meter)
        - vx_mps, vy_mps, vz_mps: Velocity components (m/s)
        - ax_mps2: Acceleration in x-direction (m/s²)
        - theta_rad: Pitch angle (radians)
        - phi_rad: Roll angle (radians)
        """
        if len(self.xs) < 2:
            self.get_logger().warn("Not enough data to save.")
            return

        try:
            # Convert lists to numpy arrays for efficient computation
            x_np = np.array(self.xs)
            y_np = np.array(self.ys)
            z_np = np.array(self.zs)
            vx_np = np.array(self.vxs)
            ax_np = np.array(self.axs)

            # Calculate spatial derivatives using gradient
            dx = np.gradient(x_np, edge_order=2)
            dy = np.gradient(y_np, edge_order=2)
            dz = np.gradient(z_np, edge_order=2)

            # Calculate cumulative path distance (arc length)
            s_np = np.sqrt(dx**2 + dy**2).cumsum()

            # Calculate heading angle from path tangent
            psi_np = np.arctan2(dy, dx)

            # Calculate pitch angle (elevation)
            theta_np = np.arctan2(dz, dx)

            # Calculate roll angle (lateral tilt)
            phi_np = np.arctan2(dz, dy)

            # Calculate path curvature using extended arrays for stable gradients
            # Extend arrays by wrapping endpoints to improve gradient accuracy at boundaries
            x = x_np.copy()
            y = y_np.copy()
            x_extended = np.concatenate((x[-2:], x, x[:2]))
            y_extended = np.concatenate((y[-2:], y, y[:2]))

            # First and second derivatives for curvature calculation
            dx_dt = np.gradient(x_extended, edge_order=2)
            dy_dt = np.gradient(y_extended, edge_order=2)
            d2x_dt2 = np.gradient(dx_dt, edge_order=2)
            d2y_dt2 = np.gradient(dy_dt, edge_order=2)

            # Curvature formula: kappa = |x'y'' - x''y'| / (x'² + y'²)^(3/2)
            denominator = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
            # Avoid division by zero by replacing zeros with machine epsilon
            denominator[denominator == 0] = np.finfo(float).eps
            kappa_np = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / denominator

            # Create DataFrame with all trajectory parameters
            df = pd.DataFrame(
                {
                    "s_m": s_np,
                    "x_m": x_np,
                    "y_m": y_np,
                    "psi_rad": psi_np,
                    "kappa_radpm": kappa_np[2:-2],
                    "vx_mps": vx_np,
                    "ax_mps2": ax_np,
                    "z_m": z_np,
                    "theta_rad": theta_np,
                    "phi_rad": phi_np,
                    "vy_mps": self.vys,
                    "vz_mps": self.vzs,
                }
            )

            output_path = self.output_dir / self.save_file_name

            # Write CSV header comments
            with open(output_path, "w") as f:
                f.write("#\n")
                f.write("#\n")
                f.write(
                    "# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; z_m; theta_rad; phi_rad; vy_mps; vz_mps\n"
                )

            # Append data to CSV file (header already written above)
            df.to_csv(
                output_path,
                sep=";",
                index=False,
                header=False,
                mode="a",
                float_format="%.6f",
            )

            self.get_logger().info(f"Saved trajectory to: {output_path}")
        except Exception as e:
            self.get_logger().error(f"Error saving trajectory: {e}")

    def destroy_node(self):
        """
        Cleanup method called when node is shutting down.

        Saves collected trajectory data before destroying the node.
        """
        self.get_logger().info("Shutting down. Saving trajectory...")
        self.save_data()
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the trajectory logger node.

    Initializes ROS2, creates the trajectory logger node, and handles graceful shutdown.

    Args:
        args: Command line arguments (typically from sys.argv). Can be None.
    """
    parser = argparse.ArgumentParser(description="Trajectory Logger Node")
    parser.add_argument(
        "--file_name",
        type=str,
        default="trajectory_log.csv",
        help="Name of the output file to save the trajectory data.",
    )
    parsed_args, unknown = parser.parse_known_args()
    if parsed_args.file_name:
        print(f"Using output file: {parsed_args.file_name}")

    rclpy.init(args=args)
    print("Starting logger node...")
    node = TrajectoryLogger(file_name=parsed_args.file_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Node destroyed. Shutdown complete.")


if __name__ == "__main__":
    main()
