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
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import os
from pathlib import Path
from context_msgs.msg import STCombined, STControl
import argparse


class TrajectoryLogger(Node):
    def __init__(self, file_name="trajectory_log.csv"):
        super().__init__("trajectory_logger")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1,
        )

        # Declare and get the odometry topic parameter
        self.pose_sub = self.create_subscription(
            STCombined, "/ground_truth/combined", self.state_callback, qos_profile
        )
        self.declare_parameter("min_ds", 0.1)
        min_ds = self.get_parameter("min_ds").get_parameter_value().double_value
        self.get_logger().info(f"Using minimum between points: {min_ds} m")

        self.prev_time = self.get_clock().now().seconds_nanoseconds()
        self.prev_time = self.prev_time[0] + self.prev_time[1] * 1e-9
        self.prev_vx = None

        self.timestamps = []
        self.xs = []
        self.ys = []
        self.zs = []
        self.vxs = []
        self.vys = []
        self.vzs = []
        self.axs = []
        self.omega_xs = []
        self.omega_ys = []
        self.omega_zs = []
        self.min_num_points = 100
        self.loop_back_threshold = 0.5

        self.output_dir = Path(os.getcwd()) / "trajectory_logs"
        self.output_dir.mkdir(exist_ok=True)
        self.save_file_name = file_name

        self.get_logger().info("Trajectory logger node started.")

    def state_callback(self, msg):
        self.get_logger().debug("Received odometry message.")
        if msg is None:
            self.get_logger().warn("Received empty odometry message.")
            return

        # [X, Y, V, YAW, YAW_RATE, SLIP_ANGLE]
        x = msg.state.x
        y = msg.state.y
        z = msg.state.z

        if self.xs and self.ys:
            prev_x = self.xs[-1]
            prev_y = self.ys[-1]
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if (
                distance
                < self.get_parameter("min_ds").get_parameter_value().double_value
            ):
                return

        # If we have atleast num_points and we looped back to the start, shutdown
        if (
            len(self.xs) >= self.min_num_points
            and np.linalg.norm([x - self.xs[0], y - self.ys[0]])
            < self.loop_back_threshold
        ):
            # Terminate the node
            self.get_logger().info("Looped back to start. Shutting down...")
            raise KeyboardInterrupt  # Hacky way to stop the node, but it works

        vx = msg.state.velocity * np.cos(msg.state.slip_angle)
        vy = msg.state.velocity * np.sin(msg.state.slip_angle)
        vz = msg.state.velocity_z

        omega_x = msg.state.roll_rate
        omega_y = msg.state.pitch_rate
        omega_z = msg.state.yaw_rate

        now = self.get_clock().now().seconds_nanoseconds()
        now = now[0] + now[1] * 1e-9

        self.timestamps.append(now)
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.vxs.append(vx)
        self.vys.append(vy)
        self.vzs.append(vz)
        self.axs.append(
            (vx - self.prev_vx) / (now - self.prev_time)
            if self.prev_vx is not None
            else 0
        )

        self.prev_time = now
        self.prev_vx = vx

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        r = R.from_quat([x, y, z, w])
        return r.as_euler("zyx", degrees=False)[0]

    def save_data(self):
        if len(self.xs) < 2:
            self.get_logger().warn("Not enough data to save.")
            return

        try:
            x_np = np.array(self.xs)
            y_np = np.array(self.ys)
            z_np = np.array(self.zs)
            vx_np = np.array(self.vxs)
            ax_np = np.array(self.axs)

            dx = np.gradient(x_np, edge_order=2)
            dy = np.gradient(y_np, edge_order=2)
            dz = np.gradient(z_np, edge_order=2)
            s_np = np.sqrt(dx**2 + dy**2).cumsum()

            # Calculate heading
            psi_np = np.arctan2(dy, dx)

            # Calculate pitch
            theta_np = np.arctan2(dz, dx)
            # Calculate roll
            phi_np = np.arctan2(dz, dy)            

            # Calculate curvature
            # For more stable gradients, extend x and y by two (edge_order 2) elements on each side
            # The elements are taken from the other side of the array
            x = x_np.copy()
            y = y_np.copy()
            x_extended = np.concatenate((x[-2:], x, x[:2]))
            y_extended = np.concatenate((y[-2:], y, y[:2]))
            dx_dt = np.gradient(x_extended, edge_order=2)
            dy_dt = np.gradient(y_extended, edge_order=2)
            d2x_dt2 = np.gradient(dx_dt, edge_order=2)
            d2y_dt2 = np.gradient(dy_dt, edge_order=2)
            denominator = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
            # Avoid division by zero
            denominator[denominator == 0] = np.finfo(float).eps
            kappa_np = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / denominator

            # CAlculate teh
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

            with open(output_path, "w") as f:
                f.write("#\n")
                f.write("#\n")
                f.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; z_m; theta_rad; phi_rad; vy_mps; vz_mps\n")
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
        self.get_logger().info("Shutting down. Saving trajectory...")
        self.save_data()
        super().destroy_node()


def main(args=None):
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
