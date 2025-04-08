import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy, LivelinessPolicy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import os
from pathlib import Path

from f1tenth_gym.envs.track import Track

class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        # Declare and get the odometry topic parameter
        self.declare_parameter("odom_topic", "/odom")
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.declare_parameter("min_ds", 0.1)
        min_ds = self.get_parameter("min_ds").get_parameter_value().double_value
        self.get_logger().info(f"Using minimum between points: {min_ds} m")
        self.get_logger().info(f"Using odometry topic: {odom_topic}")

        qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    durability=DurabilityPolicy.VOLATILE,
                    liveliness=LivelinessPolicy.AUTOMATIC,
                    depth=1)
        
        self.subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile
        )

        self.prev_time = None
        self.prev_vx = None

        self.timestamps = []
        self.xs = []
        self.ys = []
        self.vxs = []

        self.output_dir = Path(os.getcwd()) / "trajectory_logs"
        self.output_dir.mkdir(exist_ok=True)

        self.get_logger().info("Trajectory logger node started.")

    def odom_callback(self, msg):
        self.get_logger().debug("Received odometry message.")
        if msg is None:
            self.get_logger().warn("Received empty odometry message.")
            return
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Check if the point is at least a minimum distance away from the previous point
        if self.xs and self.ys:
            prev_x = self.xs[-1]
            prev_y = self.ys[-1]
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if distance < self.get_parameter("min_ds").get_parameter_value().double_value:
                return
        q = msg.pose.pose.orientation
        vx = msg.twist.twist.linear.x

        now = self.get_clock().now().nanoseconds * 1e-9

        self.timestamps.append(now)
        self.xs.append(x)
        self.ys.append(y)
        self.vxs.append(vx)

        self.prev_time = now
        self.prev_vx = vx

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        r = R.from_quat([x, y, z, w])
        return r.as_euler('zyx', degrees=False)[0]

    def save_data(self):
        if len(self.xs) < 2:
            self.get_logger().warn("Not enough data to save.")
            return

        try:
            x_np = np.array(self.xs)
            y_np = np.array(self.ys)
            vx_np = np.array(self.vxs)

            track = Track.from_refline(x_np, y_np, vx_np)
            psi_np = np.array(track.raceline.yaws)
            ax_np = np.array(track.raceline.axs)

            s_list = []
            kappa_list = []
            for i in range(len(x_np)):
                s, _, _ = track.cartesian_to_frenet(x_np[i], y_np[i], psi_np[i])
                kappa = track.raceline.spline.calc_curvature(s)
                s_list.append(s)
                kappa_list.append(kappa)

            df = pd.DataFrame({
                's_m': s_list,
                'x_m': x_np,
                'y_m': y_np,
                'psi_rad': psi_np,
                'kappa_radpm': kappa_list,
                'vx_mps': vx_np,
                'ax_mps2': ax_np
            })

            output_path = self.output_dir / "trajectory_log.csv"

            with open(output_path, "w") as f:
                f.write("#\n")
                f.write("#\n")
                f.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
            df.to_csv(output_path, sep=';', index=False, header=False, mode='a', float_format='%.6f')

            self.get_logger().info(f"Saved trajectory to: {output_path}")
        except Exception as e:
            self.get_logger().error(f"Error saving trajectory: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down. Saving trajectory...")
        self.save_data()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    print("Starting logger node...")
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()