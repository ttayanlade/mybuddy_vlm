"""ROS 2 bridge that runs INSIDE the Isaac Sim process.

It makes the simulator look exactly like the real robot stack on the wire:

  publishes  <ns>/camera/color/image_raw          rgb8
             <ns>/camera/depth/image_rect_raw     32FC1 metres, aligned to colour,
                                                  with D435-style noise + dropout
             <ns>/camera/color/camera_info
             <ns>/wrist_camera/{color,depth}/...  eye-in-hand
             <ns>/mybuddy/joint_states
             /clock                               (sim time)
  subscribes <ns>/mybuddy/joint_trajectory_controller/joint_cmd   trajectory_msgs/JointTrajectory
             <ns>/mybuddy/gripper_cmd                             std_msgs/Float64
             <ns>/sim/reset                                       std_msgs/Empty

Trajectories are interpolated against sim time and applied as joint position
targets on the compliant implicit PD drives (like ros2_control's
joint_trajectory_controller does on hardware).
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Empty, Float64
from trajectory_msgs.msg import JointTrajectory


SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)


def _stamp(t: float) -> Time:
    s = int(t)
    return Time(sec=s, nanosec=int((t - s) * 1e9))


def _img(arr: np.ndarray, enc: str, t: float, frame: str) -> Image:
    m = Image()
    m.header.stamp, m.header.frame_id = _stamp(t), frame
    m.height, m.width = arr.shape[:2]
    m.encoding, m.is_bigendian = enc, 0
    arr = np.ascontiguousarray(arr)
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


def d435_depth_noise(depth: np.ndarray, k: float, dropout: float, rng: np.random.Generator) -> np.ndarray:
    """Axial noise sigma = k*z^2 (stereo depth error grows quadratically) and
    random zero-valued holes, roughly matching a D435 at 0.3–1 m."""
    d = depth.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    d[valid] += rng.normal(0.0, 1.0, valid.sum()).astype(np.float32) * k * d[valid] ** 2
    holes = rng.random(d.shape) < dropout
    d[holes | ~valid] = 0.0
    return d


class SimBridge(Node):
    def __init__(self, cfg: dict, joint_names: list[str], gripper_joint_names: list[str]):
        prof = cfg["ros"]["profiles"]["sim"]
        super().__init__("isaac_sim_bridge", namespace=prof["namespace"] or None)
        self.cfg = cfg
        t = cfg["ros"]["topics"]
        c = cfg["camera"]
        self.joint_names = joint_names
        self.gripper_joint_names = gripper_joint_names
        self.rng = np.random.default_rng(0)
        self._lock = threading.Lock()
        self._traj = None            # (t0, times[N], positions[N,J], name->col)
        self._gripper = None
        self.reset_requested = False

        self.pub_color = self.create_publisher(Image, t["color"], SENSOR_QOS)
        self.pub_depth = self.create_publisher(Image, t["depth"], SENSOR_QOS)
        self.pub_info = self.create_publisher(CameraInfo, t["camera_info"], SENSOR_QOS)
        self.pub_wcolor = self.create_publisher(Image, t["wrist_color"], SENSOR_QOS)
        self.pub_wdepth = self.create_publisher(Image, t["wrist_depth"], SENSOR_QOS)
        self.pub_js = self.create_publisher(JointState, t["joint_states"], 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.create_subscription(JointTrajectory, t["joint_cmd"], self._on_traj, 10)
        self.create_subscription(Float64, t["gripper_cmd"], self._on_gripper, 10)
        self.create_subscription(Empty, "sim/reset", lambda _: setattr(self, "reset_requested", True), 1)

        self.info = CameraInfo()
        self.info.width, self.info.height = c["width"], c["height"]
        self.info.distortion_model = "plumb_bob"
        self.info.d = [0.0] * 5
        self.info.k = [c["fx"], 0.0, c["cx"], 0.0, c["fy"], c["cy"], 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        self.info.p = [c["fx"], 0.0, c["cx"], 0.0, 0.0, c["fy"], c["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info.header.frame_id = "camera_color_optical_frame"
        self.sim_time = 0.0
        self.get_logger().info(f"sim bridge up in namespace '{self.get_namespace()}'")

    # ---------------------------------------------------------------- inputs
    def _on_traj(self, msg: JointTrajectory):
        if not msg.points:
            return
        cols = {n: i for i, n in enumerate(msg.joint_names)}
        missing = [n for n in self.joint_names if n not in cols]
        if missing:
            self.get_logger().error(f"trajectory missing joints {missing}")
            return
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
        pos = np.array([[p.positions[cols[n]] for n in self.joint_names] for p in msg.points])
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            # zero stamp => start now (ros2_control semantics)
            self._traj = (st if st > 0 else self.sim_time, times, pos)
        self.get_logger().info(f"trajectory: {len(times)} pts over {times[-1]:.2f}s")

    def _on_gripper(self, msg: Float64):
        with self._lock:
            self._gripper = float(np.clip(msg.data, 0.0, 1.0))

    def joint_targets(self, sim_time: float, current: np.ndarray) -> np.ndarray | None:
        """Linear interpolation of the active trajectory at sim_time. Before the
        first point we blend from the current position (implicit start point)."""
        with self._lock:
            tr = self._traj
        if tr is None:
            return None
        t0, times, pos = tr
        t = sim_time - t0
        if t <= times[0]:
            a = 0.0 if times[0] <= 0 else max(t, 0.0) / times[0]
            return current + a * (pos[0] - current) if t >= 0 else None
        if t >= times[-1]:
            return pos[-1]
        i = int(np.searchsorted(times, t))
        a = (t - times[i - 1]) / (times[i] - times[i - 1])
        return pos[i - 1] + a * (pos[i] - pos[i - 1])

    def gripper_target(self) -> float | None:
        with self._lock:
            return self._gripper

    # --------------------------------------------------------------- outputs
    def publish_clock(self, sim_time: float):
        self.sim_time = sim_time
        self.pub_clock.publish(Clock(clock=_stamp(sim_time)))

    def publish_joint_states(self, sim_time: float, q: np.ndarray, qd: np.ndarray):
        m = JointState()
        m.header.stamp = _stamp(sim_time)
        m.name = list(self.joint_names)
        m.position = [float(x) for x in q]
        m.velocity = [float(x) for x in qd]
        self.pub_js.publish(m)

    def publish_camera(self, sim_time: float, rgb: np.ndarray, depth: np.ndarray, wrist: bool = False):
        c = self.cfg["camera"]
        depth = d435_depth_noise(np.squeeze(depth), c["sim_depth_noise_k"], c["sim_depth_dropout"], self.rng)
        frame = "wrist_camera_color_optical_frame" if wrist else "camera_color_optical_frame"
        (self.pub_wcolor if wrist else self.pub_color).publish(_img(rgb[..., :3].astype(np.uint8), "rgb8", sim_time, frame))
        (self.pub_wdepth if wrist else self.pub_depth).publish(_img(depth, "32FC1", sim_time, frame))
        if not wrist:
            self.info.header.stamp = _stamp(sim_time)
            self.pub_info.publish(self.info)

    def clear_trajectory(self):
        with self._lock:
            self._traj = None


def spin_bridge(node: Node):
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    return ex
