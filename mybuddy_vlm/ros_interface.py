"""Step 5 — ROS 2 abstraction layer.

The reasoning node only ever talks to RELATIVE topic names resolved under a
namespace chosen by --sim / --real. Isaac Sim (sim/ros2_sim_bridge.py) and the
hardware stack (RealSense driver + real/mybuddy_driver_node.py) publish and
subscribe on exactly the same relative names, so nothing above this layer
knows which world it is in.

No cv_bridge dependency: images are converted with numpy directly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, String
from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint

from .geometry import CameraIntrinsics
from .kinematics import JointTrajectory

log = logging.getLogger(__name__)

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=2)


# ----------------------------------------------------------------------------
# Image conversion
# ----------------------------------------------------------------------------
def image_msg_to_numpy(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    dtype, ch = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "16uc1": (np.uint16, 1), "mono16": (np.uint16, 1), "32fc1": (np.float32, 1),
    }[enc]
    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    if msg.is_bigendian != (np.little_endian is False) and itemsize > 1:
        arr = arr.byteswap()
    arr = arr.reshape(msg.height, msg.step // (itemsize * ch), ch)[:, :msg.width]
    if enc.startswith("bgr"):
        arr = arr[..., [2, 1, 0] + ([3] if ch == 4 else [])]
    if ch == 4:
        arr = arr[..., :3]
    return arr[..., 0] if ch == 1 else np.ascontiguousarray(arr)


def numpy_to_image_msg(arr: np.ndarray, encoding: str, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    arr = np.ascontiguousarray(arr)
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def to_duration(t: float) -> Duration:
    sec = int(t)
    return Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray   # raw (uint16 mm on real, float32 m in sim)
    stamp: float


# ----------------------------------------------------------------------------
class RobotIO(Node):
    """Sensors in, trajectories out. Thread-safe latest-value caches."""

    def __init__(self, cfg: dict, mode: str):
        prof = cfg["ros"]["profiles"][mode]
        super().__init__("mybuddy_vlm_reasoner", namespace=prof["namespace"] or None,
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL,
                                                        bool(prof["use_sim_time"]))])
        self.cfg, self.mode = cfg, mode
        t = cfg["ros"]["topics"]
        self.joint_names = list(cfg["robot"]["joint_names"])
        self._lock = threading.Lock()
        self._rgb = self._depth = None
        self._rgb_t = self._depth_t = 0.0
        self._joints: dict[str, float] = {}
        self.camera_info_K: CameraIntrinsics | None = None

        self.create_subscription(Image, t["color"], self._on_rgb, SENSOR_QOS)
        self.create_subscription(Image, t["depth"], self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, t["camera_info"], self._on_info, SENSOR_QOS)
        self.create_subscription(JointState, t["joint_states"], self._on_joints, 10)
        self.traj_pub = self.create_publisher(JointTrajectoryMsg, t["joint_cmd"], 10)
        self.grip_pub = self.create_publisher(Float64, t["gripper_cmd"], 10)
        self.status_pub = self.create_publisher(String, t["status"], 10)
        self.get_logger().info(
            f"[{mode}] ns='{self.get_namespace()}' color={self.resolve_topic_name(t['color'])} "
            f"depth={self.resolve_topic_name(t['depth'])} cmd={self.resolve_topic_name(t['joint_cmd'])}")

    # callbacks -----------------------------------------------------------
    def _stamp(self, msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_rgb(self, msg: Image):
        with self._lock:
            self._rgb, self._rgb_t = image_msg_to_numpy(msg), self._stamp(msg)

    def _on_depth(self, msg: Image):
        with self._lock:
            self._depth, self._depth_t = image_msg_to_numpy(msg), self._stamp(msg)

    def _on_info(self, msg: CameraInfo):
        if self.camera_info_K is None:
            self.camera_info_K = CameraIntrinsics.from_K(msg.k, msg.width, msg.height)
            self.get_logger().info(f"camera_info K: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} "
                                   f"cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}")

    def _on_joints(self, msg: JointState):
        with self._lock:
            self._joints.update(zip(msg.name, msg.position))

    # API -------------------------------------------------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def get_observation(self, timeout: float = 5.0, max_skew: float = 0.1,
                        newer_than: float = 0.0) -> Observation:
        """Latest colour+depth pair whose stamps agree within ``max_skew`` and
        that were captured after ``newer_than`` (so we never reason on a frame
        taken while the arm was still moving)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if (self._rgb is not None and self._depth is not None
                        and abs(self._rgb_t - self._depth_t) <= max_skew
                        and min(self._rgb_t, self._depth_t) > newer_than):
                    return Observation(self._rgb.copy(), self._depth.copy(), self._rgb_t)
            time.sleep(0.02)
        raise TimeoutError("no synchronised colour/depth frames — is the camera publishing?")

    def get_joint_positions(self, timeout: float = 5.0) -> np.ndarray:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if all(n in self._joints for n in self.joint_names):
                    return np.array([self._joints[n] for n in self.joint_names])
            time.sleep(0.02)
        raise TimeoutError(f"no joint_states for {self.joint_names}")

    def send_trajectory(self, traj: JointTrajectory, start_delay: float = 0.2):
        msg = JointTrajectoryMsg()
        msg.joint_names = traj.joint_names
        # zero stamp == "start now" for ros2_control; sim bridge treats it the same
        for q, t in zip(traj.positions, traj.times):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            pt.time_from_start = to_duration(float(t) + start_delay)
            msg.points.append(pt)
        self.traj_pub.publish(msg)

    def execute(self, traj: JointTrajectory, tol: float = 0.05, settle_timeout: float = 4.0) -> bool:
        """Send and block until the arm reaches the final point (or timeout).
        Uses the node clock, so it follows sim time in --sim mode."""
        self.send_trajectory(traj)
        t_end = self.now_s() + float(traj.times[-1]) + 0.2
        while self.now_s() < t_end:
            time.sleep(0.02)
        deadline = time.monotonic() + settle_timeout
        goal = traj.positions[-1]
        while time.monotonic() < deadline:
            err = np.max(np.abs(self.get_joint_positions() - goal))
            if err < tol:
                return True
            time.sleep(0.05)
        self.get_logger().warn(f"trajectory finished with joint error {err:.3f} rad "
                               "(blocked by clutter or torque-limited?)")
        return False

    def send_gripper(self, value: float, wait: float = 1.0):
        self.grip_pub.publish(Float64(data=float(value)))
        time.sleep(wait)

    def status(self, text: str):
        self.status_pub.publish(String(data=text))


# ----------------------------------------------------------------------------
class MoveItArm:
    """Optional IK backend: MoveIt 2 /compute_ik with the tool pointing down.
    Duck-types ArmModel (fk/ik) so the trajectory planner is backend-agnostic."""

    def __init__(self, node: Node, arm_model, group: str = "arm", ee_link: str = "joint6_flange",
                 base_frame: str = "base"):
        from moveit_msgs.srv import GetPositionIK

        self.node, self.arm, self.group, self.ee, self.base = node, arm_model, group, ee_link, base_frame
        self.joint_names = node.joint_names
        self.cli = node.create_client(GetPositionIK, "/compute_ik")
        if not self.cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_ik not available — is move_group running?")

    def fk(self, q):
        return self.arm.fk(q)

    def ik(self, p, q_seed, n_restarts: int = 0, **_):
        from moveit_msgs.srv import GetPositionIK
        from .kinematics import IKError

        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = self.group, self.ee, False
        r.robot_state.joint_state.name = self.joint_names
        r.robot_state.joint_state.position = [float(x) for x in q_seed]
        ps = r.pose_stamped
        ps.header.frame_id = self.base
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        ps.pose.orientation.x, ps.pose.orientation.w = 1.0, 0.0  # tool z pointing down
        r.timeout.nanosec = 50_000_000
        fut = self.cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < 1.0:
            time.sleep(0.005)
        res = fut.result() if fut.done() else None
        if res is None or res.error_code.val != 1:
            raise IKError(f"MoveIt IK failed for p={np.round(p, 3)}")
        js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return np.array([js[n] for n in self.joint_names])


def spin_in_background(node: Node):
    ex = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    return ex, th
