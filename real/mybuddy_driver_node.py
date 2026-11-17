#!/usr/bin/env python3
"""ROS 2 driver for ONE arm of the physical MyBuddy 280 (pymycobot over serial).

Exposes the same interface the Isaac Sim bridge exposes, so the reasoning node
is unchanged between --sim and --real:

  subscribes  mybuddy/joint_trajectory_controller/joint_cmd   trajectory_msgs/JointTrajectory
              mybuddy/gripper_cmd                             std_msgs/Float64 (0 closed .. 1 open)
  publishes   mybuddy/joint_states                            sensor_msgs/JointState (rad)

  python real/mybuddy_driver_node.py --port /dev/ttyACM0 --arm left

MyBuddy ids in pymycobot: 1 = left arm, 2 = right arm, 3 = waist.
Joint angles are streamed with send_radians(id, q, speed), exactly how Elephant's
own MyBuddy slider_control.py drives the arm from the same URDF, so URDF joint
angles == servo angles (real_joint_signs = 1, offsets = 0). Note our base frame is
the upstream joint-1 frame yawed by 180 deg (--mount-yaw), which only affects the
camera extrinsic you calibrate, not the joint angles.

Safety: trajectories are streamed at <= ``--rate`` Hz with send_angles(); the
servo's own speed limit (``--speed``, %) caps motion. Keep a hand on the e-stop
(release_all_servos) for first tests and start with --speed 20.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mybuddy_vlm.config import load_config  # noqa: E402

ARM_ID = {"left": 1, "right": 2}


class MyBuddyDriver(Node):
    def __init__(self, cfg: dict, port: str, baud: int, arm: str, speed: int, rate: float):
        prof = cfg["ros"]["profiles"]["real"]
        super().__init__("mybuddy_driver", namespace=prof["namespace"] or None)
        from pymycobot.mybuddy import MyBuddy

        self.mb = MyBuddy(port, baud)
        time.sleep(0.05)
        self.mb.set_fresh_mode(1)   # latest command wins (needed for streaming), as in Elephant's slider_control.py
        time.sleep(0.05)
        self.arm_id = ARM_ID[arm]
        self.speed, self.period = speed, 1.0 / rate
        r = cfg["robot"]
        self.joint_names = r["joint_names"]
        self.signs = np.array(r.get("real_joint_signs", [1] * 6), float)
        self.offs = np.radians(np.array(r.get("real_joint_offsets_deg", [0] * 6), float))
        self.lower, self.upper = np.array(r["joint_lower"]), np.array(r["joint_upper"])
        self.serial = threading.Lock()
        self._traj = None
        self._traj_lock = threading.Lock()
        self._last_q = None

        t = cfg["ros"]["topics"]
        self.pub_js = self.create_publisher(JointState, t["joint_states"], 10)
        self.create_subscription(JointTrajectory, t["joint_cmd"], self._on_traj, 10)
        self.create_subscription(Float64, t["gripper_cmd"], self._on_gripper, 10)
        self.create_timer(0.1, self._publish_state)          # 10 Hz state (serial is slow)
        threading.Thread(target=self._stream, daemon=True).start()
        self.get_logger().info(f"MyBuddy {arm} arm (id {self.arm_id}) on {port}")

    # model <-> servo -------------------------------------------------------
    def to_servo_deg(self, q):
        return list(np.degrees(self.to_servo_rad(q)).round(2))

    def to_servo_rad(self, q):
        # Elephant's own driver sends URDF radians unchanged (signs=1, offsets=0)
        return [float(x) for x in self.signs * (np.asarray(q) - self.offs)]

    def from_servo_deg(self, deg):
        return self.signs * np.radians(np.asarray(deg, float)) + self.offs

    # callbacks ---------------------------------------------------------------
    def _on_traj(self, msg: JointTrajectory):
        cols = {n: i for i, n in enumerate(msg.joint_names)}
        if any(n not in cols for n in self.joint_names) or not msg.points:
            self.get_logger().error("bad trajectory (missing joints or empty)")
            return
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
        pos = np.array([[p.positions[cols[n]] for n in self.joint_names] for p in msg.points])
        pos = np.clip(pos, self.lower, self.upper)
        with self._traj_lock:
            self._traj = (time.monotonic(), times, pos)

    def _on_gripper(self, msg: Float64):
        with self.serial:
            self.mb.set_gripper_value(self.arm_id, int(np.clip(msg.data, 0, 1) * 100), self.speed)

    def _publish_state(self):
        with self.serial:
            deg = self.mb.get_angles(self.arm_id)
        if not deg or len(deg) != 6:
            return  # serial hiccup; skip a cycle
        q = self.from_servo_deg(deg)
        self._last_q = q
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(self.joint_names)
        m.position = [float(x) for x in q]
        self.pub_js.publish(m)

    def _stream(self):
        while rclpy.ok():
            t_loop = time.monotonic()
            with self._traj_lock:
                tr = self._traj
            if tr is not None:
                t0, times, pos = tr
                t = time.monotonic() - t0
                if t >= times[-1]:
                    q = pos[-1]
                    with self._traj_lock:
                        self._traj = None
                else:
                    i = max(1, int(np.searchsorted(times, t)))
                    a = np.clip((t - times[i - 1]) / max(times[i] - times[i - 1], 1e-6), 0, 1)
                    q = pos[i - 1] + a * (pos[i] - pos[i - 1])
                with self.serial:
                    self.mb.send_radians(self.arm_id, self.to_servo_rad(q), self.speed)
            time.sleep(max(0.0, self.period - (time.monotonic() - t_loop)))

    def stop(self):
        with self.serial:
            self.mb.release_all_servos()  # NOTE: arm goes limp — support it before calling


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--speed", type=int, default=30, help="servo speed %% (1-100)")
    ap.add_argument("--rate", type=float, default=10.0, help="trajectory streaming rate (Hz)")
    args = ap.parse_args()
    rclpy.init()
    node = MyBuddyDriver(load_config(args.config), args.port, args.baud, args.arm, args.speed, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
