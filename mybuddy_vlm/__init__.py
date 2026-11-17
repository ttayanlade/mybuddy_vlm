"""MyBuddy single-arm zero-shot VLM search pipeline.

Pure-Python core (no Isaac Sim / ROS imports at module level) so the same
reasoning, grounding and motion-primitive code runs against simulation and
hardware. ROS 2 is only touched in ``ros_interface`` and ``reasoning_node``.
"""

__version__ = "0.1.0"
