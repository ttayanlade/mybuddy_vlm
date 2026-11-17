#!/usr/bin/env python3
"""Isolate ONE 6-DoF arm from the dual-arm MyBuddy 280 URDF.

The MyBuddy description (Elephant Robotics `mycobot_ros`/`mycobot_ros2`, mybuddy
package) contains a torso/waist plus left and right myCobot-280-class arms. Isaac
Sim needs a single articulation rooted at the arm's shoulder, so we keep:

    root_link -> ... -> tip_link   (the kinematic chain)
  + every link/joint hanging below tip_link (flange, gripper, camera mounts)

and drop everything else. Optionally rename joints to the generic names used in
config/pipeline.yaml (joint1..joint6) and resolve ``package://`` mesh URIs to
absolute paths so Isaac Sim's importer can find them without a ROS environment.

Pure Python (xml.etree) — runs anywhere, no ROS or Isaac needed.

Example:
  python sim/extract_single_arm.py \
      --urdf ~/mycobot_ros2/mycobot_description/urdf/mybuddy/mybuddy.urdf \
      --root left_arm_base --tip left_arm_link6 \
      --package mycobot_description=~/mycobot_ros2/mycobot_description \
      --rename-joints auto \
      --out assets/urdf/mybuddy_left_arm.urdf

Use ``--list`` first to print the link/joint tree and pick root/tip names.
If you only have a myCobot 280 URDF (already single-arm, kinematically identical),
you can pass it with --root/--tip set to its base and flange links.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def parse_tree(robot: ET.Element):
    joints = {j.get("name"): j for j in robot.findall("joint")}
    links = {l.get("name"): l for l in robot.findall("link")}
    child_to_joint, children = {}, defaultdict(list)
    for name, j in joints.items():
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        child_to_joint[c] = name
        children[p].append(name)
    return links, joints, child_to_joint, children


def print_tree(robot: ET.Element):
    links, joints, child_to_joint, children = parse_tree(robot)
    roots = [l for l in links if l not in child_to_joint]

    def rec(link, depth):
        print("  " * depth + f"[link] {link}")
        for jn in children.get(link, []):
            j = joints[jn]
            print("  " * (depth + 1) + f"(joint {jn}: {j.get('type')})")
            rec(j.find("child").get("link"), depth + 2)

    for r in roots:
        rec(r, 0)


def chain(child_to_joint, joints, root, tip):
    path, link = [], tip
    while link != root:
        if link not in child_to_joint:
            raise SystemExit(f"'{tip}' is not a descendant of '{root}'")
        jn = child_to_joint[link]
        path.append(jn)
        link = joints[jn].find("parent").get("link")
    return list(reversed(path))


def descendants(children, joints, link):
    out = []
    for jn in children.get(link, []):
        out.append(jn)
        out += descendants(children, joints, joints[jn].find("child").get("link"))
    return out


def resolve_uri(uri: str, packages: dict[str, str], urdf_dir: Path) -> str:
    if uri.startswith("package://"):
        pkg, _, rest = uri[len("package://"):].partition("/")
        if pkg not in packages:
            raise SystemExit(f"mesh uses package '{pkg}' — pass --package {pkg}=/path/to/{pkg}")
        return str(Path(os.path.expanduser(packages[pkg])) / rest)
    if uri.startswith("file://"):
        return uri[len("file://"):]
    p = Path(uri)
    return str(p if p.is_absolute() else (urdf_dir / p).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--list", action="store_true", help="print the link/joint tree and exit")
    ap.add_argument("--root", help="first link of the arm (becomes the articulation base)")
    ap.add_argument("--tip", help="last arm link (6th joint's child / flange)")
    ap.add_argument("--package", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--rename-joints", default="none",
                    help="'none', 'auto' (revolute chain -> joint1..joint6), or old:new,old:new")
    ap.add_argument("--rename-root", default="base", help="rename root link (matches robot.base_link)")
    ap.add_argument("--out", default="assets/urdf/mybuddy_left_arm.urdf")
    args = ap.parse_args()

    urdf_path = Path(os.path.expanduser(args.urdf)).resolve()
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    if args.list:
        print_tree(robot)
        return
    if not (args.root and args.tip):
        ap.error("--root and --tip are required (use --list to inspect)")

    links, joints, child_to_joint, children = parse_tree(robot)
    arm_joints = chain(child_to_joint, joints, args.root, args.tip)
    revolute = [j for j in arm_joints if joints[j].get("type") in ("revolute", "continuous")]
    if len(revolute) != 6:
        print(f"WARNING: chain has {len(revolute)} actuated joints (expected 6): {revolute}", file=sys.stderr)
    tool_joints = descendants(children, joints, args.tip)
    keep_j = arm_joints + tool_joints
    keep_l = {args.root} | {joints[j].find("child").get("link") for j in keep_j}

    packages = dict(p.split("=", 1) for p in args.package)
    new = ET.Element("robot", {"name": robot.get("name", "mybuddy") + "_single_arm"})
    for m in robot.findall("material"):
        new.append(m)
    for ln in links:
        if ln in keep_l:
            el = links[ln]
            for mesh in el.iter("mesh"):
                mesh.set("filename", resolve_uri(mesh.get("filename"), packages, urdf_path.parent))
            new.append(el)
    for jn in keep_j:
        new.append(joints[jn])

    # renames ------------------------------------------------------------------
    jmap = {}
    if args.rename_joints == "auto":
        jmap = {old: f"joint{i + 1}" for i, old in enumerate(revolute)}
    elif args.rename_joints != "none":
        jmap = dict(kv.split(":") for kv in args.rename_joints.split(","))
    for j in new.findall("joint"):
        if j.get("name") in jmap:
            j.set("name", jmap[j.get("name")])
    if args.rename_root and args.rename_root != args.root:
        for el in new.iter():
            if el.tag == "link" and el.get("name") == args.root:
                el.set("name", args.rename_root)
            if el.tag in ("parent", "child") and el.get("link") == args.root:
                el.set("link", args.rename_root)
    # the shoulder link's origin in the torso is dropped: Isaac Lab places the base via init_state

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(new) if hasattr(ET, "indent") else None
    ET.ElementTree(new).write(out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out}: {len(keep_l)} links, {len(keep_j)} joints; actuated = "
          f"{[jmap.get(j, j) for j in revolute]}")
    print("Check that these match robot.joint_names / base_link / ee_link in config/pipeline.yaml.")


if __name__ == "__main__":
    main()
