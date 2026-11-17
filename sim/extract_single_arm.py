#!/usr/bin/env python3
"""Isolate ONE 6-DoF arm from the dual-arm MyBuddy 280 URDF and make it physics-ready.

The upstream MyBuddy description (elephantrobotics/mycobot_ros, mycobot_description/
urdf/mybuddy) is a visualisation model: base -> waist joint -> torso (link1) -> two
arms mounted sideways. For a single arm on a table we:

  * keep the chain  root -> ... -> tip  (+ anything hanging below tip);
  * optionally stand the arm upright (--mount-upright Z): the first arm joint's origin
    becomes (0, 0, Z) with no rotation, attached to a fresh `base` link;
  * rename actuated joints to joint1..joint6 (--rename-joints auto);
  * resolve package:// meshes, optionally as paths relative to the output file;
  * fix what the upstream URDF lacks for physics:
      --link-masses      add <inertial> (box inertia from each link's mesh bounds)
      --velocity-limit   upstream velocity="0" would freeze PhysX joints
      --effort-limit     upstream effort="1000" is not a 280-class servo
      --collision-from-visual   link4_L / link5_L collision origins disagree with visuals

Pure Python (xml.etree + numpy) — no ROS or Isaac needed.

MyBuddy left arm (what config/pipeline.yaml expects):
  python sim/extract_single_arm.py --urdf urdf/mybuddy/mybuddy.urdf \
      --root link1 --tip link6_L --mount-upright 0.062 --mount-yaw 3.14159 \
      --package mycobot_description=urdf/../ --relative-meshes --rename-joints auto \
      --collision-from-visual --velocity-limit 2.8 --effort-limit 1.5 \
      --link-masses 0.13,0.12,0.10,0.07,0.06,0.03 \
      --out assets/urdf/mybuddy_left_arm.urdf

Use --list first to print the link/joint tree.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_utils import mesh_bounds  # noqa: E402


# ----------------------------------------------------------------------------
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

    def rec(link, depth):
        print("  " * depth + f"[link] {link}")
        for jn in children.get(link, []):
            j = joints[jn]
            print("  " * (depth + 1) + f"(joint {jn}: {j.get('type')})")
            rec(j.find("child").get("link"), depth + 2)

    for r in (l for l in links if l not in child_to_joint):
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


def resolve_uri(uri: str, packages: dict[str, str], urdf_dir: Path) -> Path:
    if uri.startswith("package://"):
        pkg, _, rest = uri[len("package://"):].partition("/")
        if pkg not in packages:
            raise SystemExit(f"mesh uses package '{pkg}' — pass --package {pkg}=/path/to/{pkg}")
        base = Path(os.path.expanduser(packages[pkg]))
        return (base if base.is_absolute() else Path.cwd() / base).resolve() / rest
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    p = Path(uri)
    return p if p.is_absolute() else (urdf_dir / p).resolve()


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_of(el):
    o = el.find("origin") if el is not None else None
    xyz = np.array((o.get("xyz") or "0 0 0").split(), float) if o is not None else np.zeros(3)
    rpy = np.array((o.get("rpy") or "0 0 0").split(), float) if o is not None else np.zeros(3)
    return xyz, rpy


def link_box_in_link_frame(link: ET.Element):
    """AABB (link frame) of the visual mesh, applying the visual origin."""
    vis = link.find("visual")
    mesh = vis.find("geometry/mesh") if vis is not None else None
    if mesh is None:
        return None
    scale = np.array((mesh.get("scale") or "1 1 1").split(), float)
    lo, hi = mesh_bounds(mesh.get("_abs"), scale)
    xyz, rpy = origin_of(vis)
    R = rpy_to_matrix(*rpy)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = corners @ R.T + xyz
    return c.min(0), c.max(0)


def set_inertial(link: ET.Element, mass: float):
    box = link_box_in_link_frame(link)
    if box is None:
        com, size = np.zeros(3), np.full(3, 0.03)
    else:
        com, size = (box[0] + box[1]) / 2, np.maximum(box[1] - box[0], 0.005)
    sx, sy, sz = size
    ixx, iyy, izz = (mass / 12 * (sy ** 2 + sz ** 2), mass / 12 * (sx ** 2 + sz ** 2), mass / 12 * (sx ** 2 + sy ** 2))
    for old in link.findall("inertial"):
        link.remove(old)
    inn = ET.SubElement(link, "inertial")
    ET.SubElement(inn, "origin", {"xyz": " ".join(f"{v:.5f}" for v in com), "rpy": "0 0 0"})
    ET.SubElement(inn, "mass", {"value": f"{mass:.4f}"})
    ET.SubElement(inn, "inertia", {"ixx": f"{ixx:.3e}", "iyy": f"{iyy:.3e}", "izz": f"{izz:.3e}",
                                   "ixy": "0", "ixz": "0", "iyz": "0"})
    return com, size


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--list", action="store_true", help="print the link/joint tree and exit")
    ap.add_argument("--root", help="link the arm hangs from (its first joint is kept)")
    ap.add_argument("--tip", help="last arm link (6th joint's child / flange)")
    ap.add_argument("--package", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--relative-meshes", action="store_true", help="write mesh paths relative to --out")
    ap.add_argument("--rename-joints", default="none",
                    help="'none', 'auto' (revolute chain -> joint1..joint6), or old:new,old:new")
    ap.add_argument("--base-name", default="base", help="name of the new root link (robot.base_link)")
    ap.add_argument("--mount-upright", type=float, default=None, metavar="Z",
                    help="replace the first joint's origin with xyz=(0,0,Z), rpy=0 and drop the "
                         "root link's geometry (torso): the arm stands on the table")
    ap.add_argument("--mount-yaw", type=float, default=0.0,
                    help="yaw (rad) of the upright mount. MyBuddy's arm bends toward -x at q1=0, so "
                         "3.14159 makes q1=0 face +x (the workspace) and centres joint-1 travel on it")
    ap.add_argument("--collision-from-visual", action="store_true")
    ap.add_argument("--velocity-limit", type=float, default=None, help="rad/s for all actuated joints")
    ap.add_argument("--effort-limit", type=float, default=None, help="N·m for all actuated joints")
    ap.add_argument("--link-masses", default=None,
                    help="comma list of kg for the chain's child links in order (adds <inertial>)")
    ap.add_argument("--base-mass", type=float, default=0.2)
    ap.add_argument("--out", default="assets/urdf/mybuddy_left_arm.urdf")
    args = ap.parse_args()

    urdf_path = Path(os.path.expanduser(args.urdf)).resolve()
    robot = ET.parse(urdf_path).getroot()
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
    keep_j = arm_joints + descendants(children, joints, args.tip)
    chain_links = [joints[j].find("child").get("link") for j in arm_joints]
    keep_l = [args.root] + [joints[j].find("child").get("link") for j in keep_j]

    out = Path(args.out).resolve()
    packages = dict(p.split("=", 1) for p in args.package)
    new = ET.Element("robot", {"name": robot.get("name", "mybuddy") + "_single_arm"})
    new.append(ET.Comment(f" generated by sim/extract_single_arm.py from {urdf_path.name}: "
                          f"{args.root} -> {args.tip} "))
    new_links, new_joints = {}, {}
    for ln in keep_l:
        el = copy.deepcopy(links[ln])
        for mesh in el.iter("mesh"):
            p = resolve_uri(mesh.get("filename"), packages, urdf_path.parent)
            if not p.exists():
                raise SystemExit(f"mesh not found: {p}")
            mesh.set("_abs", str(p))
            mesh.set("filename", os.path.relpath(p, out.parent) if args.relative_meshes else str(p))
        new_links[ln] = el
    for jn in keep_j:
        new_joints[jn] = copy.deepcopy(joints[jn])

    # --- upright mount: fresh base, first joint origin straight up -------------
    root_el = new_links[args.root]
    if args.mount_upright is not None:
        for tag in ("visual", "collision", "inertial"):
            for e in root_el.findall(tag):
                root_el.remove(e)
        first = new_joints[arm_joints[0]]
        o = first.find("origin")
        if o is None:
            o = ET.SubElement(first, "origin")
        o.set("xyz", f"0 0 {args.mount_upright}")
        o.set("rpy", f"0 0 {args.mount_yaw}")

    # --- collision == visual ------------------------------------------------
    if args.collision_from_visual:
        for el in new_links.values():
            vis, col = el.find("visual"), el.find("collision")
            if vis is None or col is None:
                continue
            vo = vis.find("origin")
            co = col.find("origin")
            if co is not None:
                col.remove(co)
            if vo is not None:
                col.insert(0, copy.deepcopy(vo))

    # --- limits ---------------------------------------------------------------
    for jn in revolute:
        lim = new_joints[jn].find("limit")
        if lim is None:
            lim = ET.SubElement(new_joints[jn], "limit")
        if args.velocity_limit is not None:
            lim.set("velocity", str(args.velocity_limit))
        if args.effort_limit is not None:
            lim.set("effort", str(args.effort_limit))
        if float(lim.get("velocity", "0")) <= 0:
            print(f"WARNING: {jn} velocity limit is 0 — PhysX will not move it; pass --velocity-limit",
                  file=sys.stderr)

    # --- inertials --------------------------------------------------------------
    if args.link_masses:
        masses = [float(m) for m in args.link_masses.split(",")]
        if len(masses) != len(chain_links):
            raise SystemExit(f"--link-masses needs {len(chain_links)} values for {chain_links}")
        print("inertials (from mesh bounds):")
        for ln, m in zip(chain_links, masses):
            com, size = set_inertial(new_links[ln], m)
            print(f"  {ln:10s} m={m:.3f} kg  com={np.round(com, 3)}  box={np.round(size, 3)}")
        set_inertial(root_el, args.base_mass)
        print(f"  arm total {sum(masses):.3f} kg (+ base {args.base_mass} kg, fixed)")

    # --- renames ------------------------------------------------------------------
    jmap = {}
    if args.rename_joints == "auto":
        jmap = {old: f"joint{i + 1}" for i, old in enumerate(revolute)}
    elif args.rename_joints != "none":
        jmap = dict(kv.split(":") for kv in args.rename_joints.split(","))
    for jn, el in new_joints.items():
        el.set("name", jmap.get(jn, jn))
    if args.base_name and args.base_name != args.root:
        root_el.set("name", args.base_name)
        for el in new_joints.values():
            for tag in ("parent", "child"):
                t = el.find(tag)
                if t.get("link") == args.root:
                    t.set("link", args.base_name)

    for m in robot.findall("material"):
        new.append(copy.deepcopy(m))
    for el in new_links.values():
        for mesh in el.iter("mesh"):
            mesh.attrib.pop("_abs", None)
        new.append(el)
    for el in new_joints.values():
        new.append(el)

    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(new)
    ET.ElementTree(new).write(out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out}: {len(new_links)} links, {len(new_joints)} joints; actuated = "
          f"{[jmap.get(j, j) for j in revolute]}; ee link = {args.tip}")


if __name__ == "__main__":
    main()
