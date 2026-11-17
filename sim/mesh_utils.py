"""Mesh bounding boxes for COLLADA (.dae) and binary/ASCII STL, pure numpy.

Applies the COLLADA <unit>, the full node transform hierarchy and <instance_node>
references. The MyBuddy meshes need all three: joint5_L/joint6_L declare metres
but store millimetres under a 0.001 scale node, and joint4_L places its geometry
through library_nodes.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _node_tf(node, ns) -> np.ndarray:
    T = np.eye(4)
    for ch in node:
        tag = ch.tag.split("}")[1]
        if tag not in ("matrix", "translate", "scale", "rotate") or not ch.text:
            continue
        v = np.array(ch.text.split(), float)
        if tag == "matrix":
            M = v.reshape(4, 4)
        elif tag == "translate":
            M = np.eye(4)
            M[:3, 3] = v
        elif tag == "scale":
            M = np.diag([*v, 1.0])
        else:
            ax, a = v[:3] / np.linalg.norm(v[:3]), np.radians(v[3])
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            M = np.eye(4)
            M[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        T = T @ M
    return T


def dae_points(path: str | Path) -> np.ndarray:
    r = ET.parse(path).getroot()
    ns = {"c": r.tag.split("}")[0].strip("{")}
    u = r.find(".//c:asset/c:unit", ns)
    scale = float(u.get("meter")) if u is not None else 1.0
    geoms = {}
    for g in r.iterfind(".//c:library_geometries/c:geometry", ns):
        verts = g.find(".//c:mesh/c:vertices/c:input[@semantic='POSITION']", ns)
        src_id = verts.get("source").lstrip("#") if verts is not None else None
        for src in g.iterfind(".//c:mesh/c:source", ns):
            if src.get("id") == src_id or (src_id is None and "pos" in src.get("id", "").lower()):
                geoms[g.get("id")] = np.array(src.find("c:float_array", ns).text.split(), float).reshape(-1, 3)
                break
    lib_nodes = {n.get("id"): n for n in r.iterfind(".//c:library_nodes//c:node", ns)}
    pts = []

    def walk(node, T, depth=0):
        if depth > 32:
            return
        T = T @ _node_tf(node, ns)
        for ig in node.findall("c:instance_geometry", ns):
            a = geoms.get(ig.get("url").lstrip("#"))
            if a is not None:
                pts.append((np.c_[a, np.ones(len(a))] @ T.T)[:, :3])
        for inn in node.findall("c:instance_node", ns):
            ref = lib_nodes.get(inn.get("url").lstrip("#"))
            if ref is not None:
                walk(ref, T, depth + 1)
        for ch in node.findall("c:node", ns):
            walk(ch, T, depth + 1)

    for vs in r.iterfind(".//c:library_visual_scenes/c:visual_scene", ns):
        for n in vs.findall("c:node", ns):
            walk(n, np.eye(4))
    if not pts:
        raise ValueError(f"no geometry found in {path}")
    return np.vstack(pts) * scale


def stl_points(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        vals = [l.split()[1:4] for l in data.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        return np.array(vals, float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def mesh_bounds(path: str | Path, scale=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    p = str(path).lower()
    pts = dae_points(path) if p.endswith(".dae") else stl_points(path) if p.endswith(".stl") else None
    if pts is None:
        raise ValueError(f"unsupported mesh type: {path}")
    pts = pts * np.asarray(scale, float)
    return pts.min(0), pts.max(0)
