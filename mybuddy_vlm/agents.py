"""Decision makers: the VLM spatial reasoner, plus a geometric heuristic baseline.

Both look at one RGB image of the plant with the target fruit's region boxed (Find the
Fruit also gives the policy the target location as a mask) and return a Decision:
a point ON the plant to push and an image-space direction to push it.

The VLM client speaks the OpenAI chat-completions protocol over plain HTTP (urllib),
so it runs inside Isaac Sim's Python with no extra packages; the server is vLLM
(open-source Qwen3-VL by default) or OpenAI.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)
DIRECTIONS = ("left", "right", "up", "down")


@dataclass
class Decision:
    point_px: tuple[float, float]          # pixel in the image the agent saw
    direction: str                         # left | right | up | down  (image space)
    reasoning: str = ""
    raw: str = field(default="", repr=False)


class DecisionError(ValueError):
    pass


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def annotate_target(rgb: np.ndarray, box_xyxy, color=(255, 230, 0), width=3) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb[..., :3].astype(np.uint8))
    ImageDraw.Draw(img).rectangle([float(v) for v in box_xyxy], outline=color, width=width)
    return np.asarray(img)


def draw_decision(rgb: np.ndarray, d: Decision, note: str = "") -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb[..., :3].astype(np.uint8))
    dr = ImageDraw.Draw(img)
    u, v = d.point_px
    du, dv = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[d.direction]
    L = 0.12 * img.width
    dr.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(0, 255, 255), width=3)
    dr.line([u, v, u + du * L, v + dv * L], fill=(0, 255, 255), width=4)
    dr.text((8, 8), f"{d.direction} @ ({u:.0f},{v:.0f}) {note}", fill=(255, 255, 255))
    return np.asarray(img)


def _jpeg_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------
# VLM
# ----------------------------------------------------------------------------
SYSTEM = ("You are the high-level spatial reasoner for a MyBuddy robot arm working on a plant. "
          "A separate controller executes your choice. Reply with one JSON object only.")

PROMPT = """This image is from a camera looking at a plant. A {color} fruit is hidden behind the plant's \
stems and leaves inside the YELLOW BOX. The robot arm can push one part of the plant in one direction.

Choose the single plant part (stem or leaf) that most blocks the view of the boxed fruit, give a point \
ON that part, and the direction to push it so the fruit becomes visible.

Rules:
- The point must lie on the plant itself (not on the fruit, the background, or the robot).
- Never push into a fruit: the arm must not touch any fruit.
- Directions are as seen in this image: left, right, up, down.
- Coordinates: {coord_rule}
{history}
Respond with JSON: {{"point": [x, y], "direction": "left|right|up|down", "reasoning": "<one short sentence>"}}"""

COORD_RULES = {
    "norm1000": "x and y are integers from 0 to 1000, relative to the image width and height (0,0 = top-left).",
    "pixels": "x and y are pixel coordinates in this {w}x{h} image (0,0 = top-left).",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "direction": {"type": "string", "enum": list(DIRECTIONS)},
        "reasoning": {"type": "string"},
    },
    "required": ["point", "direction", "reasoning"],
    "additionalProperties": False,
}

THINK_RE = re.compile(r"(?:<think>.*?</think>|^.*?</think>)", re.S)


def parse_decision(text: str, width: int, height: int, point_format: str) -> Decision:
    s = THINK_RE.sub("", text).strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise DecisionError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict) or not {"point", "direction"} <= obj.keys():
        raise DecisionError("need keys 'point' and 'direction'")
    pt, direction = obj["point"], str(obj["direction"]).strip().lower()
    if direction not in DIRECTIONS:
        raise DecisionError(f"direction must be one of {DIRECTIONS}")
    if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
        raise DecisionError("point must be [x, y] numbers")
    x, y = float(pt[0]), float(pt[1])
    if point_format == "norm1000":
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise DecisionError("point must be within [0, 1000]")
        x, y = x / 1000 * width, y / 1000 * height
    elif not (0 <= x < width and 0 <= y < height):
        raise DecisionError(f"point must be inside the {width}x{height} image")
    return Decision((x, y), direction, str(obj.get("reasoning", "")).strip(), raw=text)


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Previous actions this episode (the image shows the result):"]
    for i, h in enumerate(history[-4:], 1):
        lines.append(f"  {i}. push {h['direction']} at {h['point_shown']} -> {h['outcome']}")
    return "\n".join(lines) + "\n"


class VLMAgent:
    name = "vlm"

    def __init__(self, cfg: dict):
        self.c = cfg
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.key = os.environ.get(cfg.get("api_key_env") or "OPENAI_API_KEY", "EMPTY")

    def _post(self, payload: dict) -> str:
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.c.get("timeout_s", 120)) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""

    def decide(self, rgb_annotated: np.ndarray, fruit_color: str, history: list[dict]) -> Decision:
        h, w = rgb_annotated.shape[:2]
        fmt = self.c.get("point_format", "norm1000")
        text = PROMPT.format(color=fruit_color, coord_rule=COORD_RULES[fmt].format(w=w, h=h),
                             history=format_history(history))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _jpeg_b64(rgb_annotated)}}]}]
        payload = {"model": self.c["model"], "messages": messages, "temperature": self.c.get("temperature", 0.2),
                   "max_tokens": self.c.get("max_tokens", 400)}
        if self.c.get("structured_output", "json_schema") == "json_schema":
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "decision", "schema": SCHEMA, "strict": True}}
        err = None
        for attempt in range(self.c.get("max_retries", 2) + 1):
            t0 = time.time()
            reply = self._post(payload)
            try:
                d = parse_decision(reply, w, h, fmt)
                log.info("VLM (%.1fs): push %s at (%.0f,%.0f) — %s", time.time() - t0, d.direction, *d.point_px, d.reasoning)
                return d
            except DecisionError as e:
                err = e
                log.warning("VLM reply rejected: %s | %r", e, reply[:200])
                payload["messages"] = messages + [{"role": "assistant", "content": reply},
                                                  {"role": "user", "content": f"Invalid: {e}. Reply with only the JSON."}]
        raise DecisionError(f"VLM gave no valid decision: {err}")


# ----------------------------------------------------------------------------
# Baseline (no VLM): push the plant part nearest the target sideways, away from the target
# ----------------------------------------------------------------------------
class HeuristicAgent:
    """Picks the plant pixel nearest the target box centre and pushes it horizontally
    away from the box centre (toward the emptier side if it sits dead centre). Same interface as VLMAgent; used to
    test the loop without a model and as a comparison baseline."""

    name = "heuristic"

    def __init__(self, *_):
        self.plant = None
        self.box = None

    def set_context(self, plant_mask: np.ndarray, box_xyxy):
        self.plant, self.box = plant_mask, box_xyxy

    def decide(self, rgb_annotated: np.ndarray, fruit_color: str, history: list[dict]) -> Decision:
        x0, y0, x1, y1 = self.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vs, us = np.nonzero(self.plant)
        if len(us) == 0:
            return Decision((cx, cy), "left", "no plant pixels visible")
        blocked = [(h["point_shown"], h["direction"]) for h in history if h.get("blocked")]

        def refused(u, v, d):                     # never repeat a push the planner already refused
            return any(dd == d and abs(p[0] - u) <= 10 and abs(p[1] - v) <= 10 for p, dd in blocked)

        order = np.argsort((us - cx) ** 2 + (vs - cy) ** 2)
        u = v = direction = None
        for i in order[:5000]:
            uu, vv = float(us[i]), float(vs[i])
            if abs(uu - cx) > 2:                  # push the occluder away from the box centre
                first = "left" if uu < cx else "right"
            else:                                 # dead centre: toward the side with less plant
                win = self.plant[int(max(0, y0)):int(y1), :]
                first = "left" if win[:, :int(cx)].sum() < win[:, int(cx):].sum() else "right"
            for d in (first, "right" if first == "left" else "left"):
                if not refused(uu, vv, d):
                    u, v, direction = uu, vv, d
                    break
            if direction:
                break
        if direction is None:
            i = order[0]
            u, v, direction = float(us[i]), float(vs[i]), "left"
        return Decision((u, v), direction, f"nearest plant pixel to the target; push it {direction}, away from it")
