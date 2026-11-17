"""Step 2 — VLM spatial reasoner.

Works with OpenAI GPT-4o or any OpenAI-compatible server (local VILA served by
vLLM / NVIDIA NIM: set ``vlm.provider: openai_compatible`` and ``vlm.base_url``).
The response is validated against a strict schema; on a malformed reply the
validation error is fed back to the model and the call is retried.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

PRIMITIVES = ("push_left", "push_right", "lift", "pull")

SYSTEM_PROMPT = (
    "You are the high-level spatial reasoner of a MyBuddy 280 robot arm "
    "(6-DoF, ~28 cm reach, 250 g payload) mounted at the edge of a table. "
    "You only choose WHAT to move and HOW; a separate perception and control "
    "stack executes it. Reply with a single JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = (
    "You are a MyBuddy robot arm tasked with finding {target}. Analyze this image "
    "of a cluttered table. The target is hidden. Identify the most logical occluding "
    "object to move. Output your response in strictly formatted JSON containing: "
    '{{"target_obstacle": "<description>", '
    '"primitive_action": "<push_left | push_right | lift | pull>", '
    '"reasoning": "<brief explanation>"}}\n\n'
    "Rules:\n"
    "- \"target_obstacle\" must be a short visual noun phrase an open-vocabulary "
    "detector can find (e.g. \"blue towel\", \"white mug\"), not a sentence.\n"
    "- \"left\"/\"right\" are as seen in THIS image.\n"
    "- Prefer actions that uncover the most hidden area without knocking objects off the table.\n"
    "{history}"
)


@dataclass
class VLMDecision:
    target_obstacle: str
    primitive_action: str
    reasoning: str
    raw: str = field(default="", repr=False)


class VLMResponseError(ValueError):
    pass


def parse_decision(text: str) -> VLMDecision:
    """Strictly validate the VLM's JSON. Tolerates ```json fences (common with
    local models) but nothing else."""
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if fence:
        s = fence.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise VLMResponseError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise VLMResponseError("top-level JSON must be an object")
    required = {"target_obstacle", "primitive_action", "reasoning"}
    missing = required - obj.keys()
    if missing:
        raise VLMResponseError(f"missing keys: {sorted(missing)}")
    extra = obj.keys() - required
    if extra:
        raise VLMResponseError(f"unexpected keys: {sorted(extra)}")
    for k in required:
        if not isinstance(obj[k], str) or not obj[k].strip():
            raise VLMResponseError(f"'{k}' must be a non-empty string")
    action = obj["primitive_action"].strip().lower()
    if action not in PRIMITIVES:
        raise VLMResponseError(f"primitive_action '{action}' not in {PRIMITIVES}")
    return VLMDecision(obj["target_obstacle"].strip(), action, obj["reasoning"].strip(), raw=text)


def encode_image_jpeg_b64(rgb: np.ndarray, max_side: int = 640) -> str:
    from PIL import Image
    import io

    img = Image.fromarray(rgb[..., :3].astype(np.uint8))
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["\nPrevious attempts this episode (avoid repeating failures):"]
    for i, h in enumerate(history[-5:], 1):
        lines.append(f"  {i}. {h['primitive_action']} on '{h['target_obstacle']}' -> {h['outcome']}")
    return "\n".join(lines) + "\n"


class VLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        from openai import OpenAI

        api_key = os.environ.get(cfg.get("api_key_env") or "OPENAI_API_KEY")
        if cfg["provider"] == "openai_compatible":
            api_key = api_key or "EMPTY"  # vLLM / NIM usually ignore it
        self.client = OpenAI(api_key=api_key, base_url=cfg.get("base_url"),
                             timeout=cfg.get("timeout_s", 30))

    def decide(self, rgb: np.ndarray, target: str, history: list[dict] | None = None) -> VLMDecision:
        img_b64 = encode_image_jpeg_b64(rgb, self.cfg.get("image_max_side", 640))
        prompt = USER_PROMPT_TEMPLATE.format(target=target, history=format_history(history or []))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]},
        ]
        kwargs = dict(model=self.cfg["model"], messages=messages,
                      temperature=self.cfg.get("temperature", 0.2),
                      max_tokens=self.cfg.get("max_tokens", 300))
        if self.cfg["provider"] == "openai":
            kwargs["response_format"] = {"type": "json_object"}  # GPT-4o JSON mode

        last_err = None
        for attempt in range(self.cfg.get("max_retries", 2) + 1):
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            try:
                d = parse_decision(text)
                log.info("VLM: %s on '%s' — %s", d.primitive_action, d.target_obstacle, d.reasoning)
                return d
            except VLMResponseError as e:
                last_err = e
                log.warning("VLM reply rejected (attempt %d): %s | %r", attempt + 1, e, text[:200])
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Invalid response: {e}. Reply again with ONLY the JSON object."},
                ]
                kwargs["messages"] = messages
        raise VLMResponseError(f"VLM failed to produce valid JSON: {last_err}")
