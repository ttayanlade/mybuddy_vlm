"""VLM-driven occluded-fruit search on the MyBuddy (Find the Fruit task, no RL).

Pure-Python core (numpy; pxr only for loading the robot USD) so it runs both inside
Isaac Sim and on a plain CPU for tests.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    import os

    import yaml

    with open(path or PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for env, key in (("VLM_BASE_URL", "base_url"), ("VLM_MODEL", "model"),
                     ("VLM_STRUCTURED_OUTPUT", "structured_output"), ("VLM_POINT_FORMAT", "point_format")):
        if os.environ.get(env):
            cfg["vlm"][key] = os.environ[env]
    return cfg


def project_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
