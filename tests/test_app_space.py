"""app.py builds without launching; UI helpers render; build_space assembles a complete Space."""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import EXAMPLES, ROOT


@pytest.fixture(scope="module")
def app_module():
    os.environ["TRUTHLENS_FORCE_STUB"] = "1"  # hermetic and fast; the real detector is covered in test_contract
    try:
        spec = importlib.util.spec_from_file_location("truthlens_app", ROOT / "app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("TRUTHLENS_FORCE_STUB", None)
    return mod


def test_app_builds_blocks(app_module):
    import gradio as gr

    assert isinstance(app_module.demo, gr.Blocks)
    assert app_module.IS_STUB
    assert len(app_module.EXAMPLES) == 12 and len(app_module.HARD_EXAMPLES) == 2
    assert app_module.BAND == (0.35, 0.65)


def test_analyze_handler_on_example(app_module):
    path = str(EXAMPLES / "ai_veo_3_stone_villa.jpg")
    card, overlay, spec, state, _ = app_module.analyze(path, 0.5)
    assert "tl-badge-title" in card and "Ground truth" in card and "Veo 3" in card
    assert overlay.shape == (224, 224, 3) and spec.shape[2] == 3 and state["box"] == (0, 0, 224, 224)
    assert app_module.analyze(None, 0.5)[0] == app_module.ui.empty_card()


def test_metrics_page_renders_real_schema():
    from truthlens import ui

    page = ui.metrics_html(ROOT / "reports" / "metrics.json")
    assert "never seen in training" in page and "Robustness" in page and "Hardest" in page
    assert "Metrics coming soon" in ui.metrics_html(ROOT / "nope.json")


def test_pretty_names():
    from truthlens.ui import pretty_source

    assert pretty_source("seedream_v4_5") == "Seedream v4.5"
    assert pretty_source("flux_2_klein_9b") == "FLUX.2 Klein 9B"
    assert pretty_source("gpt_image_1_5") == "GPT Image 1.5"


def test_build_space(tmp_path):
    from build_space import build

    out = tmp_path / "space"
    assert build(out) == 0
    for rel in ["app.py", "requirements.txt", "README.md", ".gitattributes", "src/truthlens/inference.py",
                "src/truthlens/ui.py", "assets/style.css", "examples/labels.csv", "reports/metrics.json",
                "models/config.json", "models/config_fast.json", "models/cnn_efficientnet_b0.safetensors",
                "models/clip_head.npz", "models/clip_vitb16_multi_head.npz"]:
        assert (out / rel).is_file(), rel
    assert not (out / "models" / "cache").exists()
    assert not list(out.rglob("__pycache__"))
    readme = (out / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: gradio" in readme and "app_file: app.py" in readme
    assert "timm/vit_large_patch14_clip_224.openai open_clip_model.safetensors" in readme
    attrs = (out / ".gitattributes").read_text()
    assert "*.safetensors filter=lfs" in attrs and "*.npz filter=lfs" in attrs
