"""Contract 2: ``Detector.predict`` output keys, types, ranges and shapes (real weights, CPU)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import EXAMPLES

KEYS = {"prob_ai", "verdict", "confidence", "models", "heatmap", "spectrum", "explanation", "latency_ms"}


def check(out: dict, size: tuple[int, int]) -> None:
    from eval_contract import check_output  # the model agent's checker, reused as-is

    assert set(out) == KEYS
    assert check_output(out, size) == []
    assert isinstance(out["prob_ai"], float) and isinstance(out["confidence"], float)
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in out["models"].items())
    assert out["spectrum"].shape == (224, 224)
    if out["heatmap"] is not None:
        assert out["heatmap"].shape == (size[1], size[0])


@pytest.mark.parametrize("name", ["ai_veo_3_stone_villa.jpg", "real_pexels_shark.jpg"])
def test_predict_contract_on_examples(detector, name):
    im = Image.open(EXAMPLES / name).convert("RGB")
    out = detector.predict(im)
    check(out, im.size)
    assert out["heatmap"] is not None
    assert out["verdict"] == ("AI-generated" if name.startswith("ai_") else "Real")


def test_heatmap_maps_back_to_upload_size(detector):
    im = Image.open(EXAMPLES / "real_pexels_shark.jpg").convert("RGB").resize((360, 224))
    out = detector.predict(im)
    check(out, im.size)
    hm = out["heatmap"]
    left = (360 - 224) // 2
    assert float(hm[:, :left].max()) == 0.0 and float(hm[:, left + 224:].max()) == 0.0  # outside the analysed square


def test_tiny_and_rgba_inputs(detector):
    tiny = Image.open(EXAMPLES / "real_pexels_shark.jpg").convert("RGB").resize((120, 90))
    out = detector.predict(tiny)
    check(out, tiny.size)
    assert out["verdict"] == "Uncertain"  # decision.uncertain_if_upsampled

    rgba = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 300, 4), dtype=np.uint8), mode="RGBA")
    check(detector.predict(rgba), rgba.size)


def test_gallery_examples_behave_as_documented(detector, example_rows):
    """The demo gallery: regular examples are right; hard cases fail exactly as labelled in examples/README.md."""
    want = {"AI": "AI-generated", "Real": "Real"}
    for r in example_rows:
        out = detector.predict(Image.open(EXAMPLES / r["file"]).convert("RGB"))
        if r["hard_case"]:
            assert out["verdict"] != want[r["label"]], r["file"]
        else:
            assert out["verdict"] == want[r["label"]] and out["confidence"] >= 0.9, (r["file"], out["prob_ai"])
